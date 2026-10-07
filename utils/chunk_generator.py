"""
utils/chunk_generator.py
Logique de génération des chunks JSON pour le RAG.
Fonctions pures uniquement — aucun widget Streamlit.
"""
import io
import json
import math
import re
import zipfile
from datetime import datetime
from typing import Any
import unicodedata

import pandas as pd
import numpy as np

from utils.profiler import safe_fillna


# ══════════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════════

def safe_filename(text: str, max_len: int = 80) -> str:
    """
    Transforme une valeur quelconque en nom de fichier sûr :
    - Normalise les accents (é→e, ç→c, etc.)
    - Remplace les caractères non-ASCII par _
    - Remplace les espaces et | par _
    - Supprime les _ multiples
    - Limite la longueur
    """
    text = str(text)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text
                   if unicodedata.category(c) != "Mn")
    text = re.sub(r"[\s|/\\]+", "_", text)
    text = re.sub(r"[^\w\-]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] or "segment"


# ══════════════════════════════════════════════════════════════════
# COLONNE FUSIONNÉE
# ══════════════════════════════════════════════════════════════════

def build_fusion_column(
    df: pd.DataFrame,
    cols: list,
    template: str,
) -> pd.Series:
    """
    Remplace les {NomColonne} dans le template par les vraies valeurs
    de chaque ligne. Les valeurs null/NaN sont remplacées par "N/A".

    Exemple :
      template = "Produit {Denomination} famille {Grande_Famille}"
      → Series avec une valeur par ligne
    """
    result = []
    for _, row in df.iterrows():
        text = template
        for col in cols:
            val = row.get(col, "")
            try:
                is_na = pd.isna(val)
            except Exception:
                is_na = False
            if is_na:
                val = "N/A"
            else:
                val = str(val)
            text = text.replace(f"{{{col}}}", val)
        result.append(text)
    return pd.Series(result, index=df.index)


# ══════════════════════════════════════════════════════════════════
# VALIDATION DU TEMPLATE JSON
# ══════════════════════════════════════════════════════════════════

def validate_json_template(
    template_str: str,
    available_cols: list,
) -> dict:
    """
    Valide le template JSON saisi par l'utilisateur.

    Retourne :
    {
      "valid": bool,
      "error_msg": str,
      "error_line": int,        # -1 si N/A
      "placeholders_ok": [...], # {col} trouvés et valides
      "placeholders_bad": [...] # {col} introuvables
    }
    """
    # Remplacer temporairement {col} par une chaîne valide.
    # Ordre important : d'abord "{col}" (placeholder déjà entouré de
    # guillemets JSON) → "__PH__", puis {col} nu → "__PH__".
    # Sans cet ordre on obtient ""__PH__"" (double guillemets) = JSON invalide.
    temp = re.sub(r'"\{[^{}]+\}"', '"__PH__"', template_str)
    temp = re.sub(r"\{[^{}]+\}",   '"__PH__"', temp)
    try:
        json.loads(temp)
    except json.JSONDecodeError as e:
        error_line = e.lineno if hasattr(e, "lineno") else -1
        return {
            "valid": False,
            "error_msg": (
                f"JSON invalide ligne {error_line} : {e.msg}"
                if error_line > 0
                else f"JSON invalide : {e.msg}"
            ),
            "error_line": error_line,
            "placeholders_ok":  [],
            "placeholders_bad": [],
        }

    found = re.findall(r"\{([^{}]+)\}", template_str)
    placeholders_ok  = list(set(p for p in found if p in available_cols))
    placeholders_bad = list(set(p for p in found if p not in available_cols))

    return {
        "valid":            True,
        "error_msg":        "",
        "error_line":       -1,
        "placeholders_ok":  placeholders_ok,
        "placeholders_bad": placeholders_bad,
    }


# ══════════════════════════════════════════════════════════════════
# REMPLISSAGE DU TEMPLATE
# ══════════════════════════════════════════════════════════════════

def fill_template_row(
    template_str: str,
    row: pd.Series,
    df_dtypes: dict,
) -> Any:
    """
    Remplace chaque {NomColonne} par la valeur réelle de la ligne.
    - Valeur manquante (NaN/None) → null JSON
    - Valeur numérique → nombre brut (sans guillemets)
    - Valeur texte → string avec échappement JSON

    Retourne l'objet Python parsé depuis le JSON rempli.
    """
    filled = template_str
    placeholders = re.findall(r"\{([^{}]+)\}", template_str)

    for col in placeholders:
        if col not in row.index:
            filled = re.sub(
                r'"?\{' + re.escape(col) + r'\}"?',
                "null",
                filled,
            )
            continue

        val = row[col]

        # Valeur manquante → null
        is_na = False
        try:
            is_na = pd.isna(val)
        except Exception:
            pass

        if is_na:
            filled = re.sub(
                r'"?\{' + re.escape(col) + r'\}"?',
                "null",
                filled,
            )
            continue

        # Détecter si numérique
        is_numeric = False
        try:
            is_numeric = pd.api.types.is_numeric_dtype(
                pd.Series([val])
            )
        except Exception:
            pass

        if is_numeric:
            num_val = val.item() if hasattr(val, "item") else val
            # Remplacer avec guillemets d'abord, puis sans
            filled = re.sub(
                r'"\{' + re.escape(col) + r'\}"',
                str(num_val),
                filled,
            )
            filled = filled.replace(f"{{{col}}}", str(num_val))
        else:
            str_val = str(val).replace("\\", "\\\\").replace('"', '\\"')
            filled = filled.replace(f"{{{col}}}", str_val)

    try:
        return json.loads(filled)
    except json.JSONDecodeError:
        return filled


# ══════════════════════════════════════════════════════════════════
# GÉNÉRATION D'UN SEGMENT
# ══════════════════════════════════════════════════════════════════

def generate_segment_json(
    segment_df: pd.DataFrame,
    template_str: str,
    df_dtypes: dict,
) -> list:
    """
    Applique fill_template_row à chaque ligne du segment.
    Retourne une liste Python (tableau JSON).
    """
    results = []
    for _, row in segment_df.iterrows():
        try:
            obj = fill_template_row(template_str, row, df_dtypes)
            results.append(obj)
        except Exception:
            results.append(None)
    return results


# ══════════════════════════════════════════════════════════════════
# GÉNÉRATION DU ZIP COMPLET
# ══════════════════════════════════════════════════════════════════

def generate_zip(
    df: pd.DataFrame,
    chunk_cols: list,
    template_str: str,
    filename_source: str = "données",
) -> bytes:
    """
    Génère un ZIP contenant :
      - 1 JSON par segment (tableau de lignes)
      - index.json avec métadonnées complètes
      - README.txt avec résumé

    Utilise safe_fillna avant groupby pour les colonnes category.
    """
    valid_cols = [c for c in chunk_cols if c in df.columns]
    if not valid_cols:
        return b""

    df_dtypes = {col: str(df[col].dtype) for col in df.columns}

    # safe_fillna obligatoire avant tout groupby
    df_safe = safe_fillna(df[valid_cols])
    grouped = df_safe.groupby(valid_cols)

    zip_buffer = io.BytesIO()
    index_data = {
        "date_generation":   datetime.now().strftime("%d/%m/%Y %H:%M"),
        "fichier_source":    filename_source,
        "colonnes_chunking": valid_cols,
        "nb_segments":       grouped.ngroups,
        "segments":          [],
    }

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:

        for key, _ in grouped:
            # Construire les valeurs de segmentation
            if isinstance(key, tuple):
                label_parts = [str(k) for k in key]
                seg_vals    = {c: str(k) for c, k in zip(valid_cols, key)}
            else:
                label_parts = [str(key)]
                seg_vals    = {valid_cols[0]: str(key)}

            label = "_".join(safe_filename(p) for p in label_parts)

            # Récupérer les lignes du df original (pas df_safe)
            mask = pd.Series([True] * len(df), index=df.index)
            for col, val in seg_vals.items():
                mask &= df_safe[col].astype(str) == val
            seg_df = df[mask]

            if seg_df.empty:
                continue

            # Générer le JSON du segment
            seg_json = generate_segment_json(seg_df, template_str, df_dtypes)

            fname = f"{label}.json"
            try:
                zf.writestr(
                    fname,
                    json.dumps(seg_json, ensure_ascii=False, indent=2),
                )
            except Exception:
                continue

            index_data["segments"].append({
                "nom":               label,
                "fichier":           fname,
                "nb_lignes":         len(seg_df),
                "pct_total":         round(len(seg_df) / len(df) * 100, 2)
                                     if len(df) > 0 else 0,
                "valeurs_chunking":  seg_vals,
            })

        # index.json
        zf.writestr(
            "index.json",
            json.dumps(index_data, ensure_ascii=False, indent=2),
        )

        # README.txt
        n_seg    = grouped.ngroups
        avg_size = len(df) // n_seg if n_seg > 0 else 0
        readme   = (
            "RAG Chunking — Rapport de génération\n"
            "=====================================\n"
            f"Fichier source    : {filename_source}\n"
            f"Date              : {index_data['date_generation']}\n"
            f"Colonnes chunking : {' × '.join(valid_cols)}\n"
            f"Nb segments       : {n_seg}\n"
            f"Taille moyenne    : ~{avg_size} lignes/segment\n"
            f"Nb lignes total   : {len(df):,}\n"
            "\n"
            "Template JSON utilisé :\n"
            f"{template_str}\n"
        )
        zf.writestr("README.txt", readme)

    return zip_buffer.getvalue()


# ══════════════════════════════════════════════════════════════════
# PRÉVISUALISATION DES SEGMENTS
# ══════════════════════════════════════════════════════════════════

def compute_segments_preview(
    df: pd.DataFrame,
    chunk_cols: list,
    lignes_par_page: float,
) -> list:
    """
    Calcule les métadonnées de tous les segments sans générer les fichiers JSON.

    Retourne une liste de dicts triée par nb_lignes décroissant :
    {
      "label":      str,   # nom du fichier sans .json
      "fname":      str,   # nom avec .json
      "nb_lignes":  int,
      "nb_pages":   int,
      "depasse":    bool,  # mis à jour côté page selon seuil
      "seg_vals":   dict,  # {col: val} pour filtrage
    }
    """
    from utils.rag_scorer import estimate_pages

    valid_cols = [c for c in chunk_cols if c in df.columns]
    if not valid_cols:
        return []

    try:
        df_safe = safe_fillna(df[valid_cols])
        grouped = df_safe.groupby(valid_cols)
    except Exception:
        return []

    results = []
    for key, grp in grouped:
        try:
            if isinstance(key, tuple):
                label_parts = [str(k) for k in key]
                seg_vals    = {c: str(k) for c, k in zip(valid_cols, key)}
            else:
                label_parts = [str(key)]
                seg_vals    = {valid_cols[0]: str(key)}

            label    = "_".join(safe_filename(p) for p in label_parts)
            nb_lignes = len(grp)
            nb_pages, _ = estimate_pages(nb_lignes, lignes_par_page)

            results.append({
                "label":     label,
                "fname":     f"{label}.json",
                "nb_lignes": nb_lignes,
                "nb_pages":  nb_pages,
                "depasse":   False,
                "seg_vals":  seg_vals,
            })
        except Exception:
            pass

    results.sort(key=lambda x: x["nb_lignes"], reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════
# DÉCOUPE PAR 3ÈME COLONNE
# ══════════════════════════════════════════════════════════════════

def split_segment_by_column(
    seg_df: pd.DataFrame,
    split_col: str,
    label: str,
    template_str: str,
    df_dtypes: dict,
    lignes_par_page: float,
) -> list:
    """
    Découpe un segment en sous-segments selon les valeurs d'une 3ème colonne.

    Retourne une liste de dicts triée par nb_lignes décroissant :
    [
      {
        "label":     str,   # label_SousValeur
        "fname":     str,   # label_SousValeur.json
        "nb_lignes": int,
        "nb_pages":  int,
        "json_data": list,  # liste d'objets JSON générés
        "split_val": str,   # valeur de split_col
      },
    ]
    """
    from utils.rag_scorer import estimate_pages

    if split_col not in seg_df.columns:
        return []

    results = []
    try:
        df_safe_split = safe_fillna(seg_df[[split_col]])
        safe_series   = df_safe_split[split_col].astype(str)
        unique_vals   = safe_series.unique()
    except Exception:
        return []

    for val in unique_vals:
        try:
            val_str = str(val)
            mask    = safe_series == val_str
            sub_df  = seg_df.loc[mask[mask].index]
            if len(sub_df) == 0:
                continue

            sub_label = f"{label}_{safe_filename(val_str)}"
            sub_fname = f"{sub_label}.json"
            nb_lignes = len(sub_df)
            nb_pages, _ = estimate_pages(nb_lignes, lignes_par_page)

            sub_json = generate_segment_json(sub_df, template_str, df_dtypes)

            results.append({
                "label":     sub_label,
                "fname":     sub_fname,
                "nb_lignes": nb_lignes,
                "nb_pages":  nb_pages,
                "json_data": sub_json,
                "split_val": val_str,
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["nb_lignes"], reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════
# FUSION AUTOMATIQUE DES PETITS SEGMENTS
# ══════════════════════════════════════════════════════════════════

def compute_auto_merge(
    df: pd.DataFrame,
    chunk_cols: list,
    col_to_remove: str,
    seuil_min_pages: float,
    lignes_par_page: float,
) -> list:
    """
    Calcule la fusion automatique des petits segments après suppression
    d'une colonne de filtre.

    Logique :
    1. Calculer les segments avec chunk_cols complet
    2. Identifier les petits segments (< seuil_min_pages)
    3. Pour ces petits segments, regrouper selon chunk_cols sans col_to_remove
    4. Retourner seulement les regroupements avec 2+ segments d'origine

    Retourne :
    [
      {
        "nouveau_label":    str,
        "segments_origine": list[str],
        "nb_lignes":        int,
        "nb_pages":         int,
      }
    ]
    """
    from utils.rag_scorer import estimate_pages

    valid_cols = [c for c in chunk_cols if c in df.columns]
    if col_to_remove not in valid_cols:
        return []

    reduced_cols = [c for c in valid_cols if c != col_to_remove]
    if not reduced_cols:
        return []

    try:
        df_safe = safe_fillna(df[valid_cols])
        grouped_full = df_safe.groupby(valid_cols)
    except Exception:
        return []

    # Identifier les petits segments
    small_segments = []
    for key, grp in grouped_full:
        nb_lignes = len(grp)
        nb_pages, _ = estimate_pages(nb_lignes, lignes_par_page)
        if nb_pages < seuil_min_pages:
            if isinstance(key, tuple):
                label_parts = [str(k) for k in key]
            else:
                label_parts = [str(key)]
            label = "_".join(safe_filename(p) for p in label_parts)
            small_segments.append({
                "label":     label,
                "key":       key,
                "nb_lignes": nb_lignes,
            })

    if not small_segments:
        return []

    # Regrouper les petits segments par colonnes réduites
    merges: dict = {}
    try:
        for seg_info in small_segments:
            key = seg_info["key"]
            if not isinstance(key, tuple):
                key = (key,)

            reduced_vals = tuple(
                str(k) for c, k in zip(valid_cols, key)
                if c != col_to_remove
            )
            reduced_label = "_".join(safe_filename(v) for v in reduced_vals)

            if reduced_label not in merges:
                merges[reduced_label] = {
                    "nouveau_label":    reduced_label,
                    "segments_origine": [],
                    "nb_lignes":        0,
                }
            merges[reduced_label]["segments_origine"].append(seg_info["label"])
            merges[reduced_label]["nb_lignes"] += seg_info["nb_lignes"]
    except Exception:
        return []

    results = []
    for m in merges.values():
        nb_pages, _ = estimate_pages(m["nb_lignes"], lignes_par_page)
        m["nb_pages"] = nb_pages
        if len(m["segments_origine"]) >= 2:
            results.append(m)

    return results


# ══════════════════════════════════════════════════════════════════
# GÉNÉRATION ZIP AVANCÉE (fusions + découpes)
# ══════════════════════════════════════════════════════════════════

def generate_zip_advanced(
    df: pd.DataFrame,
    chunk_cols: list,
    template_str: str,
    filename_source: str,
    fusions: list,
    decoupes: dict,
    lignes_par_page: float,
) -> bytes:
    """
    Version avancée de generate_zip() avec support des fusions et découpes.

    fusions  : liste de {"nom": str, "segments": list[str]}
    decoupes : dict {label_segment: {"mode": "lignes"|"colonne",
                                     "nb_parts": int,  # si mode lignes
                                     "col": str}}      # si mode colonne
    lignes_par_page : pour le calcul des pages dans index.json

    Structure du ZIP :
      - 1 JSON par segment normal (non découpé)
      - N JSON pour chaque segment découpé (_part1, _part2, ...)
      - 1 JSON par colonne pour les découpes colonne
      - 1 JSON par fusion (préfixe FUSION_)
      - index.json avec segments + fusions + decoupes
      - README.txt avec résumé complet
    """
    from utils.rag_scorer import estimate_pages

    valid_cols = [c for c in chunk_cols if c in df.columns]
    if not valid_cols:
        return b""

    df_dtypes = {col: str(df[col].dtype) for col in df.columns}

    # safe_fillna obligatoire avant tout groupby
    df_safe = safe_fillna(df[valid_cols])
    grouped = df_safe.groupby(valid_cols)

    # Construire segments_map : label → df original
    segments_map: dict = {}
    for key, _ in grouped:
        try:
            if isinstance(key, tuple):
                label_parts = [str(k) for k in key]
                seg_vals    = {c: str(k) for c, k in zip(valid_cols, key)}
            else:
                label_parts = [str(key)]
                seg_vals    = {valid_cols[0]: str(key)}

            label = "_".join(safe_filename(p) for p in label_parts)

            mask = pd.Series([True] * len(df), index=df.index)
            for col, val in seg_vals.items():
                mask &= df_safe[col].astype(str) == val
            segments_map[label] = df[mask]
        except Exception:
            pass

    # Segments inclus dans au moins une fusion → exclus des fichiers individuels
    segments_in_fusions: set = set()
    for fusion in fusions:
        for seg_label in fusion.get("segments", []):
            if seg_label:
                segments_in_fusions.add(seg_label)

    zip_buffer = io.BytesIO()
    now_str    = datetime.now().strftime("%d/%m/%Y %H:%M")

    index_data = {
        "date_generation":   now_str,
        "fichier_source":    filename_source,
        "colonnes_chunking": valid_cols,
        "nb_segments_total": len(segments_map),
        "segments":          [],
        "fusions":           [],
        "decoupes":          [],
    }

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:

        # ── Segments normaux et découpés ──────────────────────────────
        for label, seg_df in segments_map.items():
            nb_lignes = len(seg_df)
            nb_pages, _ = estimate_pages(nb_lignes, lignes_par_page)

            if label in decoupes:
                decoupe_cfg = decoupes[label]
                mode = decoupe_cfg.get("mode", "lignes") if isinstance(decoupe_cfg, dict) else "lignes"

                if mode == "colonne":
                    # ── Découpe par 3ème colonne ──────────────────────
                    split_col = decoupe_cfg.get("col", "")
                    if not split_col or split_col not in seg_df.columns:
                        # Fallback : segment normal
                        if label not in segments_in_fusions:
                            try:
                                seg_json = generate_segment_json(
                                    seg_df, template_str, df_dtypes
                                )
                                fname = f"{label}.json"
                                zf.writestr(
                                    fname,
                                    json.dumps(seg_json, ensure_ascii=False, indent=2),
                                )
                                index_data["segments"].append({
                                    "nom":       label,
                                    "fichier":   fname,
                                    "nb_lignes": nb_lignes,
                                    "nb_pages":  nb_pages,
                                    "pct_total": round(nb_lignes / len(df) * 100, 2)
                                                 if len(df) > 0 else 0,
                                })
                            except Exception:
                                pass
                    else:
                        sub_results = split_segment_by_column(
                            seg_df, split_col, label,
                            template_str, df_dtypes, lignes_par_page
                        )
                        decoupe_info = {
                            "segment_original": label,
                            "mode":             "colonne",
                            "colonne_decoupe":  split_col,
                            "nb_parts":         len(sub_results),
                            "nb_lignes_total":  nb_lignes,
                            "fichiers":         [],
                        }
                        for sub in sub_results:
                            try:
                                zf.writestr(
                                    sub["fname"],
                                    json.dumps(
                                        sub["json_data"],
                                        ensure_ascii=False, indent=2,
                                    ),
                                )
                                decoupe_info["fichiers"].append({
                                    "fichier":   sub["fname"],
                                    "nb_lignes": sub["nb_lignes"],
                                    "nb_pages":  sub["nb_pages"],
                                    "valeur":    sub["split_val"],
                                })
                            except Exception:
                                pass
                        index_data["decoupes"].append(decoupe_info)

                else:
                    # ── Découpe par lignes égales ─────────────────────
                    nb_parts = max(2, int(
                        decoupe_cfg.get("nb_parts", 2)
                        if isinstance(decoupe_cfg, dict) else decoupe_cfg
                    ))
                    part_size    = math.ceil(nb_lignes / nb_parts)
                    decoupe_info = {
                        "segment_original": label,
                        "mode":             "lignes",
                        "nb_parts":         nb_parts,
                        "nb_lignes_total":  nb_lignes,
                        "fichiers":         [],
                    }
                    for part_idx in range(nb_parts):
                        start   = part_idx * part_size
                        end     = min(start + part_size, nb_lignes)
                        part_df = seg_df.iloc[start:end]
                        if part_df.empty:
                            continue
                        try:
                            part_json = generate_segment_json(
                                part_df, template_str, df_dtypes
                            )
                            part_pages, _ = estimate_pages(
                                len(part_df), lignes_par_page
                            )
                            part_fname = f"{label}_part{part_idx + 1}.json"
                            zf.writestr(
                                part_fname,
                                json.dumps(part_json, ensure_ascii=False, indent=2),
                            )
                            decoupe_info["fichiers"].append({
                                "fichier":   part_fname,
                                "nb_lignes": len(part_df),
                                "nb_pages":  part_pages,
                            })
                        except Exception:
                            pass
                    index_data["decoupes"].append(decoupe_info)

            elif label not in segments_in_fusions:
                # ── Segment normal ────────────────────────────────────
                try:
                    seg_json = generate_segment_json(
                        seg_df, template_str, df_dtypes
                    )
                    fname = f"{label}.json"
                    zf.writestr(
                        fname,
                        json.dumps(seg_json, ensure_ascii=False, indent=2),
                    )
                    index_data["segments"].append({
                        "nom":       label,
                        "fichier":   fname,
                        "nb_lignes": nb_lignes,
                        "nb_pages":  nb_pages,
                        "pct_total": round(nb_lignes / len(df) * 100, 2)
                                     if len(df) > 0 else 0,
                    })
                except Exception:
                    pass

        # ── Fusions ───────────────────────────────────────────────────
        for fusion in fusions:
            fusion_nom      = fusion.get("nom", "fusion")
            fusion_segments = fusion.get("segments", [])

            if not fusion_segments:
                continue

            parts = []
            for seg_label in fusion_segments:
                if seg_label not in segments_map:
                    continue
                try:
                    seg_objects = generate_segment_json(
                        segments_map[seg_label], template_str, df_dtypes
                    )
                    for obj in seg_objects:
                        if isinstance(obj, dict):
                            obj["_segment"] = seg_label
                        parts.append(obj)
                except Exception:
                    pass

            if not parts:
                continue

            try:
                fusion_fname    = f"FUSION_{safe_filename(fusion_nom)}.json"
                nb_lignes_fusion = len(parts)
                nb_pages_fusion, _ = estimate_pages(
                    nb_lignes_fusion, lignes_par_page
                )
                zf.writestr(
                    fusion_fname,
                    json.dumps(parts, ensure_ascii=False, indent=2),
                )
                index_data["fusions"].append({
                    "nom":                fusion_nom,
                    "fichier":            fusion_fname,
                    "segments_fusionnes": fusion_segments,
                    "nb_lignes_total":    nb_lignes_fusion,
                    "nb_pages_estimees":  nb_pages_fusion,
                })
            except Exception:
                pass

        # ── index.json ────────────────────────────────────────────────
        zf.writestr(
            "index.json",
            json.dumps(index_data, ensure_ascii=False, indent=2),
        )

        # ── README.txt ────────────────────────────────────────────────
        n_seg    = len(segments_map)
        n_fus    = len([f for f in fusions if f.get("segments")])
        n_dec    = len(decoupes)
        avg_size = len(df) // n_seg if n_seg > 0 else 0

        readme_lines = [
            "RAG Chunking — Rapport de génération",
            "=" * 40,
            f"Fichier source    : {filename_source}",
            f"Date              : {now_str}",
            f"Colonnes chunking : {' × '.join(valid_cols)}",
            f"Nb segments       : {n_seg}",
            f"Taille moyenne    : ~{avg_size} lignes/segment",
            f"Nb lignes total   : {len(df):,}",
            f"Fusions créées    : {n_fus}",
            f"Segments découpés : {n_dec}",
            "",
            "Template JSON utilisé :",
            template_str,
        ]
        if fusions:
            readme_lines += ["", "Fusions :"]
            for f in fusions:
                if f.get("segments"):
                    segs = ", ".join(f["segments"])
                    readme_lines.append(f"  - {f['nom']} ← {segs}")
        if decoupes:
            readme_lines += ["", "Découpes :"]
            for seg_lbl, dec_cfg in decoupes.items():
                if isinstance(dec_cfg, dict):
                    if dec_cfg.get("mode") == "colonne":
                        readme_lines.append(
                            f"  - {seg_lbl} → par colonne "
                            f"'{dec_cfg.get('col', '?')}'"
                        )
                    else:
                        readme_lines.append(
                            f"  - {seg_lbl} → "
                            f"{dec_cfg.get('nb_parts', '?')} parties"
                        )
                else:
                    readme_lines.append(f"  - {seg_lbl} → {dec_cfg} parties")
        excluded_from_individual = [
            lbl for lbl in segments_in_fusions if lbl in segments_map
        ]
        if excluded_from_individual:
            readme_lines += [
                "",
                "Segments exclus des fichiers individuels (inclus dans une fusion) :",
            ]
            for seg_lbl in sorted(excluded_from_individual):
                fus_names = [
                    f["nom"]
                    for f in fusions
                    if seg_lbl in f.get("segments", [])
                ]
                readme_lines.append(
                    f"  - {seg_lbl} → fusion(s) : {', '.join(fus_names)}"
                )

        zf.writestr("README.txt", "\n".join(readme_lines))

    return zip_buffer.getvalue()
