"""
pages/9_📦_Chunks_JSON.py
Génération des Chunks JSON — découpage RAG avec template personnalisable.
"""
import json
import math
import re

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Chunks JSON", page_icon="📦", layout="wide")

from utils.guards import check_data_loaded
from utils.profiler import DARK_LAYOUT as DARK, safe_fillna  # noqa: F401
from utils.rag_scorer import estimate_pages
from utils.chunk_generator import (
    build_fusion_column,
    validate_json_template,
    generate_segment_json,
    generate_zip_advanced,
    safe_filename,
    compute_segments_preview,
    split_segment_by_column,
    compute_auto_merge,
)

_, df, profiles = check_data_loaded()


# ── En-tête ───────────────────────────────────────────────────────────────────
st.markdown("# 📦 Génération des Chunks JSON")
st.markdown(
    "Définissez votre découpage, personnalisez la structure JSON "
    "et exportez vos chunks prêts pour le RAG."
)

with st.expander("💡 Comment utiliser cette page", expanded=False):
    st.markdown("""
1. **Colonnes de chunking** : choisissez 1 à 4 colonnes qui définissent
   les segments (1 fichier JSON par segment)
2. **Colonne fusionnée** (optionnel) : créez une colonne qui combine
   plusieurs colonnes avec du texte libre
3. **Template JSON** : modifiez l'exemple généré automatiquement —
   utilisez `{NomColonne}` comme placeholders
4. **Aperçu** : vérifiez le rendu sur le premier segment
5. **Générer** : téléchargez le ZIP complet
    """)

st.markdown("---")

# ════════════════════════════════════════════════════════════
# SECTION 2 — Colonne fusionnée (avant Section 1 car modifie df)
# ════════════════════════════════════════════════════════════
st.markdown("### 2️⃣ Colonne fusionnée (optionnel)")

use_fusion = st.checkbox("Créer une colonne fusionnée", key="cj_use_fusion")

fusion_col_name   = None
df_with_fusion    = df.copy()
fusion_src_cols   = []

if use_fusion:
    fc1, fc2 = st.columns([2, 2])

    with fc1:
        fusion_src_cols = st.multiselect(
            "Colonnes à fusionner",
            options=list(df.columns),
            key="cj_fusion_src_cols",
            help="Colonnes disponibles dans le template de fusion",
        )
        fusion_col_name = st.text_input(
            "Nom de la nouvelle colonne",
            value="texte_fusion",
            key="cj_fusion_col_name",
        )

    with fc2:
        example_parts = " ".join(
            f"{{{c}}}" for c in fusion_src_cols[:3]
        ) if fusion_src_cols else "{Colonne1} {Colonne2}"

        fusion_template = st.text_area(
            "Template de fusion ({NomColonne})",
            value=example_parts,
            height=100,
            key="cj_fusion_template",
            help="Ex : Produit {Denomination} famille {Grande_Famille} prix {Prix}€",
        )

    opt1, opt2, opt3 = st.columns(3)
    with opt1:
        include_in_json = st.checkbox(
            "Inclure dans le JSON", value=True, key="cj_fusion_include"
        )
    with opt2:
        replace_originals = st.checkbox(
            "Remplacer les colonnes originales", value=False, key="cj_fusion_replace"
        )
    with opt3:
        use_as_chunk = st.checkbox(
            "Utiliser comme critère de chunking", value=False, key="cj_fusion_as_chunk"
        )

    if fusion_src_cols and fusion_template.strip() and fusion_col_name.strip():
        valid_src = [c for c in fusion_src_cols if c in df.columns]
        try:
            fusion_series = build_fusion_column(df, valid_src, fusion_template)
            df_with_fusion = df.copy()
            df_with_fusion[fusion_col_name] = fusion_series

            st.markdown("**Aperçu de la colonne fusionnée :**")
            st.dataframe(
                pd.DataFrame({fusion_col_name: fusion_series.head(5)}),
                use_container_width=True,
                height=180,
            )
            if use_as_chunk:
                st.info(
                    f"✅ `{fusion_col_name}` disponible comme critère de chunking "
                    "— ajoutez-la dans la Section 1."
                )
        except Exception as e:
            st.error(f"Erreur colonne fusionnée : {e}")

st.markdown("---")

# ════════════════════════════════════════════════════════════
# SECTION 1 — Colonnes de chunking
# ════════════════════════════════════════════════════════════
st.markdown("### 1️⃣ Colonnes de chunking")

available_cols = list(df_with_fusion.columns)

cat_cols = [
    c for c, p in profiles.items()
    if p.col_type in ("categorical", "binary")
    and c in df.columns
]

chunk_cols = st.multiselect(
    "Colonnes de segmentation (1 à 4 colonnes max)",
    options=available_cols,
    default=cat_cols[:1] if cat_cols else [],
    max_selections=4,
    key="cj_chunk_cols",
    help="1 fichier JSON sera généré par combinaison unique de ces colonnes",
)

# Aperçu nb segments en temps réel
if chunk_cols:
    try:
        valid_cc = [c for c in chunk_cols if c in df_with_fusion.columns]
        df_safe_preview = safe_fillna(df_with_fusion[valid_cc])
        n_segs = df_safe_preview.groupby(valid_cc).ngroups
        avg_s  = len(df) // n_segs if n_segs > 0 else 0
        color  = (
            "#3fb950" if n_segs <= 100
            else "#d29922" if n_segs <= 500
            else "#f85149"
        )
        st.markdown(
            f'<div style="color:{color};font-weight:600;font-size:0.95rem">'
            f'→ {n_segs:,} segments générés · ~{avg_s:,} lignes/segment</div>',
            unsafe_allow_html=True,
        )
        if n_segs > 1000:
            st.warning(
                f"⚠️ {n_segs:,} segments = {n_segs:,} fichiers JSON dans le ZIP "
                "— réduisez le nombre de colonnes ou choisissez des colonnes "
                "à plus faible cardinalité."
            )
    except Exception as e:
        st.error(f"Erreur calcul segments : {e}")
else:
    st.info("👆 Sélectionnez au moins une colonne.")

st.markdown("---")

# ════════════════════════════════════════════════════════════
# SECTION 3 — Template JSON
# ════════════════════════════════════════════════════════════
st.markdown("### 3️⃣ Template JSON")
st.caption(
    "Modifiez l'exemple ci-dessous. "
    "Utilisez `{NomColonne}` pour les valeurs dynamiques."
)


def _build_example_template(
    df_wf: pd.DataFrame,
    profiles_: dict,
    fusion_name,
    include_fusion: bool,
    replace_orig: bool,
    fusion_src: list,
) -> str:
    """Génère un exemple JSON avec toutes les colonnes comme placeholders."""
    obj = {}
    for col in df_wf.columns:
        if replace_orig and col in fusion_src:
            continue
        if col == fusion_name and not include_fusion:
            continue
        obj[col] = f"{{{col}}}"
    return json.dumps(obj, ensure_ascii=False, indent=2)


if st.button("🔄 Regénérer l'exemple depuis mes colonnes", key="cj_regen_example"):
    example_tmpl = _build_example_template(
        df_with_fusion,
        profiles,
        fusion_col_name,
        use_fusion and st.session_state.get("cj_fusion_include", True),
        use_fusion and st.session_state.get("cj_fusion_replace", False),
        fusion_src_cols,
    )
    st.session_state["cj_json_template"] = example_tmpl

if "cj_json_template" not in st.session_state:
    st.session_state["cj_json_template"] = _build_example_template(
        df_with_fusion, profiles, fusion_col_name, True, False, [],
    )

json_template = st.text_area(
    "Template JSON (modifiable)",
    value=st.session_state["cj_json_template"],
    height=320,
    key="cj_json_template_input",
    help="Chaque {NomColonne} sera remplacé par la vraie valeur. "
         "Les valeurs manquantes deviennent null.",
)

# ── Validation en temps réel ──────────────────────────────────────────────────
template_available_cols = list(df_with_fusion.columns)
validation: dict = {"valid": False}

if json_template.strip():
    validation = validate_json_template(json_template, template_available_cols)

    if validation["valid"]:
        st.success("✅ JSON valide")
    else:
        msg  = validation["error_msg"]
        line = validation["error_line"]
        st.error(f"❌ {msg}")
        if line > 0:
            lines = json_template.split("\n")
            if 0 < line <= len(lines):
                st.code(f"Ligne {line} : {lines[line-1].strip()}", language="json")

    if validation.get("placeholders_ok"):
        st.markdown(
            "**Placeholders valides :** "
            + " ".join(
                f'<span style="background:#1a3a1a;color:#3fb950;'
                f'padding:2px 8px;border-radius:4px;'
                f'font-size:0.82rem;margin:2px">{p}</span>'
                for p in validation["placeholders_ok"]
            ),
            unsafe_allow_html=True,
        )

    if validation.get("placeholders_bad"):
        st.markdown(
            "**Placeholders introuvables :** "
            + " ".join(
                f'<span style="background:#2a1a1a;color:#f85149;'
                f'padding:2px 8px;border-radius:4px;'
                f'font-size:0.82rem;margin:2px">{p}</span>'
                for p in validation["placeholders_bad"]
            ),
            unsafe_allow_html=True,
        )

st.markdown("---")

# ════════════════════════════════════════════════════════════
# SECTION 4 — Aperçu d'un segment
# ════════════════════════════════════════════════════════════
st.markdown("### 4️⃣ Aperçu d'un segment")

can_preview = (
    bool(chunk_cols)
    and bool(json_template.strip())
    and validation.get("valid", False)
)

if can_preview:
    if st.button("👁️ Aperçu du premier segment", key="cj_preview_btn"):
        try:
            valid_cc = [c for c in chunk_cols if c in df_with_fusion.columns]
            df_safe_p = safe_fillna(df_with_fusion[valid_cc])
            grouped_p = df_safe_p.groupby(valid_cc)

            first_key = next(iter(grouped_p.groups))
            # Construire le masque sur df_with_fusion
            if isinstance(first_key, tuple):
                mask = pd.Series([True] * len(df_with_fusion),
                                 index=df_with_fusion.index)
                for col, val in zip(valid_cc, first_key):
                    mask &= df_safe_p[col].astype(str) == str(val)
                label_prev = "_".join(str(k) for k in first_key)
            else:
                mask = df_safe_p[valid_cc[0]].astype(str) == str(first_key)
                label_prev = str(first_key)

            seg_df_prev = df_with_fusion[mask]
            df_dtypes_p = {col: str(df_with_fusion[col].dtype)
                           for col in df_with_fusion.columns}
            preview_json = generate_segment_json(
                seg_df_prev.head(5), json_template, df_dtypes_p
            )

            st.markdown(
                f"**Segment :** `{label_prev}` "
                f"({len(seg_df_prev):,} lignes) — aperçu des 5 premières lignes :"
            )
            st.code(
                json.dumps(preview_json, ensure_ascii=False, indent=2),
                language="json",
            )
            st.caption(
                f"Le fichier complet contiendra {len(seg_df_prev):,} objets JSON."
            )
        except Exception as e:
            st.error(f"Erreur aperçu : {e}")
else:
    if not chunk_cols:
        st.info("Sélectionnez les colonnes de chunking pour activer l'aperçu.")
    elif not json_template.strip():
        st.info("Saisissez un template JSON pour activer l'aperçu.")
    else:
        st.warning("Corrigez les erreurs JSON pour activer l'aperçu.")

st.markdown("---")

# ════════════════════════════════════════════════════════════
# SECTION 4.5 — Tableau segments + fusions + découpes
# ════════════════════════════════════════════════════════════
st.markdown("### 📋 Segments générés")

# ── Paramètres seuil pages ───────────────────────────────────────
seuil_col, seuil_min_col, info_col = st.columns([1, 1, 2])
with seuil_col:
    seuil_pages = st.number_input(
        "⚠️ Seuil maximum (pages)",
        min_value=50,
        max_value=5000,
        value=300,
        step=50,
        key="cj_seuil_pages",
        help="Segments dépassant ce seuil sont signalés en rouge",
    )
with seuil_min_col:
    seuil_min_pages = st.number_input(
        "📉 Seuil minimum (pages)",
        min_value=1,
        max_value=100,
        value=10,
        step=5,
        key="cj_seuil_min_pages",
        help="Segments en dessous de ce seuil sont signalés comme trop petits",
    )
with info_col:
    lignes_par_page_cj = float(st.session_state.get("lignes_par_page", 3))
    limite_lignes_cj   = int(seuil_pages * lignes_par_page_cj)
    st.info(
        f"Seuil max {seuil_pages} pages = **{limite_lignes_cj:,} lignes** · "
        f"Seuil min {seuil_min_pages} pages = **{int(seuil_min_pages * lignes_par_page_cj):,} lignes** "
        f"({lignes_par_page_cj:.0f} lignes/page · modifiable page Simulateur RAG)"
    )

# ── Calculer les segments ────────────────────────────────────────
segments_preview = []
if chunk_cols and can_preview:
    try:
        valid_cc_p = [c for c in chunk_cols if c in df_with_fusion.columns]
        segments_preview = compute_segments_preview(
            df_with_fusion, valid_cc_p, lignes_par_page_cj
        )
        for seg in segments_preview:
            seg["depasse"] = seg["nb_pages"] > seuil_pages
    except Exception as e:
        st.error(f"Erreur calcul segments : {e}")

if segments_preview:

    st.caption(
        f"{len(segments_preview)} segments · "
        f"🔵 < {seuil_min_pages} pages (trop petit) · "
        f"🟢 ≤ {int(seuil_pages * 0.67)} pages · "
        f"🟡 ≤ {seuil_pages} pages · "
        f"🔴 > {seuil_pages} pages"
    )

    # Initialiser session_state
    if "cj_decoupes" not in st.session_state:
        st.session_state["cj_decoupes"] = {}
    if "cj_fusions" not in st.session_state:
        st.session_state["cj_fusions"] = []

    # ── Gestion des fusions ───────────────────────────────────────
    st.markdown("#### 🔀 Fusions de segments")
    st.caption(
        "Créez des fusions pour combiner plusieurs segments dans un seul fichier JSON. "
        "Sélectionnez une fusion active, puis cochez les segments à y inclure."
    )

    fus_header_col, fus_btn_col = st.columns([5, 1])
    with fus_btn_col:
        if st.button("➕ Créer une fusion", key="cj_add_fusion", use_container_width=True):
            st.session_state["cj_fusions"].append({
                "nom":      f"fusion_{len(st.session_state['cj_fusions']) + 1}",
                "segments": [],
            })
            # Pas de st.rerun() → les cases à cocher conservent leur état

    fusions_to_delete = []
    for i, fusion in enumerate(st.session_state["cj_fusions"]):
        fi1, fi2, fi3 = st.columns([3, 5, 1])
        with fi1:
            new_nom = st.text_input(
                "Nom",
                value=fusion["nom"],
                key=f"cj_fusion_nom_{i}",
                label_visibility="collapsed",
                placeholder="Nom de la fusion...",
            )
            st.session_state["cj_fusions"][i]["nom"] = new_nom
        with fi2:
            segs_f = fusion.get("segments", [])
            n_segs_in_fusion = len(segs_f)
            if n_segs_in_fusion:
                total_lines_f = 0
                for seg_label in segs_f:
                    # Segment direct dans segments_preview ?
                    direct = next(
                        (s["nb_lignes"] for s in segments_preview
                         if s["label"] == seg_label),
                        None,
                    )
                    if direct is not None:
                        total_lines_f += direct
                        continue
                    # Sous-segment — chercher le parent puis compter
                    for seg in segments_preview:
                        if seg_label.startswith(seg["label"] + "_"):
                            try:
                                applied_col = st.session_state.get(
                                    f"cj_dec_applied_col_{seg['label']}", ""
                                )
                                if applied_col and applied_col in df_with_fusion.columns:
                                    suffix = seg_label[len(seg["label"]) + 1:]
                                    valid_cc_p = [
                                        c for c in chunk_cols
                                        if c in df_with_fusion.columns
                                    ]
                                    df_safe_p = safe_fillna(
                                        df_with_fusion[valid_cc_p]
                                    )
                                    mask = pd.Series(
                                        [True] * len(df_with_fusion),
                                        index=df_with_fusion.index,
                                    )
                                    for col_k, val_k in seg["seg_vals"].items():
                                        if col_k in df_safe_p.columns:
                                            mask &= (
                                                df_safe_p[col_k].astype(str)
                                                == str(val_k)
                                            )
                                    seg_df_check = df_with_fusion[mask]
                                    sub_vals = (
                                        seg_df_check[applied_col]
                                        .dropna().astype(str).unique()
                                    )
                                    for v in sub_vals:
                                        if safe_filename(v) == suffix:
                                            total_lines_f += int(
                                                (seg_df_check[applied_col]
                                                 .fillna("").astype(str) == v)
                                                .sum()
                                            )
                                            break
                            except Exception:
                                pass
                            break
                total_pages_f = math.ceil(total_lines_f / max(lignes_par_page_cj, 1))
                page_color_f = (
                    "#3fb950" if total_pages_f <= seuil_pages
                    else "#d29922" if total_pages_f <= seuil_pages * 1.5
                    else "#f85149"
                )
                st.markdown(
                    f'<div style="padding-top:0.45rem;font-size:0.85rem;color:#8b949e">'
                    f'{n_segs_in_fusion} segment(s) · {total_lines_f:,} lignes · '
                    f'<span style="color:{page_color_f}">~{total_pages_f:,} pages</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="padding-top:0.45rem;font-size:0.85rem;color:#8b949e">'
                    'Aucun segment sélectionné</div>',
                    unsafe_allow_html=True,
                )
        with fi3:
            if st.button("🗑️", key=f"cj_del_fusion_{i}", help="Supprimer cette fusion"):
                fusions_to_delete.append(i)

    if fusions_to_delete:
        st.session_state["cj_fusions"] = [
            f for j, f in enumerate(st.session_state["cj_fusions"])
            if j not in fusions_to_delete
        ]
        # Réajuster l'index de fusion active
        active = st.session_state.get("cj_fusion_active_idx", 0)
        n_left = len(st.session_state["cj_fusions"])
        if n_left == 0:
            st.session_state["cj_fusion_active_idx"] = None
        elif isinstance(active, int) and active >= n_left:
            st.session_state["cj_fusion_active_idx"] = n_left - 1
        st.rerun()

    # ── Sélecteur fusion active ───────────────────────────────────
    fusion_names      = [f["nom"] for f in st.session_state["cj_fusions"]]
    fusion_active_idx = None
    if fusion_names:
        active_options = ["— Aucune —"] + fusion_names
        active_label = st.selectbox(
            "🎯 Fusion active — les cases à cocher s'appliquent à cette fusion",
            options=active_options,
            key="cj_fusion_active_select",
        )
        if active_label != "— Aucune —" and active_label in fusion_names:
            fusion_active_idx = fusion_names.index(active_label)

    # ── Index complet fusions ──────────────────────────────────────
    # Tous les labels présents dans au moins une fusion
    all_fused_labels: set = set()
    # Segment parent → liste des sous-labels de ce parent qui sont fusionnés
    parent_to_fused_sublabels: dict = {}

    for fusion in st.session_state.get("cj_fusions", []):
        for s in fusion.get("segments", []):
            if not s:
                continue
            all_fused_labels.add(s)
            # Déterminer si s est un sous-segment d'un parent
            for seg in segments_preview:
                parent_lbl = seg["label"]
                if s != parent_lbl and s.startswith(parent_lbl + "_"):
                    if parent_lbl not in parent_to_fused_sublabels:
                        parent_to_fused_sublabels[parent_lbl] = []
                    if s not in parent_to_fused_sublabels[parent_lbl]:
                        parent_to_fused_sublabels[parent_lbl].append(s)

    # Mapping label → liste de noms de fusions (pour les badges)
    seg_to_fusions: dict = {}
    for fusion in st.session_state["cj_fusions"]:
        for seg_label in fusion.get("segments", []):
            seg_to_fusions.setdefault(seg_label, []).append(fusion["nom"])

    st.markdown("---")

    # ── En-têtes du tableau ───────────────────────────────────────
    hdr_cb, hdr1, hdr2, hdr3, hdr4 = st.columns([0.5, 3, 1.5, 1.5, 2])
    with hdr_cb: st.markdown("**✓**")
    with hdr1:   st.markdown("**Fichier**")
    with hdr2:   st.markdown("**Lignes**")
    with hdr3:   st.markdown("**Pages**")
    with hdr4:   st.markdown("**Découpe**")

    for seg in segments_preview:
        col_cb, col_fname, col_lines, col_pages, col_action = st.columns(
            [0.5, 3, 1.5, 1.5, 2]
        )

        # ── Case à cocher ─────────────────────────────────────────
        with col_cb:
            if fusion_active_idx is not None:
                current_segs = st.session_state["cj_fusions"][fusion_active_idx].get(
                    "segments", []
                )
                is_checked = seg["label"] in current_segs
                checked = st.checkbox(
                    "",
                    value=is_checked,
                    key=f"cj_cb_{seg['label']}_{fusion_active_idx}",
                    label_visibility="collapsed",
                )
                if checked and seg["label"] not in current_segs:
                    st.session_state["cj_fusions"][fusion_active_idx]["segments"].append(
                        seg["label"]
                    )
                elif not checked and seg["label"] in current_segs:
                    st.session_state["cj_fusions"][fusion_active_idx]["segments"].remove(
                        seg["label"]
                    )
            else:
                st.checkbox(
                    "",
                    value=False,
                    disabled=True,
                    key=f"cj_cb_{seg['label']}_none",
                    label_visibility="collapsed",
                )

        # ── Nom fichier + badge fusion ─────────────────────────────
        with col_fname:
            fusion_badge = ""
            in_any_fusion = (
                seg["label"] in all_fused_labels
                or seg["label"] in parent_to_fused_sublabels
            )
            if in_any_fusion:
                fusion_nom_badge = ""
                # Fusion directe
                for fu in st.session_state.get("cj_fusions", []):
                    if seg["label"] in fu.get("segments", []):
                        fusion_nom_badge = fu["nom"]
                        break
                # Via sous-segments
                if not fusion_nom_badge:
                    sub_labels = parent_to_fused_sublabels.get(seg["label"], [])
                    for fu in st.session_state.get("cj_fusions", []):
                        if any(sl in fu.get("segments", []) for sl in sub_labels):
                            n_sub_in = sum(
                                1 for sl in sub_labels
                                if sl in fu.get("segments", [])
                            )
                            fusion_nom_badge = (
                                f"{fu['nom']} ({n_sub_in} sous-seg.)"
                            )
                            break
                if fusion_nom_badge:
                    fusion_badge = (
                        f' <span style="background:#1f6feb;color:#fff;'
                        f'border-radius:4px;padding:1px 5px;font-size:0.72rem;'
                        f'margin-left:4px">🔀 {fusion_nom_badge}</span>'
                    )
            st.markdown(
                f'<span style="font-size:0.88rem;color:#c9d1d9">'
                f'{seg["fname"]}{fusion_badge}</span>',
                unsafe_allow_html=True,
            )

        with col_lines:
            st.markdown(
                f'<span style="font-size:0.88rem;color:#8b949e">'
                f'{seg["nb_lignes"]:,} lignes</span>',
                unsafe_allow_html=True,
            )

        with col_pages:
            if seg["nb_pages"] > seuil_pages:
                color, icon, badge = "#f85149", "🔴", "⬆️ trop grand"
            elif seg["nb_pages"] < seuil_min_pages:
                color, icon, badge = "#8b949e", "🔵", "⬇️ trop petit"
            elif seg["nb_pages"] > seuil_pages * 0.67:
                color, icon, badge = "#d29922", "🟡", ""
            else:
                color, icon, badge = "#3fb950", "🟢", ""
            st.markdown(
                f'<span style="color:{color};font-size:0.87rem;font-weight:600">'
                f'{icon} {seg["nb_pages"]:,}'
                f'{"  " + badge if badge else ""}'
                f'</span>',
                unsafe_allow_html=True,
            )

        with col_action:
            if seg["depasse"]:
                dec_mode = st.selectbox(
                    "Mode",
                    ["✂️ Par lignes", "📂 Par colonne"],
                    key=f"cj_dec_mode_{seg['label']}",
                    label_visibility="collapsed",
                )

                if dec_mode == "✂️ Par lignes":
                    suggestion  = max(2, math.ceil(seg["nb_pages"] / seuil_pages))
                    current_cfg = st.session_state["cj_decoupes"].get(seg["label"], {})
                    current_val = (
                        current_cfg.get("nb_parts", suggestion)
                        if isinstance(current_cfg, dict)
                        else suggestion
                    )
                    nb_parts = st.number_input(
                        "Nb parties",
                        min_value=2,
                        max_value=50,
                        value=int(current_val),
                        step=1,
                        key=f"cj_dec_nb_{seg['label']}",
                        label_visibility="collapsed",
                    )
                    pages_part  = math.ceil(seg["nb_pages"]  / nb_parts)
                    lignes_part = math.ceil(seg["nb_lignes"] / nb_parts)
                    st.caption(f"→ {nb_parts}×~{lignes_part:,}L (~{pages_part}p)")
                    st.session_state["cj_decoupes"][seg["label"]] = {
                        "mode": "lignes", "nb_parts": nb_parts
                    }

                else:  # Par colonne
                    extra_cols = [
                        c for c in df_with_fusion.columns
                        if c not in chunk_cols
                    ]
                    if not extra_cols:
                        st.caption("Aucune colonne disponible")
                    else:
                        split_col = st.selectbox(
                            "Colonne de découpe",
                            options=extra_cols,
                            key=f"cj_dec_col_{seg['label']}",
                            label_visibility="collapsed",
                        )

                        # Bouton de confirmation avant le calcul
                        apply_key      = f"cj_dec_apply_{seg['label']}"
                        applied_key    = f"cj_dec_applied_{seg['label']}"
                        applied_col_key = f"cj_dec_applied_col_{seg['label']}"

                        if st.button(
                            "▶ Appliquer",
                            key=apply_key,
                            help="Calculer les sous-segments avec cette colonne",
                        ):
                            st.session_state[applied_key]     = True
                            st.session_state[applied_col_key] = split_col

                        if st.session_state.get(applied_key):
                            active_split_col = st.session_state.get(
                                applied_col_key, split_col
                            )
                            st.session_state["cj_decoupes"][seg["label"]] = {
                                "mode": "colonne", "col": active_split_col
                            }

                            # ── Calculer les sous-segments ──────────────
                            try:
                                valid_cc_p = [
                                    c for c in chunk_cols
                                    if c in df_with_fusion.columns
                                ]
                                df_safe_p = safe_fillna(
                                    df_with_fusion[valid_cc_p]
                                )
                                mask = pd.Series(
                                    [True] * len(df_with_fusion),
                                    index=df_with_fusion.index,
                                )
                                for col_k, val_k in seg["seg_vals"].items():
                                    if col_k in df_safe_p.columns:
                                        mask &= (
                                            df_safe_p[col_k].astype(str)
                                            == str(val_k)
                                        )
                                seg_df_detail = df_with_fusion[mask]

                                if active_split_col not in seg_df_detail.columns:
                                    st.caption("Colonne absente du segment")
                                else:
                                    df_split_safe = safe_fillna(
                                        seg_df_detail[[active_split_col]]
                                    )
                                    # Convertir en str pour supprimer les catégories
                                    # fantômes (colonnes category conservent toutes
                                    # leurs valeurs même dans un sous-ensemble)
                                    df_split_safe[active_split_col] = (
                                        df_split_safe[active_split_col].astype(str)
                                    )
                                    sub_groups = (
                                        df_split_safe
                                        .groupby(active_split_col)
                                        .size()
                                        .reset_index(name="nb_lignes")
                                    )
                                    # Exclure les groupes vides (catégories fantômes)
                                    sub_groups = sub_groups[
                                        sub_groups["nb_lignes"] > 0
                                    ].sort_values("nb_lignes", ascending=False)
                                    sub_groups = sub_groups.reset_index(drop=True)

                                    n_sub   = len(sub_groups)
                                    avg_sub = (
                                        seg["nb_lignes"] // n_sub
                                        if n_sub > 0 else 0
                                    )
                                    st.caption(
                                        f"→ {n_sub} sous-segments "
                                        f"(~{avg_sub:,} lignes/seg)"
                                    )

                                    # ── Tableau des sous-segments ───────────
                                    ss_key = f"cj_sub_sel_{seg['label']}"
                                    if ss_key not in st.session_state:
                                        st.session_state[ss_key] = []

                                    with st.expander(
                                        f"📂 Voir les {n_sub} sous-segments",
                                        expanded=True,
                                    ):
                                        sh0, sh1, sh2, sh3 = st.columns(
                                            [0.4, 3.5, 1.2, 1.5]
                                        )
                                        sh0.markdown(
                                            '<span style="color:#8b949e;'
                                            'font-size:0.72rem">✔</span>',
                                            unsafe_allow_html=True,
                                        )
                                        sh1.markdown(
                                            '<span style="color:#8b949e;'
                                            'font-size:0.72rem">Fichier</span>',
                                            unsafe_allow_html=True,
                                        )
                                        sh2.markdown(
                                            '<span style="color:#8b949e;'
                                            'font-size:0.72rem">Lignes</span>',
                                            unsafe_allow_html=True,
                                        )
                                        sh3.markdown(
                                            '<span style="color:#8b949e;'
                                            'font-size:0.72rem">Pages</span>',
                                            unsafe_allow_html=True,
                                        )

                                        current_selection = list(
                                            st.session_state.get(ss_key, [])
                                        )

                                        for _, sub_row in sub_groups.iterrows():
                                            sub_val  = str(sub_row[active_split_col])
                                            sub_nb_l = int(sub_row["nb_lignes"])
                                            sub_nb_p, _ = estimate_pages(
                                                sub_nb_l, lignes_par_page_cj
                                            )
                                            sub_label = (
                                                f"{seg['label']}_"
                                                f"{safe_filename(sub_val)}"
                                            )
                                            sub_fname = f"{sub_label}.json"

                                            sc0, sc1, sc2, sc3 = st.columns(
                                                [0.4, 3.5, 1.2, 1.5]
                                            )

                                            with sc0:
                                                sub_checked = st.checkbox(
                                                    "",
                                                    value=sub_label in current_selection,
                                                    key=(
                                                        f"cj_sub_cb_"
                                                        f"{seg['label']}_"
                                                        f"{safe_filename(sub_val)}"
                                                    ),
                                                    label_visibility="collapsed",
                                                )
                                                if sub_checked and sub_label not in current_selection:
                                                    current_selection.append(sub_label)
                                                elif not sub_checked and sub_label in current_selection:
                                                    current_selection.remove(sub_label)

                                            with sc1:
                                                in_fusion = any(
                                                    sub_label in f.get("segments", [])
                                                    for f in st.session_state.get(
                                                        "cj_fusions", []
                                                    )
                                                )
                                                badge = ""
                                                if in_fusion:
                                                    fusion_nom = next(
                                                        (
                                                            f["nom"]
                                                            for f in st.session_state.get(
                                                                "cj_fusions", []
                                                            )
                                                            if sub_label in f.get("segments", [])
                                                        ),
                                                        "",
                                                    )
                                                    badge = (
                                                        f' <span style="background:#1a2a3a;'
                                                        f'color:#58a6ff;font-size:0.7rem;'
                                                        f'padding:1px 5px;border-radius:4px">'
                                                        f'🔀 {fusion_nom}</span>'
                                                    )
                                                st.markdown(
                                                    f'<span style="font-size:0.83rem;'
                                                    f'color:#c9d1d9">'
                                                    f'{sub_fname}{badge}</span>',
                                                    unsafe_allow_html=True,
                                                )

                                            with sc2:
                                                st.markdown(
                                                    f'<span style="font-size:0.83rem;'
                                                    f'color:#8b949e">'
                                                    f'{sub_nb_l:,}</span>',
                                                    unsafe_allow_html=True,
                                                )

                                            with sc3:
                                                if sub_nb_p > seuil_pages:
                                                    s_color, s_icon = "#f85149", "🔴"
                                                elif sub_nb_p < seuil_min_pages:
                                                    s_color, s_icon = "#8b949e", "🔵"
                                                elif sub_nb_p > seuil_pages * 0.67:
                                                    s_color, s_icon = "#d29922", "🟡"
                                                else:
                                                    s_color, s_icon = "#3fb950", "🟢"
                                                st.markdown(
                                                    f'<span style="color:{s_color};'
                                                    f'font-size:0.83rem;font-weight:600">'
                                                    f'{s_icon} {sub_nb_p:,}</span>',
                                                    unsafe_allow_html=True,
                                                )

                                        # Sauvegarder la sélection
                                        st.session_state[ss_key] = current_selection

                                        # ── Fusion des sous-segments sélectionnés ──
                                        selected_subs = st.session_state.get(ss_key, [])
                                        if selected_subs:
                                            st.markdown("---")
                                            fus_col1, fus_col2 = st.columns([2, 2])

                                            with fus_col1:
                                                fusion_names_avail = [
                                                    f["nom"]
                                                    for f in st.session_state.get(
                                                        "cj_fusions", []
                                                    )
                                                ]
                                                if fusion_names_avail:
                                                    target_fusion = st.selectbox(
                                                        "Ajouter à la fusion",
                                                        options=fusion_names_avail,
                                                        key=(
                                                            f"cj_sub_fusion_target_"
                                                            f"{seg['label']}"
                                                        ),
                                                        label_visibility="collapsed",
                                                    )
                                                else:
                                                    target_fusion = None
                                                    st.caption(
                                                        "Créez d'abord une fusion "
                                                        "dans la section Fusions."
                                                    )

                                            with fus_col2:
                                                if target_fusion and selected_subs:
                                                    st.caption(
                                                        f"{len(selected_subs)} "
                                                        f"sous-segment(s) sélectionné(s)"
                                                    )
                                                    if st.button(
                                                        f"🔀 Ajouter à «{target_fusion}»",
                                                        key=f"cj_sub_add_{seg['label']}",
                                                        type="primary",
                                                    ):
                                                        for fi, fu in enumerate(
                                                            st.session_state.get(
                                                                "cj_fusions", []
                                                            )
                                                        ):
                                                            if fu["nom"] == target_fusion:
                                                                existing = list(
                                                                    fu.get("segments", [])
                                                                )
                                                                for sl in selected_subs:
                                                                    if sl not in existing:
                                                                        existing.append(sl)
                                                                st.session_state[
                                                                    "cj_fusions"
                                                                ][fi]["segments"] = existing
                                                                break
                                                        st.success(
                                                            f"✅ {len(selected_subs)} "
                                                            f"sous-segment(s) ajouté(s) "
                                                            f"à «{target_fusion}»"
                                                        )
                                                        st.session_state[ss_key] = []
                                                        st.rerun()
                            except Exception as e:
                                st.error(f"Erreur calcul sous-segments : {e}")
                        else:
                            st.caption(
                                "👆 Choisissez une colonne puis "
                                "cliquez ▶ Appliquer"
                            )

            elif seg["nb_pages"] < seuil_min_pages:
                st.markdown(
                    '<span style="color:#8b949e;font-size:0.8rem">'
                    "⬇️ Segment petit</span>",
                    unsafe_allow_html=True,
                )
                st.caption("💡 Cochez pour l'ajouter à une fusion")
            else:
                st.session_state["cj_decoupes"].pop(seg["label"], None)

    # ── Carte résumé fusion active ────────────────────────────────
    if fusion_active_idx is not None:
        fusion_active = st.session_state["cj_fusions"][fusion_active_idx]
        active_segs   = fusion_active.get("segments", [])
        if active_segs:
            total_lines_a = sum(
                s["nb_lignes"] for s in segments_preview if s["label"] in active_segs
            )
            total_pages_a = math.ceil(total_lines_a / max(lignes_par_page_cj, 1))
            page_color_a  = (
                "#3fb950" if total_pages_a <= seuil_pages
                else "#d29922" if total_pages_a <= seuil_pages * 1.5
                else "#f85149"
            )
            st.markdown(
                f'<div style="background:#161b22;border:1px solid #21262d;'
                f'border-radius:8px;padding:0.8rem 1rem;margin-top:0.6rem">'
                f'<strong>🔀 {fusion_active["nom"]}</strong> — '
                f'{len(active_segs)} segment(s) · {total_lines_a:,} lignes · '
                f'<span style="color:{page_color_a}">~{total_pages_a:,} pages</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Fusion automatique petits segments ───────────────────
    n_small = sum(
        1 for s in segments_preview
        if s["nb_pages"] < seuil_min_pages
    )

    if n_small > 0:
        st.markdown("---")
        st.markdown(
            f"#### ⬇️ {n_small} segment(s) trop petit(s) "
            f"(< {seuil_min_pages} pages)"
        )
        st.caption(
            "Option : supprimer une colonne de filtre pour "
            "fusionner automatiquement ces petits segments."
        )

        am_col1, am_col2 = st.columns([2, 3])
        with am_col1:
            if len(chunk_cols) > 1:
                col_to_remove = st.selectbox(
                    "Colonne à supprimer",
                    options=chunk_cols,
                    key="cj_auto_merge_col",
                    help=(
                        "Les petits segments seront regroupés "
                        "selon les colonnes restantes"
                    ),
                )
                if st.button(
                    "🔍 Prévisualiser la fusion auto",
                    key="cj_auto_merge_preview",
                ):
                    try:
                        auto_merges = compute_auto_merge(
                            df_with_fusion,
                            chunk_cols,
                            col_to_remove,
                            seuil_min_pages,
                            lignes_par_page_cj,
                        )
                        st.session_state["cj_auto_merges"]    = auto_merges
                        st.session_state["cj_auto_merge_col_saved"] = col_to_remove
                    except Exception as e:
                        st.error(f"Erreur : {e}")
            else:
                st.info(
                    "Impossible : une seule colonne "
                    "de chunking active."
                )

        with am_col2:
            auto_merges = st.session_state.get("cj_auto_merges", [])
            if auto_merges:
                st.markdown(
                    f"**{len(auto_merges)} fusion(s) "
                    f"automatique(s) prévue(s) :**"
                )
                for am in auto_merges:
                    p_color = (
                        "#3fb950"
                        if am["nb_pages"] <= seuil_pages
                        else "#d29922"
                    )
                    st.markdown(
                        f'<div style="background:#161b22;'
                        f'border-left:3px solid #58a6ff;'
                        f'border-radius:0 6px 6px 0;'
                        f'padding:0.4rem 0.8rem;'
                        f'margin-bottom:0.4rem">'
                        f'<span style="color:#c9d1d9;font-size:0.85rem">'
                        f'<b>{am["nouveau_label"]}.json</b>'
                        f' ← '
                        f'{", ".join(am["segments_origine"])}'
                        f'</span><br>'
                        f'<span style="color:{p_color};font-size:0.8rem">'
                        f'{am["nb_lignes"]:,} lignes · '
                        f'~{am["nb_pages"]} pages</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                if st.button(
                    "✅ Appliquer la fusion automatique",
                    type="primary",
                    key="cj_apply_auto_merge",
                ):
                    for am in auto_merges:
                        st.session_state["cj_fusions"].append({
                            "nom":      am["nouveau_label"],
                            "segments": am["segments_origine"],
                        })
                    st.session_state["cj_auto_merges"] = []
                    st.success(
                        f"✅ {len(auto_merges)} fusion(s) ajoutée(s) — "
                        f"les petits segments seront regroupés dans le ZIP."
                    )
                    st.rerun()

    # ── Segments complètement fusionnés (pour résumé) ───────────
    segs_fully_fused: set = set()
    for seg in segments_preview:
        dec_cfg = st.session_state.get("cj_decoupes", {}).get(seg["label"], {})

        if isinstance(dec_cfg, dict) and dec_cfg.get("mode") == "colonne":
            applied_col = st.session_state.get(
                f"cj_dec_applied_col_{seg['label']}", ""
            )
            if applied_col and applied_col in df_with_fusion.columns:
                try:
                    valid_cc_p = [
                        c for c in chunk_cols
                        if c in df_with_fusion.columns
                    ]
                    df_safe_p = safe_fillna(df_with_fusion[valid_cc_p])
                    mask = pd.Series(
                        [True] * len(df_with_fusion),
                        index=df_with_fusion.index,
                    )
                    for col_k, val_k in seg["seg_vals"].items():
                        if col_k in df_safe_p.columns:
                            mask &= (
                                df_safe_p[col_k].astype(str) == str(val_k)
                            )
                    seg_df_check = df_with_fusion[mask]
                    sub_vals = (
                        seg_df_check[applied_col]
                        .dropna().astype(str).unique()
                    )
                    sub_labels = {
                        f"{seg['label']}_{safe_filename(v)}"
                        for v in sub_vals
                    }
                    if sub_labels and sub_labels.issubset(all_fused_labels):
                        segs_fully_fused.add(seg["label"])
                except Exception:
                    pass
        elif seg["label"] in all_fused_labels:
            segs_fully_fused.add(seg["label"])

    if segs_fully_fused:
        st.markdown("---")
        st.caption(
            f"ℹ️ {len(segs_fully_fused)} segment(s) entièrement inclus dans des fusions "
            f"— ils ne seront pas générés comme fichiers individuels dans le ZIP : "
            + ", ".join(sorted(segs_fully_fused))
        )

elif chunk_cols:
    st.info("Générez un template JSON valide pour voir le tableau des segments.")

st.markdown("---")

# ════════════════════════════════════════════════════════════
# SECTION 5 — Génération du ZIP
# ════════════════════════════════════════════════════════════
st.markdown("### 5️⃣ Générer le ZIP")

can_generate = (
    bool(chunk_cols)
    and bool(json_template.strip())
    and validation.get("valid", False)
)

if not can_generate:
    st.warning(
        "⚠️ Vérifiez que les colonnes de chunking sont sélectionnées "
        "et que le template JSON est valide."
    )
else:
    valid_cc = [c for c in chunk_cols if c in df_with_fusion.columns]
    try:
        df_safe_g = safe_fillna(df_with_fusion[valid_cc])
        n_segs_g  = df_safe_g.groupby(valid_cc).ngroups
        avg_s_g   = len(df) // n_segs_g if n_segs_g > 0 else 0
    except Exception:
        n_segs_g, avg_s_g = 0, 0

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.metric("📁 Segments", f"{n_segs_g:,}")
    with g2:
        st.metric("📏 Taille moy.", f"~{avg_s_g:,} lignes")
    with g3:
        n_fusions_actives = len([
            f for f in st.session_state.get("cj_fusions", [])
            if f.get("segments")
        ])
        st.metric("🔀 Fusions", f"{n_fusions_actives}")
    with g4:
        n_dec_actives = len(st.session_state.get("cj_decoupes", {}))
        st.metric("✂️ Découpes", f"{n_dec_actives}")

    if st.button(
        "🚀 Générer le ZIP",
        type="primary",
        use_container_width=True,
        key="cj_generate_btn",
    ):
        with st.spinner(f"Génération de {n_segs_g:,} fichiers JSON…"):
            try:
                filename_src = st.session_state.get("filename", "données")
                zip_bytes = generate_zip_advanced(
                    df=df_with_fusion,
                    chunk_cols=valid_cc,
                    template_str=json_template,
                    filename_source=filename_src,
                    fusions=st.session_state.get("cj_fusions", []),
                    decoupes=st.session_state.get("cj_decoupes", {}),
                    lignes_par_page=lignes_par_page_cj,
                )
                st.session_state["cj_zip_bytes"] = zip_bytes
                st.success(
                    f"✅ ZIP généré — {n_segs_g:,} segments"
                    + (f" · {n_fusions_actives} fusion(s)" if n_fusions_actives else "")
                    + (f" · {n_dec_actives} découpe(s)" if n_dec_actives else "")
                    + " + index.json + README.txt"
                )
            except Exception as e:
                st.error(f"Erreur génération ZIP : {e}")

    if st.session_state.get("cj_zip_bytes"):
        fname_src  = st.session_state.get("filename", "données").replace(".", "_")
        cols_label = "_".join(safe_filename(c)[:20] for c in valid_cc)
        st.download_button(
            label="📥 Télécharger le ZIP",
            data=st.session_state["cj_zip_bytes"],
            file_name=f"chunks_{fname_src}_{cols_label}.zip",
            mime="application/zip",
            use_container_width=True,
        )
