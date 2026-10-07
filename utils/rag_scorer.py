"""
utils/rag_scorer.py
Scoring de combinaisons de colonnes pour la segmentation RAG.

Fonctions :
  estimate_pages(nb_lignes, lignes_par_page) → (nb_pages, depasse_limite)
  score_combination(df, cols, lignes_par_page) → dict
  scan_all_combinations(df, profiles, lignes_par_page) → List[dict]
"""
import itertools
import math
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from utils.profiler import safe_fillna, score_balance, get_rag_candidates


# ── Estimation du nombre de pages ────────────────────────────────────────────

def estimate_pages(nb_lignes: int, lignes_par_page: float) -> Tuple[int, bool]:
    """
    Retourne (nb_pages_estimé, depasse_limite_300).
    lignes_par_page : nb de lignes Excel = 1 page PDF (défini par l'utilisateur).
    """
    if lignes_par_page <= 0:
        lignes_par_page = 1.0
    nb_pages = math.ceil(nb_lignes / lignes_par_page)
    return nb_pages, nb_pages > 300


# ── Score d'une combinaison ───────────────────────────────────────────────────

def score_combination(
    df: pd.DataFrame,
    cols: List[str],
    lignes_par_page: float,
) -> Dict[str, Any]:
    """
    Calcule le score (0-100) d'une combinaison de colonnes pour la segmentation RAG,
    en tenant compte de la contrainte de 300 pages par segment.

    Retourne un dict ou {} si la combinaison est invalide.
    """
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return {}

    try:
        df_safe = safe_fillna(df[valid_cols])
        grouped = df_safe.groupby(valid_cols)
    except Exception:
        return {}

    segments: List[Dict[str, Any]] = []
    for _, grp in grouped:
        nb_lignes = len(grp)
        nb_pages, depasse = estimate_pages(nb_lignes, lignes_par_page)
        segments.append({"nb_lignes": nb_lignes, "nb_pages": nb_pages, "depasse": depasse})

    if not segments:
        return {}

    nb_segments  = len(segments)
    sizes        = [s["nb_lignes"] for s in segments]
    pages_list   = [s["nb_pages"]  for s in segments]

    taille_moyenne = float(np.mean(sizes))
    taille_min     = int(min(sizes))
    taille_max     = int(max(sizes))
    pages_moyenne  = float(np.mean(pages_list))
    pages_max      = int(max(pages_list))

    n_hors_limite  = sum(1 for s in segments if s["depasse"])
    n_trop_petits  = sum(1 for s in sizes     if s < 10)

    pct_hors_limite = round(n_hors_limite / nb_segments * 100, 1)
    pct_trop_petits = round(n_trop_petits / nb_segments * 100, 1)

    # ── Score d'équilibre ────────────────────────────────────────────────────
    try:
        if len(valid_cols) == 1:
            equil = score_balance(df_safe[valid_cols[0]])
        else:
            combined = df_safe[valid_cols].astype(str).agg(" | ".join, axis=1)
            equil = score_balance(combined)
    except Exception:
        equil = 0.0

    # ── Calcul du score composite (max théorique = 100) ──────────────────────
    score = 0.0

    # a. Équilibre (max 30)
    score += equil * 30

    # b. Contrainte pages (max 30)
    if pct_hors_limite == 0:
        score += 30
    elif pct_hors_limite <= 10:
        score += 15
    # >10% → 0

    # c. Nb segments raisonnable (max 20)
    if 3 <= nb_segments <= 50:
        score += 20
    elif 51 <= nb_segments <= 200:
        score += 15
    elif 201 <= nb_segments <= 500:
        score += 10
    # >500 → 0

    # d. Qualité segments (max 20)
    if pct_trop_petits <= 5:
        score += 20
    elif pct_trop_petits <= 20:
        score += 10
    # >20% → 0

    score_final = min(100, max(0, round(score)))

    if score_final >= 72:
        statut = "excellent"
    elif score_final >= 48:
        statut = "bon"
    elif score_final >= 28:
        statut = "attention"
    else:
        statut = "problématique"

    return {
        "cols":           valid_cols,
        "label":          " × ".join(valid_cols),
        "score":          score_final,
        "nb_segments":    nb_segments,
        "taille_moyenne": round(taille_moyenne, 1),
        "taille_min":     taille_min,
        "taille_max":     taille_max,
        "pages_moyenne":  round(pages_moyenne, 1),
        "pages_max":      pages_max,
        "pct_hors_limite": pct_hors_limite,
        "pct_trop_petits": pct_trop_petits,
        "alerte_pages":   pct_hors_limite > 0,
        "statut":         statut,
    }


# ── Scan toutes combinaisons ──────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def scan_all_combinations(
    df: pd.DataFrame,
    profiles: dict,
    lignes_par_page: float,
) -> List[Dict[str, Any]]:
    """
    Génère et score toutes les combinaisons 1, 2 et 3 colonnes parmi les
    meilleures candidates RAG (score ≥ 45, max 8 candidats).

    Cache invalidé automatiquement si df, profiles ou lignes_par_page changent.
    Retourne les 20 meilleures combinaisons triées par score décroissant.
    """
    candidates = get_rag_candidates(profiles, min_score=45)
    if len(candidates) > 8:
        candidates = candidates[:8]
    if not candidates:
        return []

    # Générer les combinaisons
    combos: List[List[str]] = []
    for c in candidates:
        combos.append([c])
    for c1, c2 in itertools.combinations(candidates, 2):
        combos.append([c1, c2])
    for c1, c2, c3 in itertools.combinations(candidates, 3):
        combos.append([c1, c2, c3])

    results: List[Dict[str, Any]] = []
    for cols in combos:
        try:
            res = score_combination(df, cols, lignes_par_page)
            if res:
                results.append(res)
        except Exception:
            pass

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:20]
