"""
utils/profiler.py
Profilage complet d'un DataFrame :
  - Détection de type (categorical, numeric, date, text_free, id, binary)
  - Détection sémantique (code postal, téléphone, email, URL, SIRET)
  - Score RAG par colonne (potentiel de segmentation 0-100)
  - Fonctions d'analyse : cramers_v, detect_outliers, score_balance
  - safe_fillna : compatible colonnes category
  - Thème graphique DARK_LAYOUT partagé
"""
import re
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, List, Optional, Tuple
import streamlit as st


# ── Thème dark partagé ────────────────────────────────────────────────────────
DARK_LAYOUT = dict(
    plot_bgcolor="#0d1117",
    paper_bgcolor="#0d1117",
    font=dict(color="#c9d1d9"),
)
DARK = DARK_LAYOUT  # alias pour compatibilité pages

COLORS = {
    "accent":  "#58a6ff",
    "success": "#3fb950",
    "warning": "#d29922",
    "danger":  "#f85149",
    "surface": "#161b22",
    "border":  "#21262d",
    "text":    "#c9d1d9",
    "muted":   "#8b949e",
}

# ── Types ─────────────────────────────────────────────────────────────────────
ColType     = Literal["categorical", "numeric", "date", "text_free", "id", "binary"]
SemanticType = Literal["postal", "phone", "email", "url", "siret", "generic"]
RAG_STATUS  = Literal["excellent", "good", "warning", "bad"]


# ── safe_fillna ───────────────────────────────────────────────────────────────
def safe_fillna(df: pd.DataFrame, value: str = "(non renseigné)") -> pd.DataFrame:
    """
    fillna compatible avec les colonnes de type category.
    Ajoute la valeur aux catégories avant de remplir.
    """
    df = df.copy()
    for col in df.columns:
        try:
            if hasattr(df[col], "cat"):
                if value not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories(value)
            df[col] = df[col].fillna(value)
        except Exception:
            pass
    return df


def safe_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit les colonnes 'category' en object (str) pour éviter l'erreur
    PyArrow ArrowInvalid lors de st.dataframe() avec des colonnes catégorielles.
    """
    out = df.copy()
    for col in out.columns:
        if hasattr(out[col], "cat"):
            out[col] = out[col].astype(object)
    return out


# ── ColumnProfile ─────────────────────────────────────────────────────────────
@dataclass
class ColumnProfile:
    name: str
    col_type: ColType
    semantic_type: SemanticType
    n_unique: int
    pct_missing: float
    rag_status: RAG_STATUS
    rag_score: int
    rag_reason: str
    top_values: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# ── Profilage principal ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def profile_dataframe(df: pd.DataFrame) -> dict:
    """Profiler toutes les colonnes d'un DataFrame. Résultat mis en cache."""
    profiles = {}
    n = len(df)

    for col in df.columns:
        try:
            series = df[col]
            n_unique = int(series.nunique())
            pct_missing = round(float(series.isna().mean() * 100), 1)
            col_type = _detect_type(series, n_unique, n)
            semantic = _detect_semantic(series, col_type)
            top_values = _get_top_values(series, col_type)
            stats = _compute_stats(series, col_type)
            rag_score, rag_status, rag_reason = _score_rag(
                col_type, n_unique, pct_missing, n,
                series=series, n_rows_total=n,
            )

            profiles[col] = ColumnProfile(
                name=col,
                col_type=col_type,
                semantic_type=semantic,
                n_unique=n_unique,
                pct_missing=pct_missing,
                rag_status=rag_status,
                rag_score=rag_score,
                rag_reason=rag_reason,
                top_values=top_values,
                stats=stats,
            )
        except Exception:
            profiles[col] = ColumnProfile(
                name=col, col_type="categorical", semantic_type="generic",
                n_unique=0, pct_missing=100.0, rag_status="bad",
                rag_score=0, rag_reason="Erreur lors du profilage",
            )

    return profiles


# ── Détection de type ─────────────────────────────────────────────────────────
def _detect_type(series: pd.Series, n_unique: int, n: int) -> ColType:
    dtype = series.dtype

    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "date"

    if pd.api.types.is_numeric_dtype(dtype):
        if n_unique <= 2:
            return "binary"
        return "numeric"

    if n_unique <= 2:
        return "binary"

    if n > 0 and n_unique / n > 0.85:
        return "id"

    # Tenter détection date sur object
    if dtype == object or str(dtype) == "category":
        sample = series.dropna().astype(str).head(200)
        try:
            parsed = pd.to_datetime(sample, dayfirst=True, errors="coerce")
            if parsed.notna().mean() >= 0.80:
                return "date"
        except Exception:
            pass

        avg_len = series.dropna().astype(str).str.len().mean() if len(series.dropna()) > 0 else 0
        if avg_len > 60:
            return "text_free"
        return "categorical"

    return "categorical"


# ── Détection sémantique ──────────────────────────────────────────────────────
def _detect_semantic(series: pd.Series, col_type: ColType) -> SemanticType:
    """Identifie si la colonne contient un type sémantique connu."""
    if col_type not in ("categorical", "id", "text_free"):
        return "generic"
    try:
        sample = series.dropna().astype(str).head(50).tolist()
        if not sample:
            return "generic"

        checks = {
            "postal":  r"^\d{5}$",
            "phone":   r"^(\+?\d[\s\-.]?){7,15}$",
            "email":   r"^[\w\.\+\-]+@[\w\-]+\.\w{2,}$",
            "url":     r"^https?://",
            "siret":   r"^\d{14}$",
        }
        for sem, pattern in checks.items():
            matches = sum(1 for v in sample if re.match(pattern, v.strip()))
            if matches / len(sample) >= 0.60:
                return sem  # type: ignore
    except Exception:
        pass
    return "generic"


# ── Top valeurs ───────────────────────────────────────────────────────────────
def _get_top_values(series: pd.Series, col_type: ColType) -> list:
    if col_type in ("categorical", "binary"):
        try:
            vc = series.value_counts(dropna=False).head(10)
            return [{"value": str(k), "count": int(v)} for k, v in vc.items()]
        except Exception:
            pass
    return []


# ── Stats numériques ──────────────────────────────────────────────────────────
def _compute_stats(series: pd.Series, col_type: ColType) -> dict:
    if col_type == "numeric":
        try:
            s = series.dropna().astype(float)
            desc = s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            return {
                "mean":  round(float(desc["mean"]), 4),
                "min":   round(float(desc["min"]), 4),
                "max":   round(float(desc["max"]), 4),
                "std":   round(float(desc["std"]), 4),
                "p5":    round(float(desc.get("5%", desc["min"])), 4),
                "p25":   round(float(desc["25%"]), 4),
                "p50":   round(float(desc["50%"]), 4),
                "p75":   round(float(desc["75%"]), 4),
                "p95":   round(float(desc.get("95%", desc["max"])), 4),
            }
        except Exception:
            pass
    return {}


# ── Score RAG ─────────────────────────────────────────────────────────────────
def _score_rag(
    col_type: ColType,
    n_unique: int,
    pct_missing: float,
    n: int,
    series: Optional[pd.Series] = None,
    n_rows_total: Optional[int] = None,
) -> Tuple[int, RAG_STATUS, str]:
    """
    Score RAG sur 5 critères (max brut = 125), normalisé sur 100.

    Critère 1 — Type          : max 40 pts
    Critère 2 — Cardinalité   : max 40 pts
    Critère 3 — Complétude    : max 20 pts  (peut être négatif)
    Critère 4 — Équilibre     : max 15 pts  (nécessite series)
    Critère 5 — Taille segs   : max 10 pts  (nécessite n_rows_total)
    """
    score_brut = 0
    reasons: List[str] = []

    # ── Critère 1 : Type (max 40) ──────────────────────────────────────────────
    type_scores = {
        "categorical": 40, "binary": 30, "date": 15,
        "numeric": 10, "id": 0, "text_free": 0,
    }
    score_brut += type_scores.get(col_type, 0)
    if col_type == "binary":
        reasons.append("binaire (2 valeurs)")
    elif col_type == "numeric":
        reasons.append("numérique (découpe par tranche possible)")
    elif col_type in ("id", "text_free"):
        reasons.append("trop unique / texte libre → inutilisable")
    elif col_type == "date":
        reasons.append("date (découpe temporelle possible)")

    # ── Critère 2 : Cardinalité (max 40) ──────────────────────────────────────
    if 2 <= n_unique <= 10:
        score_brut += 40
        reasons.append(f"{n_unique} valeurs → parfait")
    elif 11 <= n_unique <= 30:
        score_brut += 30
        reasons.append(f"{n_unique} valeurs → bon")
    elif 31 <= n_unique <= 100:
        score_brut += 15
        reasons.append(f"{n_unique} valeurs → granularité fine")
    else:
        reasons.append(f"{n_unique} valeurs → trop fragmenté")

    # ── Critère 3 : Complétude (max 20, min -20) ──────────────────────────────
    if pct_missing <= 5:
        score_brut += 20
    elif pct_missing <= 20:
        score_brut += 10
        reasons.append(f"{pct_missing}% manquants")
    elif pct_missing <= 50:
        reasons.append(f"⚠️ {pct_missing}% manquants")
    else:
        score_brut -= 20
        reasons.append(f"❌ {pct_missing}% manquants → inutilisable")

    # ── Critère 4 : Équilibre distribution (max 15) ───────────────────────────
    if series is not None:
        try:
            bal = score_balance(series)
            if bal >= 0.80:
                score_brut += 15
            elif bal >= 0.50:
                score_brut += 10
            elif bal >= 0.25:
                score_brut += 5
            if bal < 0.50:
                reasons.append(f"équilibre {round(bal * 100)}%")
        except Exception:
            pass

    # ── Critère 5 : Taille estimée des segments (max 10) ──────────────────────
    if n_rows_total is not None and n_unique > 0:
        taille_moy = n_rows_total / n_unique
        if 50 <= taille_moy <= 900:
            score_brut += 10
        elif 901 <= taille_moy <= 2_000:
            score_brut += 5
            reasons.append(f"~{round(taille_moy):,} lignes/segment")
        elif 2_001 <= taille_moy <= 5_000:
            score_brut += 2
            reasons.append(f"~{round(taille_moy):,} lignes/segment")
        elif taille_moy < 50:
            reasons.append(f"~{round(taille_moy):,} lignes/segment → trop petit")
        else:
            reasons.append(f"~{round(taille_moy):,} lignes/segment → trop gros")

    # ── Normalisation sur 100 (max brut théorique = 125) ──────────────────────
    score_final = min(100, max(0, round(score_brut * 100 / 125)))

    # ── Statut ────────────────────────────────────────────────────────────────
    if score_final >= 72:
        status: RAG_STATUS = "excellent"
    elif score_final >= 48:
        status = "good"
    elif score_final >= 28:
        status = "warning"
    else:
        status = "bad"

    reason = " · ".join(reasons) if reasons else "Bon candidat"
    return score_final, status, reason


# ── Fonctions d'analyse avancée ───────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_cramers_v(s1: pd.Series, s2: pd.Series) -> float:
    """
    Calcule le V de Cramér entre deux colonnes catégorielles.
    Retourne une valeur entre 0 (indépendance) et 1 (association parfaite).
    """
    try:
        from scipy.stats import chi2_contingency
        ct = pd.crosstab(s1.astype(str).fillna("_NA_"), s2.astype(str).fillna("_NA_"))
        chi2, _, _, _ = chi2_contingency(ct)
        n = ct.sum().sum()
        min_dim = min(ct.shape) - 1
        if min_dim <= 0 or n == 0:
            return 0.0
        v = float(np.sqrt(chi2 / (n * min_dim)))
        return round(min(v, 1.0), 4)
    except Exception:
        return 0.0


def detect_outliers(series: pd.Series, method: str = "iqr") -> pd.Index:
    """
    Détecte les outliers dans une série numérique.
    method = "iqr"    : règle IQR (1.5 * IQR)
    method = "zscore" : Z-score > 3
    Retourne l'index des lignes anormales.
    """
    try:
        s = series.dropna().astype(float)
        if len(s) < 4:
            return pd.Index([])

        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (series < lo) | (series > hi)
        else:  # zscore
            mean, std = s.mean(), s.std()
            if std == 0:
                return pd.Index([])
            z = (series.astype(float) - mean) / std
            mask = z.abs() > 3

        return series[mask & series.notna()].index
    except Exception:
        return pd.Index([])


def score_balance(series: pd.Series) -> float:
    """
    Entropie de Shannon normalisée (0 = déséquilibre total, 1 = parfait équilibre).
    Mesure l'équilibre de distribution d'une colonne catégorielle.
    """
    try:
        vc = series.value_counts(normalize=True, dropna=True)
        if len(vc) <= 1:
            return 0.0
        entropy = -float(np.sum(vc * np.log2(vc + 1e-12)))
        max_entropy = float(np.log2(len(vc)))
        if max_entropy == 0:
            return 0.0
        return round(entropy / max_entropy, 4)
    except Exception:
        return 0.0


# ── Helpers page ─────────────────────────────────────────────────────────────

def get_rag_candidates(profiles: dict, min_score: int = 45) -> List[str]:
    candidates = [p for p in profiles.values() if p.rag_score >= min_score]
    candidates.sort(key=lambda p: p.rag_score, reverse=True)
    return [c.name for c in candidates]


def simulate_rag_split(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame()
    try:
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            return pd.DataFrame()
        df_safe = safe_fillna(df[valid_cols])
        grouped = df_safe.groupby(valid_cols)
        records = []
        for key, grp in grouped:
            label = " | ".join([str(k) for k in key]) if isinstance(key, tuple) else str(key)
            records.append({
                "segment": label,
                "nb_lignes": len(grp),
                "pct_total": round(len(grp) / len(df) * 100, 1),
            })
        return pd.DataFrame(records).sort_values("nb_lignes", ascending=False)
    except Exception:
        return pd.DataFrame()
