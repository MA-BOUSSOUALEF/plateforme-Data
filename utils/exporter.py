"""
utils/exporter.py
Fonctions d'export :
  - generate_html_report(df, profiles) → rapport HTML complet autonome
  - export_segments_zip(df, cols, profiles) → ZIP de CSV + index.json
  - export_decision_matrix(strategies) → CSV de la matrice de décision
"""
import io
import json
import zipfile
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import numpy as np


# ── Rapport HTML ──────────────────────────────────────────────────────────────

def generate_html_report(df: pd.DataFrame, profiles: dict, filename: str = "données") -> bytes:
    """
    Génère un rapport HTML complet et autonome (inline CSS/JS).
    Retourne les bytes du fichier HTML.
    """
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    n_rows, n_cols = len(df), len(df.columns)
    avg_missing = round(sum(p.pct_missing for p in profiles.values()) / max(len(profiles), 1), 1)
    n_excellent = sum(1 for p in profiles.values() if p.rag_status == "excellent")
    n_good = sum(1 for p in profiles.values() if p.rag_status == "good")

    # Score qualité global
    quality_score = _compute_quality_score(df, profiles)

    # Tableau des colonnes
    rows_html = ""
    STATUS_COLORS = {
        "excellent": "#3fb950", "good": "#58a6ff",
        "warning": "#d29922", "bad": "#f85149",
    }
    STATUS_LABELS = {
        "excellent": "🟢 Excellent", "good": "🔵 Bon",
        "warning": "🟡 Attention", "bad": "🔴 Inutilisable",
    }
    for col, p in sorted(profiles.items(), key=lambda x: x[1].rag_score, reverse=True):
        color = STATUS_COLORS.get(p.rag_status, "#8b949e")
        label = STATUS_LABELS.get(p.rag_status, "")
        sem = f" <small>({p.semantic_type})</small>" if p.semantic_type != "generic" else ""
        rows_html += f"""
        <tr>
          <td><strong>{col}</strong>{sem}</td>
          <td>{p.col_type}</td>
          <td>{p.n_unique:,}</td>
          <td>{p.pct_missing}%</td>
          <td>
            <div style="display:flex;align-items:center;gap:8px">
              <div style="width:60px;background:#21262d;border-radius:3px;height:8px">
                <div style="width:{p.rag_score}%;background:{color};height:8px;border-radius:3px"></div>
              </div>
              <span style="color:{color};font-weight:600">{p.rag_score}</span>
            </div>
          </td>
          <td style="color:{color}">{label}</td>
          <td style="color:#8b949e;font-size:0.85em">{p.rag_reason}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rapport Qualité — {filename}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0d1117; color: #c9d1d9; padding: 2rem; }}
    h1 {{ font-size: 1.8rem; font-weight: 700; color: #f0f6fc; margin-bottom: 0.3rem; }}
    h2 {{ font-size: 1.2rem; color: #8b949e; font-weight: 500; margin: 1.5rem 0 0.8rem; }}
    .subtitle {{ color: #8b949e; font-size: 0.9rem; margin-bottom: 2rem; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
    .metric {{ background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 1rem; }}
    .metric-value {{ font-size: 1.8rem; font-weight: 700; color: #f0f6fc; }}
    .metric-label {{ font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }}
    .quality-score {{ font-size: 3rem; font-weight: 800;
      color: {'#3fb950' if quality_score >= 70 else '#d29922' if quality_score >= 40 else '#f85149'}; }}
    table {{ width: 100%; border-collapse: collapse; background: #161b22; border-radius: 8px; overflow: hidden; }}
    th {{ background: #21262d; padding: 0.7rem 1rem; text-align: left; font-size: 0.75rem;
          text-transform: uppercase; letter-spacing: 0.08em; color: #8b949e; }}
    td {{ padding: 0.6rem 1rem; border-bottom: 1px solid #21262d; font-size: 0.9rem; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #1c2128; }}
    .footer {{ margin-top: 2rem; color: #8b949e; font-size: 0.8rem; text-align: center; }}
  </style>
</head>
<body>
  <h1>📊 Rapport Qualité — {filename}</h1>
  <p class="subtitle">Généré le {now} · {n_rows:,} lignes · {n_cols} colonnes</p>

  <div class="metrics">
    <div class="metric">
      <div class="quality-score">{quality_score}</div>
      <div class="metric-label">Score qualité global /100</div>
    </div>
    <div class="metric">
      <div class="metric-value">{n_rows:,}</div>
      <div class="metric-label">Lignes</div>
    </div>
    <div class="metric">
      <div class="metric-value">{n_cols}</div>
      <div class="metric-label">Colonnes</div>
    </div>
    <div class="metric">
      <div class="metric-value" style="color:#d29922">{avg_missing}%</div>
      <div class="metric-label">Manquants moyens</div>
    </div>
    <div class="metric">
      <div class="metric-value" style="color:#3fb950">{n_excellent}</div>
      <div class="metric-label">Colonnes excellentes RAG</div>
    </div>
    <div class="metric">
      <div class="metric-value" style="color:#58a6ff">{n_good}</div>
      <div class="metric-label">Colonnes bonnes RAG</div>
    </div>
  </div>

  <h2>📋 Profil des colonnes</h2>
  <table>
    <thead>
      <tr>
        <th>Colonne</th><th>Type</th><th>Valeurs uniques</th>
        <th>% Manquant</th><th>Score RAG</th><th>Statut</th><th>Raison</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>

  <div class="footer">
    RAG Segmentation Dashboard · Rapport automatique
  </div>
</body>
</html>"""

    return html.encode("utf-8")


# ── Export ZIP segments ───────────────────────────────────────────────────────

def export_segments_zip(
    df: pd.DataFrame,
    cols: List[str],
    profiles: dict,
    min_rows: int = 1,
) -> bytes:
    """
    Génère un ZIP contenant :
      - Un CSV par segment
      - index.json avec les métadonnées de chaque segment
    """
    from utils.profiler import safe_fillna

    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return b""

    df_safe = safe_fillna(df[valid_cols])
    grouped = df_safe.groupby(valid_cols)

    zip_buffer = io.BytesIO()
    index_data: List[Dict[str, Any]] = []

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for key, grp_safe in grouped:
            if len(grp_safe) < min_rows:
                continue

            if isinstance(key, tuple):
                label = " | ".join([str(k) for k in key])
                seg_vals = {c: str(k) for c, k in zip(valid_cols, key)}
            else:
                label = str(key)
                seg_vals = {valid_cols[0]: str(key)}

            # Récupérer les lignes originales
            mask = pd.Series([True] * len(df), index=df.index)
            for col, val in seg_vals.items():
                col_s = df_safe[col].astype(str)
                mask &= col_s == val
            seg_data = df[mask]

            fname = _safe_filename(label) + ".csv"

            # Métadonnées pour index.json
            num_stats: Dict[str, Any] = {}
            for c, p in profiles.items():
                if c in seg_data.columns and p.col_type == "numeric":
                    try:
                        s = seg_data[c].dropna().astype(float)
                        num_stats[c] = {
                            "mean": round(float(s.mean()), 4),
                            "min": round(float(s.min()), 4),
                            "max": round(float(s.max()), 4),
                        }
                    except Exception:
                        pass

            index_data.append({
                "segment": label,
                "filename": fname,
                "nb_lignes": len(seg_data),
                "pct_total": round(len(seg_data) / len(df) * 100, 2) if len(df) > 0 else 0,
                "colonnes_segment": seg_vals,
                "stats_numeriques": num_stats,
            })

            zf.writestr(fname, seg_data.to_csv(index=False))

        # Ajouter index.json
        zf.writestr("index.json", json.dumps(index_data, ensure_ascii=False, indent=2))

    return zip_buffer.getvalue()


# ── Export matrice décision ───────────────────────────────────────────────────

def export_decision_matrix(strategies: List[Dict[str, Any]]) -> bytes:
    """
    Exporte la matrice de décision RAG en CSV.
    strategies : liste de dicts avec clés standardisées.
    """
    if not strategies:
        return b""
    df_mat = pd.DataFrame(strategies)
    buf = io.StringIO()
    df_mat.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_filename(label: str, max_len: int = 80) -> str:
    """Transforme un label en nom de fichier sûr."""
    import re
    s = re.sub(r'[^\w\s\-]', '_', label)
    s = re.sub(r'[\s|]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s[:max_len]


def _compute_quality_score(df: pd.DataFrame, profiles: dict) -> int:
    """
    Score qualité global 0-100 basé sur :
      - Complétude (manquants)
      - Doublons
      - Cohérence des types
    """
    if len(profiles) == 0:
        return 0

    # Complétude : avg_missing → pénalité
    avg_missing = sum(p.pct_missing for p in profiles.values()) / len(profiles)
    score_completude = max(0, 100 - avg_missing * 2)  # 0% manquant = 100, 50% = 0

    # Doublons
    try:
        n_dupes = df.duplicated().sum()
        pct_dupes = n_dupes / len(df) * 100 if len(df) > 0 else 0
        score_dupes = max(0, 100 - pct_dupes * 5)
    except Exception:
        score_dupes = 80

    # Types : % colonnes avec type détecté proprement (pas tout "categorical")
    n_typed = sum(1 for p in profiles.values() if p.col_type != "text_free")
    score_types = round(n_typed / len(profiles) * 100)

    global_score = round(score_completude * 0.5 + score_dupes * 0.3 + score_types * 0.2)
    return min(100, max(0, global_score))
