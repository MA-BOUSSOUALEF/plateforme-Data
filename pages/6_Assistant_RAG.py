"""
pages/6_Assistant_RAG.py
Assistant Décision RAG — rapport automatique (logique Python pure),
matrice de décision interactive avec pondérations, scatter stratégies.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Assistant RAG", page_icon="🤖", layout="wide")

from utils.guards import check_data_loaded
from utils.profiler import DARK_LAYOUT as DARK

_, df, profiles = check_data_loaded()

st.markdown("# 🤖 Assistant Décision RAG")
st.markdown("Rapport automatique, matrice de décision pondérée, comparaison visuelle des stratégies.")

with st.expander("💡 Comment utiliser cette page", expanded=False):
    st.markdown("""
- **Rapport automatique** : analyse des meilleures stratégies de segmentation avec justifications.
- **Matrice de décision** : tableau interactif avec sliders de pondération recalculés en temps réel.
- **Scatter stratégies** : visualisation nb segments × taille moyenne, cliquer pour détails.
- Aucun LLM — 100% logique Python reproductible.
    """)

from utils.profiler import get_rag_candidates
from utils.rag_scorer import scan_all_combinations, estimate_pages

candidates      = get_rag_candidates(profiles, min_score=25)
all_cols        = list(df.columns)
lignes_par_page = float(st.session_state.get("lignes_par_page", 3))

st.markdown("---")

# ════════════════════════════════════════════════════════════
# Scan unique via scan_all_combinations
# ════════════════════════════════════════════════════════════
with st.spinner("Analyse de toutes les combinaisons candidates…"):
    try:
        scan_results = scan_all_combinations(df, profiles, lignes_par_page)
    except Exception as e:
        st.error(f"Erreur scan combinaisons : {e}")
        st.stop()

if not scan_results:
    st.warning("Impossible de calculer des stratégies — vérifiez le fichier chargé.")
    st.stop()

# ════════════════════════════════════════════════════════════
# SECTION A — Rapport automatique
# ════════════════════════════════════════════════════════════
st.markdown("## 📄 Rapport automatique")

top1 = scan_results[0]

# Risques
risks = []
if top1["pct_trop_petits"] > 10:
    risks.append(f"⚠️ {top1['pct_trop_petits']:.0f}% de segments trop petits avec `{top1['label']}`")
if top1["pct_hors_limite"] > 0:
    risks.append(
        f"⚠️ La meilleure combinaison `{top1['label']}` a "
        f"{top1['pct_hors_limite']:.0f}% de segments dépassant 300 pages"
    )
if top1["nb_segments"] > 200:
    risks.append(f"⚠️ Trop de segments ({top1['nb_segments']}) — risque de fragmentation excessive")

conf_score = top1["score"]
conf_color = "#3fb950" if conf_score >= 72 else "#d29922" if conf_score >= 48 else "#f85149"

rap_c1, rap_c2 = st.columns([3, 1])
with rap_c1:
    statut_icon = {"excellent": "✅", "bon": "🔵", "attention": "⚠️", "problématique": "❌"}.get(
        top1["statut"], "📊"
    )
    st.markdown(f"""
**{statut_icon} Meilleure combinaison :** `{top1['label']}`
- **{top1['nb_segments']}** segments · taille moy. **{top1['taille_moyenne']:.0f}** lignes · \
**~{top1['pages_moyenne']:.0f} pages** en moyenne · {top1['pct_hors_limite']:.0f}% hors limite
""")

    st.markdown("#### 🥇 Top 3 combinaisons")
    for i, r in enumerate(scan_results[:3]):
        icon = ["🥇", "🥈", "🥉"][i]
        si = {"excellent": "✅", "bon": "🔵", "attention": "⚠️", "problématique": "❌"}.get(r["statut"], "📊")
        pros, cons = [], []
        if r["pct_hors_limite"] == 0:   pros.append("aucun segment hors limite")
        if r["pct_trop_petits"] == 0:   pros.append("aucun segment trop petit")
        if 5 <= r["nb_segments"] <= 100: pros.append("nb segments raisonnable")
        if r["pct_hors_limite"] > 20:    cons.append(f"{r['pct_hors_limite']:.0f}% hors limite")
        if r["nb_segments"] > 200:       cons.append("trop fragmenté")
        st.markdown(
            f"{icon} {si} **`{r['label']}`** — Score {r['score']}/100 · "
            f"{r['nb_segments']} segments · taille moy. {r['taille_moyenne']:.0f} · "
            f"~{r['pages_moyenne']:.0f} pages moy."
        )
        if pros: st.markdown(f"   ✅ {' · '.join(pros)}")
        if cons: st.markdown(f"   ⚠️ {' · '.join(cons)}")

    if risks:
        st.markdown("**⚠️ Risques identifiés :**")
        for r in risks:
            st.markdown(f"  {r}")
    else:
        st.markdown("**✅ Aucun risque majeur identifié.**")

with rap_c2:
    fig_conf = go.Figure(go.Indicator(
        mode="gauge+number",
        value=conf_score,
        title={"text": "Score de confiance", "font": {"color": "#c9d1d9", "size": 12}},
        number={"font": {"color": conf_color, "size": 36}, "suffix": "%"},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "#8b949e"}},
            "bar": {"color": conf_color, "thickness": 0.25},
            "bgcolor": "#21262d", "borderwidth": 0,
            "steps": [
                {"range": [0, 28],  "color": "#2a1a1a"},
                {"range": [28, 48], "color": "#2a2a1a"},
                {"range": [48, 72], "color": "#1a2a3a"},
                {"range": [72, 100],"color": "#1a3a1a"},
            ],
        },
    ))
    fig_conf.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10), **DARK)
    st.plotly_chart(fig_conf, use_container_width=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════
# SECTION B — Matrice de décision interactive
# ════════════════════════════════════════════════════════════
st.markdown("## 📊 Matrice de décision interactive")
st.caption("Ajustez les pondérations — le classement se recalcule en temps réel.")

w_col1, w_col2, w_col3, w_col4 = st.columns(4)
with w_col1:
    w_equil = st.slider("Équilibre", 0, 10, 4, key="w_equil")
with w_col2:
    w_pages = st.slider("Contrainte pages", 0, 10, 4, key="w_pages")
with w_col3:
    w_nbseg = st.slider("Nb segments", 0, 10, 2, key="w_nbseg")
with w_col4:
    w_petits = st.slider("Pénalité petits", 0, 10, 2, key="w_petits")

total_w = w_equil + w_pages + w_nbseg + w_petits or 1

try:
    mat_rows = []
    for r in scan_results[:20]:
        norm_equil  = r["score"] / 100
        norm_pages  = 1.0 - min(r["pct_hors_limite"] / 100, 1.0)
        norm_nseg   = min(r["nb_segments"], 100) / 100
        norm_petits = 1.0 - min(r["pct_trop_petits"] / 100, 1.0)

        weighted = (
            norm_equil  * w_equil
            + norm_pages  * w_pages
            + norm_nseg   * w_nbseg
            + norm_petits * w_petits
        ) / total_w * 100

        mat_rows.append({
            "Combinaison":    r["label"],
            "Nb PDFs":        r["nb_segments"],
            "Taille moy.":    round(r["taille_moyenne"], 0),
            "Pages moy.":     round(r["pages_moyenne"], 0),
            "% hors limite":  round(r["pct_hors_limite"], 1),
            "% petits":       round(r["pct_trop_petits"], 1),
            "Score RAG":      r["score"],
            "Score pondéré":  round(weighted, 1),
        })

    mat_df = pd.DataFrame(mat_rows).sort_values("Score pondéré", ascending=False).reset_index(drop=True)

    def color_score_w(val):
        if val >= 70:   return "background-color:#1a3a1a;color:#3fb950"
        elif val >= 45: return "background-color:#1a2a3a;color:#58a6ff"
        elif val >= 25: return "background-color:#2a2a1a;color:#d29922"
        else:           return "background-color:#2a1a1a;color:#f85149"

    styled = mat_df.style.map(color_score_w, subset=["Score pondéré"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=420)

    from utils.exporter import export_decision_matrix
    csv_mat = export_decision_matrix(mat_rows)
    st.download_button("📥 Exporter matrice CSV", csv_mat,
                       file_name="matrice_decision_rag.csv", mime="text/csv")
except Exception as e:
    st.error(f"Erreur matrice décision : {e}")

st.markdown("---")

# ════════════════════════════════════════════════════════════
# SECTION C — Scatter stratégies
# ════════════════════════════════════════════════════════════
st.markdown("## 🔵 Scatter des stratégies")
st.caption("Axe X = nb segments · Axe Y = taille moyenne · Taille bulle = score · Couleur = pages moy.")

try:
    scat_data = pd.DataFrame(scan_results[:20])
    if len(scat_data) > 0:
        fig_scat = px.scatter(
            scat_data,
            x="nb_segments", y="taille_moyenne",
            size="score", color="pages_moyenne",
            hover_name="label",
            hover_data={
                "nb_segments": True, "taille_moyenne": True,
                "pages_moyenne": True, "pct_hors_limite": True,
                "score": True,
            },
            text="label",
            color_continuous_scale="RdYlGn_r",
            size_max=40,
        )
        fig_scat.update_traces(
            textposition="top center",
            textfont=dict(size=9, color="#c9d1d9"),
            marker=dict(line=dict(color="#21262d", width=1)),
        )
        fig_scat.update_layout(
            height=500,
            margin=dict(l=10, r=10, t=40, b=20),
            xaxis_title="Nombre de segments",
            yaxis_title="Taille moyenne (lignes)",
            coloraxis_colorbar=dict(title="Pages moy."),
            **DARK,
        )
        fig_scat.add_hrect(y0=50, y1=5000, fillcolor="rgba(63,185,80,0.04)",
                           line_width=0, annotation_text="Zone RAG idéale",
                           annotation_font_color="#3fb950")
        st.plotly_chart(fig_scat, use_container_width=True,
                        config={"toImageButtonOptions": {"format": "png", "filename": "scatter_strategies"}})

        selected_label = st.selectbox(
            "Détails d'une combinaison",
            [r["label"] for r in scan_results[:15]],
            key="detail_strat",
        )
        chosen = next((r for r in scan_results if r["label"] == selected_label), None)
        if chosen:
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Nb segments",   chosen["nb_segments"])
            sc2.metric("Taille moy.",   f"{chosen['taille_moyenne']:.0f}")
            sc3.metric("Pages moy.",    f"{chosen['pages_moyenne']:.0f}")
            sc4.metric("Score",         f"{chosen['score']}/100")

            if st.button(f"✅ Utiliser `{selected_label}` dans le Simulateur", type="primary"):
                valid = [c for c in chosen["cols"] if c in df.columns]
                if valid:
                    st.session_state["rag_selected_cols"] = valid
                    st.session_state["sim_cols_multi"]    = valid
                    st.success(f"Combinaison `{selected_label}` configurée — allez sur la page Simulateur RAG.")
except Exception as e:
    st.error(f"Erreur scatter : {e}")
