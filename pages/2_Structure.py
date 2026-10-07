"""
pages/2_Structure.py
Structure & Hiérarchie — Treemap + Sunburst (onglets), Radar chart RAG,
tableau de recommandations automatiques, distribution d'une colonne.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Structure", page_icon="🗂️", layout="wide")

from utils.guards import check_data_loaded
from utils.profiler import DARK_LAYOUT as DARK

_, df, profiles = check_data_loaded()

st.markdown("# 🗂️ Structure & Hiérarchie")
st.markdown("Explorez l'arborescence, identifiez les candidats RAG et obtenez des recommandations automatiques.")

with st.expander("💡 Comment utiliser cette page", expanded=False):
    st.markdown("""
- **Cardinalité** : repérez la zone verte (2–10 valeurs) = meilleurs candidats RAG.
- **Treemap / Sunburst** : visualisez les volumes par branche hiérarchique (2–3 colonnes).
- **Radar chart** : comparaison des top 8 colonnes candidates sur plusieurs critères.
- **Recommandations** : tableau automatique Pour / Contre / Conseil par colonne.
    """)

# ── Filtres avancés ───────────────────────────────────────────────────────────
from utils.guards import apply_page_filters

df_page = apply_page_filters(df, profiles, "page2")

st.markdown("---")

# ── Cardinalité ───────────────────────────────────────────────────────────────
st.markdown("### 📊 Cardinalité par colonne")
try:
    card_data = [(col, p.n_unique, p.rag_status, p.rag_score) for col, p in profiles.items()]
    card_df = pd.DataFrame(card_data, columns=["Colonne", "Valeurs uniques", "Statut", "Score"])
    card_df = card_df.sort_values("Valeurs uniques")

    color_map = {"excellent": "#3fb950", "good": "#58a6ff", "warning": "#d29922", "bad": "#f85149"}
    colors = card_df["Statut"].map(color_map).tolist()

    fig_card = go.Figure(go.Bar(
        x=card_df["Valeurs uniques"], y=card_df["Colonne"], orientation="h",
        marker_color=colors,
        text=card_df["Valeurs uniques"].apply(lambda x: f"{x:,}"),
        textposition="outside",
        customdata=card_df[["Score", "Statut"]].values,
        hovertemplate="<b>%{y}</b><br>%{x:,} valeurs uniques<br>Score RAG: %{customdata[0]}<extra></extra>",
    ))
    fig_card.add_vline(x=10, line_dash="dot", line_color="#3fb950",
                       annotation_text="≤10 🟢", annotation_position="top")
    fig_card.add_vline(x=50, line_dash="dot", line_color="#d29922",
                       annotation_text="≤50 🟡", annotation_position="top")
    fig_card.update_layout(
        height=max(320, len(card_df) * 24),
        margin=dict(l=10, r=100, t=40, b=20),
        xaxis=dict(title="Valeurs uniques", type="log"),
        yaxis=dict(autorange="reversed"),
        showlegend=False, **DARK,
    )
    st.plotly_chart(fig_card, use_container_width=True,
                    config={"toImageButtonOptions": {"format": "png", "filename": "cardinalite"}})

    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.markdown("🟢 **Excellent** (score ≥ 70)")
    lc2.markdown("🔵 **Bon** (score ≥ 45)")
    lc3.markdown("🟡 **Attention** (score ≥ 25)")
    lc4.markdown("🔴 **Inutilisable**")
except Exception as e:
    st.error(f"Erreur cardinalité : {e}")

st.markdown("---")

# ── Treemap & Sunburst ────────────────────────────────────────────────────────
st.markdown("### 🌳 Visualisation hiérarchique")

cat_cols = [
    col for col, p in profiles.items()
    if p.col_type in ("categorical", "binary") and p.n_unique <= 200 and col in df_page.columns
]
num_cols = [col for col, p in profiles.items() if p.col_type == "numeric" and col in df_page.columns]

sc1, sc2, sc3 = st.columns([3, 2, 1])
with sc1:
    selected_levels = st.multiselect(
        "Colonnes hiérarchiques (ordre = niveau)",
        options=cat_cols,
        default=cat_cols[:2] if len(cat_cols) >= 2 else cat_cols,
        max_selections=3,
        help="2–3 colonnes formant une hiérarchie naturelle",
    )
with sc2:
    value_col_opts = ["(compter les lignes)"] + num_cols
    value_col = st.selectbox("Valeur à afficher", value_col_opts, key="struct_val_col")
with sc3:
    sample_note = ""
    n_sample = len(df_page)
    if n_sample > 50_000:
        n_sample = 50_000
        sample_note = f"📊 Échantillon {n_sample:,} lignes"

viz_tab1, viz_tab2 = st.tabs(["🌳 Treemap", "☀️ Sunburst"])

if len(selected_levels) >= 2:
    try:
        sample_df = df_page.iloc[:n_sample][selected_levels].copy()
        for c in selected_levels:
            if hasattr(sample_df[c], "cat"):
                sample_df[c] = sample_df[c].astype(str)
        sample_df = sample_df.fillna("(non renseigné)")

        if value_col == "(compter les lignes)":
            sample_df["_val"] = 1
            val_name, val_label = "_val", "Nb lignes"
        else:
            sample_df[value_col] = df_page.iloc[:n_sample][value_col].values
            val_name, val_label = value_col, value_col

        common_kwargs = dict(
            path=selected_levels, values=val_name,
            color=val_name, color_continuous_scale="Blues",
        )
        hover_tmpl = "<b>%{label}</b><br>" + val_label + ": %{value:,}<br>Part: %{percentRoot:.1%}<extra></extra>"

        with viz_tab1:
            if sample_note:
                st.caption(sample_note)
            fig_tree = px.treemap(sample_df, **common_kwargs)
            fig_tree.update_traces(textinfo="label+value+percent root", hovertemplate=hover_tmpl)
            fig_tree.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10),
                                   paper_bgcolor="#0d1117", font=dict(color="#c9d1d9"),
                                   coloraxis_showscale=False)
            st.plotly_chart(fig_tree, use_container_width=True,
                            config={"toImageButtonOptions": {"format": "png", "filename": "treemap"}})
            st.caption(f"Hiérarchie : {' → '.join(selected_levels)}")

        with viz_tab2:
            if sample_note:
                st.caption(sample_note)
            fig_sun = px.sunburst(sample_df, **common_kwargs)
            fig_sun.update_traces(hovertemplate=hover_tmpl)
            fig_sun.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10),
                                  paper_bgcolor="#0d1117", font=dict(color="#c9d1d9"),
                                  coloraxis_showscale=False)
            st.plotly_chart(fig_sun, use_container_width=True,
                            config={"toImageButtonOptions": {"format": "png", "filename": "sunburst"}})

    except Exception as e:
        st.error(f"Erreur visualisation hiérarchique : {e}")
elif len(selected_levels) == 1:
    st.info("Sélectionnez au moins 2 colonnes.")
else:
    st.info("👆 Sélectionnez 2 ou 3 colonnes pour afficher la hiérarchie.")

st.markdown("---")

# ── Radar chart RAG ───────────────────────────────────────────────────────────
st.markdown("### 🎯 Radar — Top colonnes candidates RAG")
try:
    from utils.profiler import score_balance

    top8 = sorted(profiles.values(), key=lambda p: p.rag_score, reverse=True)[:8]
    if len(top8) >= 3:
        categories = ["Score RAG", "Complétude", "Équilibre", "Cardinalité norm.", "Type score"]
        fig_radar = go.Figure()

        for p in top8:
            completude = max(0, 100 - p.pct_missing)
            if p.col_type in ("categorical", "binary") and p.name in df_page.columns:
                try:
                    equil = score_balance(df_page[p.name]) * 100
                except Exception:
                    equil = 50.0
            else:
                equil = 50.0

            card_norm = max(0, 100 - min(p.n_unique, 100))
            type_score_map = {"categorical": 80, "binary": 60, "numeric": 30, "date": 30, "id": 5, "text_free": 5}
            type_s = type_score_map.get(p.col_type, 20)

            vals = [p.rag_score, completude, equil, card_norm, type_s]
            vals.append(vals[0])  # fermer le polygone

            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories + [categories[0]],
                fill="toself",
                name=p.name,
                opacity=0.65,
            ))

        fig_radar.update_layout(
            polar=dict(
                bgcolor="#161b22",
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#8b949e", size=9),
                                gridcolor="#21262d", linecolor="#21262d"),
                angularaxis=dict(tickfont=dict(color="#c9d1d9"), gridcolor="#21262d", linecolor="#21262d"),
            ),
            height=450, margin=dict(l=60, r=60, t=40, b=40),
            legend=dict(font=dict(color="#c9d1d9")),
            paper_bgcolor="#0d1117", font=dict(color="#c9d1d9"),
        )
        st.plotly_chart(fig_radar, use_container_width=True,
                        config={"toImageButtonOptions": {"format": "png", "filename": "radar_rag"}})
    else:
        st.info("Pas assez de colonnes pour le radar (min 3).")
except Exception as e:
    st.error(f"Erreur radar : {e}")

st.markdown("---")

# ── Recommandations automatiques ─────────────────────────────────────────────
st.markdown("### 💡 Recommandations automatiques")
try:
    from utils.rag_scorer import estimate_pages
    from utils.profiler import score_balance

    lignes_par_page = float(st.session_state.get("lignes_par_page", 3))
    n_rows_page = len(df_page)
    alerts_pages = []

    reco_data = []
    for col, p in sorted(profiles.items(), key=lambda x: x[1].rag_score, reverse=True):
        pros, cons, conseil = [], [], ""

        if p.rag_score >= 72:
            pros.append("Score RAG excellent")
        elif p.rag_score >= 48:
            pros.append("Bon score RAG")

        if p.col_type == "categorical" and 2 <= p.n_unique <= 30:
            pros.append(f"{p.n_unique} catégories — idéal")
        if p.pct_missing <= 5:
            pros.append("Très peu de manquants")
        if p.semantic_type != "generic":
            pros.append(f"Type sémantique détecté : {p.semantic_type}")

        if p.pct_missing > 20:
            cons.append(f"{p.pct_missing}% manquants")
        if p.col_type == "id":
            cons.append("Trop unique — chaque ligne = 1 segment")
        if p.col_type == "text_free":
            cons.append("Texte libre — non segmentable")
        if p.n_unique > 100:
            cons.append(f"{p.n_unique} valeurs → segmentation trop fine")
        if p.n_unique == 1:
            cons.append("1 seule valeur — aucune segmentation possible")

        if p.rag_score >= 72:
            conseil = "✅ Utiliser comme colonne principale de segmentation"
        elif p.rag_score >= 48:
            conseil = "🔵 Bon en colonne secondaire ou croisée"
        elif p.col_type == "numeric" and p.pct_missing < 30:
            conseil = "💡 Discrétiser par tranches (ex : quartiles)"
        elif p.col_type == "date":
            conseil = "💡 Segmenter par mois / trimestre / année"
        elif p.pct_missing > 50:
            conseil = "⚠️ Compléter les données avant d'utiliser"
        else:
            conseil = "🔴 Éviter pour la segmentation RAG"

        # Pages moyennes estimées pour cette colonne seule
        pages_moy_str = "—"
        if p.n_unique > 0 and n_rows_page > 0:
            try:
                taille_moy_seg = n_rows_page / p.n_unique
                nb_pages_moy, depasse = estimate_pages(int(taille_moy_seg), lignes_par_page)
                pages_moy_str = f"{nb_pages_moy}"
                if depasse:
                    pages_moy_str += " ⚠️"
                    alerts_pages.append((col, nb_pages_moy))
            except Exception:
                pass

        reco_data.append({
            "Colonne":   col,
            "Score":     p.rag_score,
            "Pages moy.": pages_moy_str,
            "✅ Pour":   " · ".join(pros) if pros else "—",
            "⚠️ Contre": " · ".join(cons) if cons else "—",
            "💡 Conseil": conseil,
        })

    # Alertes pages > 300
    for col_alert, nb_p in alerts_pages[:3]:
        st.warning(
            f"⚠️ **{col_alert}** seule génère des PDFs de **~{nb_p} pages** en moyenne "
            f"(> 300 pages — limite dépassée avec {lignes_par_page:.0f} lignes/page)"
        )

    reco_df = pd.DataFrame(reco_data)
    st.dataframe(reco_df, use_container_width=True, hide_index=True, height=420)
    st.caption(f"💡 Pages moy. calculées avec {lignes_par_page:.0f} lignes/page (modifiable page Simulateur RAG). ⚠️ = dépasse 300 pages.")
except Exception as e:
    st.error(f"Erreur recommandations : {e}")

