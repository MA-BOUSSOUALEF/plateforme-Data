"""
pages/3_Groupby.py
Groupby Explorer — N colonnes, multi-métriques, visualisations adaptatives.
Supporte 1 à 5 colonnes de regroupement + plusieurs métriques simultanées.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
from utils.profiler import safe_fillna, safe_df_for_display

st.set_page_config(page_title="Groupby Explorer", page_icon="🔬", layout="wide")

from utils.guards import check_data_loaded
from utils.profiler import DARK_LAYOUT as DARK

_, df, profiles = check_data_loaded()

st.markdown("# 🔬 Groupby Explorer")
st.markdown("Croisez N colonnes, calculez plusieurs métriques simultanément, explorez vos segments.")

with st.expander("💡 Comment utiliser cette page", expanded=False):
    st.markdown("""
- **Colonnes de regroupement** : sélectionnez 1 à 5 colonnes (l'ordre détermine la hiérarchie).
- **Multi-métriques** : cochez plusieurs agrégations à calculer en même temps.
- **Aperçu en temps réel** : le nombre de groupes est calculé automatiquement.
- **Visualisations adaptatives** : le bon graphique est choisi selon le nombre de colonnes.
- **Export CSV** : téléchargez le tableau des résultats.
    """)

# ── Filtres avancés ───────────────────────────────────────────────────────────
from utils.guards import apply_page_filters

df_page = apply_page_filters(df, profiles, "page3")

st.markdown("---")

# ── Sélection colonnes ────────────────────────────────────────────────────────
all_cols  = list(df_page.columns)
cat_cols  = [c for c, p in profiles.items() if p.col_type in ("categorical", "binary") and c in all_cols]
num_cols  = [c for c, p in profiles.items() if p.col_type == "numeric" and c in all_cols]

st.markdown("### ⚙️ Configuration du groupby")

cfg1, cfg2 = st.columns([2, 2])

with cfg1:
    st.markdown("**Colonnes de regroupement** (1 à 5)")
    group_cols = st.multiselect(
        "Colonnes de regroupement",
        options=all_cols,
        default=cat_cols[:1] if cat_cols else all_cols[:1],
        label_visibility="collapsed",
        placeholder="Sélectionnez 1 à 5 colonnes…",
    )

    # Réordonnement par flèches
    if len(group_cols) > 1:
        st.caption("Réordonner :")
        reorder_cols = st.columns(len(group_cols))
        for i, col in enumerate(group_cols):
            reorder_cols[i].caption(f"{i+1}. {col[:18]}…" if len(col) > 18 else f"{i+1}. {col}")

    # Aperçu nb groupes
    if group_cols:
        try:
            valid_gcols = [c for c in group_cols if c in df_page.columns]
            if valid_gcols:
                df_tmp = safe_fillna(df_page[valid_gcols])
                n_groups = df_tmp.groupby(valid_gcols).ngroups
                color = "#3fb950" if n_groups <= 50 else "#d29922" if n_groups <= 200 else "#f85149"
                st.markdown(
                    f'<div style="color:{color};font-size:0.9rem;font-weight:600">'
                    f'→ {n_groups:,} groupes distincts</div>',
                    unsafe_allow_html=True,
                )
        except Exception:
            pass

with cfg2:
    st.markdown("**Colonne valeur & métriques**")
    value_col_opts = ["(compter les lignes)"] + num_cols
    value_col = st.selectbox("Colonne à mesurer", value_col_opts, key="gb_val_col",
                             label_visibility="collapsed")

    if value_col == "(compter les lignes)":
        agg_options = ["count"]
        selected_aggs = ["count"]
    else:
        agg_options = ["count", "sum", "mean", "median", "min", "max", "std", "nunique"]
        selected_aggs = st.multiselect(
            "Métriques", agg_options,
            default=["count", "mean"],
            label_visibility="collapsed",
        )
        if not selected_aggs:
            selected_aggs = ["count"]

    top_n = st.slider("Top N résultats", 5, 200, 50, key="gb_topn")

st.markdown("---")

# ── Calcul groupby ────────────────────────────────────────────────────────────
if not group_cols:
    st.info("👆 Sélectionnez au moins une colonne de regroupement.")
    st.stop()

valid_gcols = [c for c in group_cols if c in df_page.columns]
if not valid_gcols:
    st.error("Colonnes sélectionnées absentes du DataFrame.")
    st.stop()

with st.spinner("Calcul en cours…"):
    try:
        needed = valid_gcols + ([value_col] if value_col != "(compter les lignes)" else [])
        df_work = safe_fillna(df_page[[c for c in needed if c in df_page.columns]].copy())

        if value_col == "(compter les lignes)":
            result = df_work.groupby(valid_gcols).size().reset_index(name="count")
            metric_cols = ["count"]
        else:
            agg_dict = {value_col: selected_aggs}
            result = df_work.groupby(valid_gcols)[value_col].agg(selected_aggs).reset_index()
            result.columns = valid_gcols + [f"{value_col}_{a}" for a in selected_aggs]
            metric_cols = [f"{value_col}_{a}" for a in selected_aggs]

        # Trier par première métrique
        sort_col = metric_cols[0]
        result = result.sort_values(sort_col, ascending=False).head(top_n).reset_index(drop=True)

    except Exception as e:
        st.error(f"Erreur de calcul groupby : {e}")
        st.stop()

# ── Titre dynamique ───────────────────────────────────────────────────────────
title = f"Résultat — `{'` × `'.join(valid_gcols)}`"
if value_col != "(compter les lignes)":
    title += f" · mesure `{value_col}`"
st.markdown(f"### {title}")
st.caption(f"Top {len(result)} résultats sur {n_groups:,} groupes total")

# ── Visualisations adaptatives ────────────────────────────────────────────────
PRIMARY_METRIC = metric_cols[0]

try:
    if len(valid_gcols) == 1:
        col_g = valid_gcols[0]
        viz_choice = st.radio(
            "Type de graphique",
            ["📊 Bar horizontal", "🌊 Sankey simplifié"],
            horizontal=True, key="gb_viz1",
        )

        if viz_choice == "📊 Bar horizontal":
            fig = go.Figure(go.Bar(
                x=result[PRIMARY_METRIC], y=result[col_g].astype(str), orientation="h",
                text=result[PRIMARY_METRIC].apply(lambda x: f"{x:,.1f}" if isinstance(x, float) else f"{x:,}"),
                textposition="outside",
                marker=dict(color=result[PRIMARY_METRIC], colorscale="Blues", showscale=False),
            ))
            fig.update_layout(
                height=max(350, len(result) * 26),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=10, r=100, t=20, b=20),
                xaxis_title=PRIMARY_METRIC, **DARK,
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"toImageButtonOptions": {"format": "png", "filename": "groupby_bar"}})

        elif viz_choice == "🌊 Sankey simplifié":
            try:
                cats = result[col_g].astype(str).tolist()
                all_nodes = ["Total"] + cats
                node_idx = {v: i for i, v in enumerate(all_nodes)}
                sources = [0] * len(cats)
                targets = list(range(1, len(cats) + 1))
                values_s = result[PRIMARY_METRIC].tolist()
                fig = go.Figure(go.Sankey(
                    node=dict(
                        label=all_nodes, pad=15, thickness=20,
                        color=["#58a6ff"] + ["#3fb950"] * len(cats),
                        line=dict(color="#21262d", width=0.5),
                    ),
                    link=dict(source=sources, target=targets, value=values_s,
                              color="rgba(88,166,255,0.25)"),
                ))
                fig.update_layout(height=500, paper_bgcolor="#0d1117", font=dict(color="#c9d1d9"))
                st.plotly_chart(fig, use_container_width=True,
                                config={"toImageButtonOptions": {"format": "png", "filename": "groupby_sankey1"}})
            except Exception as e:
                st.error(f"Erreur Sankey : {e}")

    elif len(valid_gcols) == 2:
        col_g1, col_g2 = valid_gcols
        n2 = result[col_g2].astype(str).nunique()
        viz_choice = st.radio(
            "Type de graphique",
            ["📊 Grouped bar", "📊 Stacked bar", "🔥 Heatmap pivot", "🌊 Sankey"],
            horizontal=True, key="gb_viz2",
        )

        if viz_choice in ("📊 Grouped bar", "📊 Stacked bar"):
            barmode = "group" if viz_choice == "📊 Grouped bar" else "stack"
            fig = px.bar(result.head(min(top_n, 200)), x=col_g1, y=PRIMARY_METRIC, color=col_g2,
                         barmode=barmode, color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=450, margin=dict(l=10, r=20, t=20, b=100),
                              xaxis_tickangle=-35, yaxis_title=PRIMARY_METRIC, **DARK)
            st.plotly_chart(fig, use_container_width=True,
                            config={"toImageButtonOptions": {"format": "png", "filename": "groupby_bar2"}})

        elif viz_choice == "🔥 Heatmap pivot":
            try:
                pivot = result.pivot_table(index=col_g1, columns=col_g2, values=PRIMARY_METRIC,
                                           aggfunc="first", fill_value=0)
                fig = px.imshow(pivot, color_continuous_scale="Blues", aspect="auto",
                                labels=dict(color=PRIMARY_METRIC))
                fig.update_layout(height=max(350, len(pivot) * 25),
                                  margin=dict(l=10, r=10, t=30, b=60),
                                  paper_bgcolor="#0d1117", font=dict(color="#c9d1d9"))
                st.plotly_chart(fig, use_container_width=True,
                                config={"toImageButtonOptions": {"format": "png", "filename": "groupby_heatmap"}})
            except Exception as e:
                st.error(f"Erreur pivot : {e}")

        elif viz_choice == "🌊 Sankey":
            try:
                vals_g1 = result[col_g1].astype(str).unique().tolist()
                vals_g2 = result[col_g2].astype(str).unique().tolist()
                all_nodes = vals_g1 + vals_g2
                node_idx = {v: i for i, v in enumerate(all_nodes)}

                sources, targets, values_s = [], [], []
                for _, row in result.iterrows():
                    sources.append(node_idx[str(row[col_g1])])
                    targets.append(node_idx[str(row[col_g2])])
                    values_s.append(float(row[PRIMARY_METRIC]))

                fig = go.Figure(go.Sankey(
                    node=dict(
                        label=all_nodes, pad=15, thickness=20,
                        color=["#58a6ff"] * len(vals_g1) + ["#3fb950"] * len(vals_g2),
                        line=dict(color="#21262d", width=0.5),
                    ),
                    link=dict(source=sources, target=targets, value=values_s,
                              color="rgba(88,166,255,0.25)"),
                ))
                fig.update_layout(height=500, paper_bgcolor="#0d1117", font=dict(color="#c9d1d9"))
                st.plotly_chart(fig, use_container_width=True,
                                config={"toImageButtonOptions": {"format": "png", "filename": "groupby_sankey"}})
            except Exception as e:
                st.error(f"Erreur Sankey : {e}")

    else:  # 3+ colonnes
        viz_choice = st.radio(
            "Type de graphique",
            ["🌟 Sunburst", "📂 Parallel Categories", "📋 Tableau hiérarchique"],
            horizontal=True, key="gb_viz3",
        )

        if viz_choice == "🌟 Sunburst":
            try:
                df_sun = result.copy()
                for c in valid_gcols:
                    df_sun[c] = df_sun[c].astype(str)
                fig = px.sunburst(df_sun, path=valid_gcols, values=PRIMARY_METRIC,
                                  color=PRIMARY_METRIC, color_continuous_scale="Blues")
                fig.update_layout(height=520, paper_bgcolor="#0d1117", font=dict(color="#c9d1d9"),
                                  coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True,
                                config={"toImageButtonOptions": {"format": "png", "filename": "groupby_sunburst"}})
            except Exception as e:
                st.error(f"Erreur Sunburst : {e}")

        elif viz_choice == "📂 Parallel Categories":
            try:
                df_parc = df_page[valid_gcols].head(10_000).copy()
                for c in valid_gcols:
                    df_parc[c] = df_parc[c].astype(str)
                fig = px.parallel_categories(df_parc, dimensions=valid_gcols,
                                             color_continuous_scale="Blues")
                fig.update_layout(height=500, paper_bgcolor="#0d1117", font=dict(color="#c9d1d9"))
                st.plotly_chart(fig, use_container_width=True,
                                config={"toImageButtonOptions": {"format": "png", "filename": "groupby_parcats"}})
            except Exception as e:
                st.error(f"Erreur Parallel Categories : {e}")

        else:
            st.caption("Tableau hiérarchique — triable par colonne")

except Exception as e:
    st.error(f"Erreur visualisation : {e}")

st.markdown("---")

# ── Tableau résultats + export ────────────────────────────────────────────────
col_table, col_insights = st.columns([3, 1])

with col_table:
    st.markdown("#### 📋 Tableau des résultats")
    display_df = result.copy()
    for c in valid_gcols:
        display_df[c] = display_df[c].astype(str)
    st.dataframe(safe_df_for_display(display_df), use_container_width=True, height=350)

    csv_buf = io.StringIO()
    result.to_csv(csv_buf, index=False)
    st.download_button(
        label="📥 Télécharger CSV",
        data=csv_buf.getvalue(),
        file_name=f"groupby_{'_'.join(valid_gcols)}.csv",
        mime="text/csv",
    )

with col_insights:
    st.markdown("#### 💡 Insights")
    try:
        if len(result) > 0:
            top1 = result.iloc[0]
            label_top1 = " / ".join([str(top1[c]) for c in valid_gcols])
            val_top1 = top1[PRIMARY_METRIC]
            val_fmt = f"{int(val_top1):,}" if float(val_top1) == int(val_top1) else f"{val_top1:,.2f}"
            st.info(f"🏆 **Top 1 :** `{label_top1}` → {val_fmt}")

        if len(result) >= 3:
            total = result[PRIMARY_METRIC].sum()
            top3_sum = result.head(3)[PRIMARY_METRIC].sum()
            pct_top3 = round(top3_sum / total * 100, 1) if total > 0 else 0
            st.info(f"📊 Top 3 = **{pct_top3}%** du total")

        if len(result) > 1:
            cv = result[PRIMARY_METRIC].std() / result[PRIMARY_METRIC].mean() * 100
            equil = "Équilibré" if cv < 50 else "Déséquilibré" if cv < 150 else "Très déséquilibré"
            color_e = "#3fb950" if cv < 50 else "#d29922" if cv < 150 else "#f85149"
            st.markdown(f'<div style="color:{color_e}">⚖️ {equil} (CV={cv:.0f}%)</div>', unsafe_allow_html=True)
    except Exception:
        pass
