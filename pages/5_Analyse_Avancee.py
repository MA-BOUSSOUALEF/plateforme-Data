"""
pages/5_Analyse_Avancee.py
Analyse Avancée — 4 onglets :
  1. Corrélations (Pearson/Spearman + Cramér's V + scatter interactif)
  2. Distributions (histogramme+KDE, boxplot, violin, percentiles, comparaison groupes)
  3. Anomalies (IQR / Z-score, tableau lignes anormales, valeurs rares catégorielles)
  4. Analyse Temporelle (line chart, gaps, heatmap calendrier)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Analyse Avancée", page_icon="📈", layout="wide")

from utils.guards import check_data_loaded
from utils.profiler import DARK_LAYOUT as DARK

_, df, profiles = check_data_loaded()

st.markdown("# 📈 Analyse Avancée")

with st.expander("💡 Comment utiliser cette page", expanded=False):
    st.markdown("""
- **Corrélations** : heatmap Pearson/Spearman entre numériques + Cramér's V entre catégorielles. Alertes corrélations > 0.9.
- **Distributions** : histogramme + boxplot + violin + percentiles pour chaque colonne numérique.
- **Anomalies** : détection IQR ou Z-score, tableau des lignes suspectes téléchargeable.
- **Temporelle** : évolution dans le temps, saisonnalité, gaps, heatmap calendrier.
    """)

# ── Filtres avancés ───────────────────────────────────────────────────────────
from utils.guards import apply_page_filters

df_page = apply_page_filters(df, profiles, "page5")

num_cols  = [c for c, p in profiles.items() if p.col_type == "numeric" and c in df_page.columns]
cat_cols  = [c for c, p in profiles.items() if p.col_type in ("categorical", "binary") and c in df_page.columns]
date_cols = [c for c, p in profiles.items() if p.col_type == "date" and c in df_page.columns]

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔗 Corrélations", "📊 Distributions", "⚠️ Anomalies", "📅 Temporelle"
])


# ════════════════════════════════════════════════════════════
# TAB 1 — CORRÉLATIONS
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🔗 Corrélations")

    # ── Heatmap numérique ─────────────────────────────────────────────────────
    if len(num_cols) >= 2:
        corr_method = st.radio("Méthode", ["Pearson", "Spearman"], horizontal=True, key="corr_method")
        try:
            sample = df_page[num_cols]
            if len(sample) > 50_000:
                sample = sample.sample(50_000, random_state=42)
                st.caption("📊 Corrélation sur échantillon 50 000 lignes")

            corr_mat = sample.corr(method=corr_method.lower())

            fig_corr = px.imshow(
                corr_mat,
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                text_auto=".2f",
                aspect="auto",
            )
            fig_corr.update_traces(textfont_size=10)
            fig_corr.update_layout(
                height=max(350, len(num_cols) * 40),
                margin=dict(l=10, r=10, t=40, b=60),
                title=f"Heatmap corrélation ({corr_method})",
                paper_bgcolor="#0d1117", font=dict(color="#c9d1d9"),
            )
            st.plotly_chart(fig_corr, use_container_width=True,
                            config={"toImageButtonOptions": {"format": "png", "filename": "correlation_heatmap"}})

            # Alertes corrélations fortes
            high_corr = []
            for i in range(len(corr_mat.columns)):
                for j in range(i+1, len(corr_mat.columns)):
                    v = abs(corr_mat.iloc[i, j])
                    if v >= 0.9:
                        high_corr.append((corr_mat.columns[i], corr_mat.columns[j], round(v, 3)))
            if high_corr:
                st.warning(f"⚠️ **{len(high_corr)} paires fortement corrélées (≥ 0.9)** — possibles colonnes redondantes :")
                for c1, c2, v in high_corr[:10]:
                    st.markdown(f"  - `{c1}` ↔ `{c2}` : {v}")
        except Exception as e:
            st.error(f"Erreur heatmap corrélation : {e}")
    else:
        st.info("⚠️ Moins de 2 colonnes numériques — heatmap indisponible.")

    st.markdown("---")

    # ── Cramér's V pour catégorielles ─────────────────────────────────────────
    if len(cat_cols) >= 2:
        st.markdown("#### 🔗 Cramér's V — corrélations catégorielles")
        try:
            from utils.profiler import compute_cramers_v
            show_cats = cat_cols[:10]  # Max 10x10

            with st.spinner("Calcul Cramér's V…"):
                cv_data = np.zeros((len(show_cats), len(show_cats)))
                for i, c1 in enumerate(show_cats):
                    for j, c2 in enumerate(show_cats):
                        if i == j:
                            cv_data[i, j] = 1.0
                        elif j > i:
                            v = compute_cramers_v(df_page[c1], df_page[c2])
                            cv_data[i, j] = v
                            cv_data[j, i] = v

            cv_df = pd.DataFrame(cv_data, index=show_cats, columns=show_cats)
            fig_cv = px.imshow(
                cv_df, color_continuous_scale="Blues",
                zmin=0, zmax=1, text_auto=".2f", aspect="auto",
            )
            fig_cv.update_traces(textfont_size=10)
            fig_cv.update_layout(
                height=max(320, len(show_cats) * 38),
                margin=dict(l=10, r=10, t=30, b=60),
                paper_bgcolor="#0d1117", font=dict(color="#c9d1d9"),
            )
            st.plotly_chart(fig_cv, use_container_width=True,
                            config={"toImageButtonOptions": {"format": "png", "filename": "cramers_v"}})
            if len(cat_cols) > 10:
                st.caption(f"⚠️ Limité aux 10 premières colonnes catégorielles sur {len(cat_cols)}.")
        except Exception as e:
            st.error(f"Erreur Cramér's V : {e}")



# ════════════════════════════════════════════════════════════
# TAB 2 — DISTRIBUTIONS
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Distributions numériques")

    if not num_cols:
        st.info("Aucune colonne numérique détectée.")
    else:
        dist_col = st.selectbox("Colonne à analyser", num_cols, key="dist_col")
        compare_by = st.selectbox(
            "Comparer par groupe (optionnel)",
            ["(aucune)"] + cat_cols,
            key="dist_group",
        )
        plot_type = st.multiselect(
            "Graphiques", ["Histogramme + KDE", "Boxplot"],
            default=["Histogramme + KDE", "Boxplot"], key="dist_plots",
        )

        try:
            s = df_page[dist_col].dropna().astype(float)
            if len(s) > 50_000:
                s = s.sample(50_000, random_state=42)
                st.caption("📊 Distribution sur échantillon 50 000 lignes")

            if not s.empty:
                # Percentiles
                pcts = np.percentile(s, [5, 25, 50, 75, 95])
                pc1, pc2, pc3, pc4, pc5 = st.columns(5)
                pc1.metric("P5",  f"{pcts[0]:,.2f}")
                pc2.metric("P25", f"{pcts[1]:,.2f}")
                pc3.metric("P50 (médiane)", f"{pcts[2]:,.2f}")
                pc4.metric("P75", f"{pcts[3]:,.2f}")
                pc5.metric("P95", f"{pcts[4]:,.2f}")

                if "Histogramme + KDE" in plot_type:
                    if compare_by != "(aucune)":
                        fig_h = px.histogram(
                            df_page[[dist_col, compare_by]].dropna(),
                            x=dist_col, color=compare_by, nbins=50,
                            barmode="overlay", opacity=0.7,
                            color_discrete_sequence=px.colors.qualitative.Set2,
                        )
                    else:
                        fig_h = px.histogram(s, nbins=50, color_discrete_sequence=["#58a6ff"])
                        try:
                            from scipy.stats import gaussian_kde
                            kde_x = np.linspace(s.min(), s.max(), 300)
                            kde = gaussian_kde(s)(kde_x)
                            kde_scaled = kde / kde.max() * s.value_counts(bins=50).max()
                            fig_h.add_trace(go.Scatter(
                                x=kde_x, y=kde_scaled, mode="lines",
                                line=dict(color="#f85149", width=2), name="KDE",
                            ))
                        except Exception:
                            pass
                    fig_h.update_layout(height=320, bargap=0.05,
                                        margin=dict(l=10, r=10, t=30, b=20), **DARK)
                    st.plotly_chart(fig_h, use_container_width=True,
                                    config={"toImageButtonOptions": {"format": "png", "filename": f"hist_{dist_col}"}})

                if "Boxplot" in plot_type:
                    if compare_by != "(aucune)":
                        fig_box = px.box(
                            df_page[[dist_col, compare_by]].dropna(),
                            x=compare_by, y=dist_col,
                            color=compare_by,
                            color_discrete_sequence=px.colors.qualitative.Set2,
                            points="outliers",
                        )
                    else:
                        fig_box = px.box(
                            pd.DataFrame({dist_col: s}), y=dist_col,
                            color_discrete_sequence=["#58a6ff"], points="outliers",
                        )
                    fig_box.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=20), **DARK)
                    st.plotly_chart(fig_box, use_container_width=True,
                                    config={"toImageButtonOptions": {"format": "png", "filename": f"box_{dist_col}"}})

        except Exception as e:
            st.error(f"Erreur distribution : {e}")


# ════════════════════════════════════════════════════════════
# TAB 3 — ANOMALIES
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### ⚠️ Détection d'anomalies")

    if not num_cols:
        st.info("Aucune colonne numérique.")
    else:
        ac1, ac2 = st.columns(2)
        with ac1:
            anom_method = st.radio("Méthode", ["IQR (1.5×)", "Z-score (>3σ)"], horizontal=True, key="anom_method")
        with ac2:
            anom_cols = st.multiselect("Colonnes analysées", num_cols, default=num_cols[:3], key="anom_cols")

        if anom_cols:
            try:
                from utils.profiler import detect_outliers

                method_key = "iqr" if "IQR" in anom_method else "zscore"
                all_outlier_idx = set()
                outlier_summary = []

                for col in anom_cols:
                    idx = detect_outliers(df_page[col], method=method_key)
                    n_out = len(idx)
                    pct_out = round(n_out / len(df_page) * 100, 2) if len(df_page) > 0 else 0
                    all_outlier_idx.update(idx.tolist())
                    outlier_summary.append({"Colonne": col, "Outliers": n_out, "% du total": pct_out})

                # Résumé
                sum_df = pd.DataFrame(outlier_summary)
                st.dataframe(sum_df, use_container_width=True, hide_index=True)

                n_total_out = len(all_outlier_idx)
                if n_total_out > 0:
                    st.warning(f"⚠️ **{n_total_out:,} lignes anormales** détectées (sur au moins une colonne)")
                    outlier_df = df_page.loc[list(all_outlier_idx)].head(500)

                    with st.expander(f"🔍 Voir les {min(n_total_out, 500)} lignes anormales"):
                        from utils.profiler import safe_df_for_display
                        st.dataframe(safe_df_for_display(outlier_df), use_container_width=True, height=300)
                        buf = outlier_df.to_csv(index=False).encode("utf-8")
                        st.download_button("📥 Télécharger les anomalies CSV", buf,
                                           file_name="anomalies.csv", mime="text/csv")
                else:
                    st.success("✅ Aucune anomalie numérique détectée.")
            except Exception as e:
                st.error(f"Erreur détection anomalies : {e}")

    st.markdown("---")

    # ── Valeurs rares catégorielles ────────────────────────────────────────────
    st.markdown("#### 🔍 Valeurs rares catégorielles (< 0.5%)")
    if cat_cols:
        try:
            rare_data = []
            for col in cat_cols[:15]:
                vc = df_page[col].value_counts(normalize=True, dropna=False)
                rares = vc[vc < 0.005]
                if len(rares) > 0:
                    rare_data.append({
                        "Colonne": col,
                        "Nb valeurs rares": len(rares),
                        "Exemples": ", ".join([str(v)[:30] for v in rares.index[:5]]),
                        "% cumulé": f"{rares.sum()*100:.2f}%",
                    })
            if rare_data:
                st.dataframe(pd.DataFrame(rare_data), use_container_width=True, hide_index=True)
                st.caption("💡 Ces valeurs peuvent représenter des erreurs de saisie ou des catégories à fusionner.")
            else:
                st.success("✅ Aucune valeur rare (<0.5%) détectée.")
        except Exception as e:
            st.error(f"Erreur valeurs rares : {e}")
    else:
        st.info("Aucune colonne catégorielle.")


# ════════════════════════════════════════════════════════════
# TAB 4 — ANALYSE TEMPORELLE
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📅 Analyse Temporelle")

    if not date_cols:
        st.info("Aucune colonne de type date détectée dans ce fichier.")
    else:
        tc1, tc2 = st.columns(2)
        with tc1:
            date_col = st.selectbox("Colonne date", date_cols, key="temp_date_col")
        with tc2:
            gran = st.selectbox("Granularité", ["Jour", "Semaine", "Mois", "Trimestre", "Année"],
                                index=2, key="temp_gran")

        gran_map = {"Jour": "D", "Semaine": "W", "Mois": "ME", "Trimestre": "QE", "Année": "YE"}

        try:
            s_dates = pd.to_datetime(df_page[date_col], errors="coerce").dropna()
            if len(s_dates) == 0:
                st.warning("Aucune date valide.")
            else:
                # Line chart évolution
                ts = s_dates.dt.to_period(gran_map[gran]).value_counts().sort_index()
                ts_df = ts.reset_index()
                ts_df.columns = ["Période", "Nb lignes"]
                ts_df["Période_str"] = ts_df["Période"].astype(str)

                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(
                    x=ts_df["Période_str"], y=ts_df["Nb lignes"],
                    mode="lines+markers", name="Nb lignes",
                    line=dict(color="#58a6ff", width=2),
                    fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
                ))
                fig_ts.update_layout(
                    height=320, margin=dict(l=10, r=10, t=30, b=60),
                    xaxis_tickangle=-45,
                    xaxis_title=gran, yaxis_title="Nb lignes",
                    **DARK,
                )
                st.plotly_chart(fig_ts, use_container_width=True,
                                config={"toImageButtonOptions": {"format": "png", "filename": "temporal_line"}})

                # Détection gaps
                mean_c = ts_df["Nb lignes"].mean()
                std_c  = ts_df["Nb lignes"].std()
                threshold = max(1, mean_c - 2 * std_c)
                gaps = ts_df[ts_df["Nb lignes"] < threshold]
                if len(gaps) > 0:
                    st.warning(
                        f"⚠️ **{len(gaps)} période(s) avec données très faibles** (< {threshold:.0f} lignes) : "
                        + ", ".join(gaps["Période_str"].tolist()[:6])
                    )

                # Heatmap calendrier (Jour ou Mois uniquement)
                if gran in ("Jour", "Mois"):
                    st.markdown("#### 📅 Heatmap calendrier")
                    try:
                        s_full = pd.to_datetime(df_page[date_col], errors="coerce").dropna()
                        cal_df = s_full.dt.to_frame()
                        cal_df.columns = ["date"]
                        cal_df["year"] = cal_df["date"].dt.year
                        cal_df["month"] = cal_df["date"].dt.month
                        cal_df["week_of_year"] = cal_df["date"].dt.isocalendar().week.astype(int)
                        cal_df["day_of_week"] = cal_df["date"].dt.dayofweek

                        if gran == "Jour":
                            pivot_cal = cal_df.groupby(["week_of_year", "day_of_week"]).size().unstack(fill_value=0)
                            day_names = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
                            pivot_cal.columns = [day_names[i] for i in pivot_cal.columns if i < len(day_names)]
                            fig_cal = px.imshow(pivot_cal.T, color_continuous_scale="Blues", aspect="auto")
                            fig_cal.update_layout(
                                height=220, margin=dict(l=10, r=10, t=20, b=20),
                                xaxis_title="Semaine de l'année",
                                paper_bgcolor="#0d1117", font=dict(color="#c9d1d9"),
                            )
                            st.plotly_chart(fig_cal, use_container_width=True,
                                            config={"toImageButtonOptions": {"format": "png", "filename": "heatmap_calendrier"}})
                        else:
                            pivot_cal = cal_df.groupby(["year", "month"]).size().unstack(fill_value=0)
                            months = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
                                      "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
                            pivot_cal.columns = [months[i-1] for i in pivot_cal.columns if 1 <= i <= 12]
                            fig_cal = px.imshow(pivot_cal, color_continuous_scale="Blues", aspect="auto",
                                                text_auto=True)
                            fig_cal.update_layout(
                                height=max(200, len(pivot_cal) * 40),
                                margin=dict(l=10, r=10, t=20, b=30),
                                paper_bgcolor="#0d1117", font=dict(color="#c9d1d9"),
                            )
                            st.plotly_chart(fig_cal, use_container_width=True,
                                            config={"toImageButtonOptions": {"format": "png", "filename": "heatmap_cal_mois"}})
                    except Exception as e:
                        st.error(f"Erreur heatmap calendrier : {e}")

        except Exception as e:
            st.error(f"Erreur analyse temporelle : {e}")
