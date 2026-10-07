"""
pages/1_Vue_Generale.py
Vue Générale — qualité, santé et structure des données.
Inclut : score qualité global (jauge), heatmap NaN, doublons,
frise chronologique si colonne date, export rapport HTML.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Vue Générale", page_icon="🏠", layout="wide")

from utils.guards import check_data_loaded
from utils.profiler import DARK_LAYOUT as DARK

df_full, df, profiles = check_data_loaded()

# ── En-tête ───────────────────────────────────────────────────────────────────
st.markdown("# 🏠 Vue Générale")
st.markdown("Qualité, structure et santé de vos données")

with st.expander("💡 Comment utiliser cette page", expanded=False):
    st.markdown("""
- **Score qualité** : jauge globale calculée sur complétude, doublons et cohérence des types (0 = catastrophique, 100 = parfait).
- **Heatmap NaN** : chaque colonne sur les tranches de 5% du fichier — repérez si les manquants sont concentrés ou éparpillés.
- **Filtres avancés** : ci-dessous pour affiner l'analyse sur un sous-ensemble.
- **Export HTML** : rapport téléchargeable autonome (s'ouvre dans le navigateur).
    """)

# ── Filtres avancés ───────────────────────────────────────────────────────────
from utils.guards import apply_page_filters
from utils.filters import FilterEngine

df_page = apply_page_filters(df, profiles, "page1")

with st.expander("💾 Sauvegarder filtre", expanded=False):
    _fe_save = FilterEngine("page1_filter")
    _preset_name = st.text_input("Nom du preset", key="p1_preset_name", placeholder="Mon filtre…")
    if st.button("💾 Sauvegarder", key="p1_save") and _preset_name:
        _fe_save.save_preset(_preset_name, profiles)
        st.success(f"Preset « {_preset_name} » sauvegardé !")

st.markdown("---")

# ── Score qualité global ──────────────────────────────────────────────────────
from utils.exporter import _compute_quality_score

quality_score = _compute_quality_score(df_page, profiles)
avg_missing   = round(sum(p.pct_missing for p in profiles.values()) / max(len(profiles), 1), 1)
n_excellent   = sum(1 for p in profiles.values() if p.rag_status == "excellent")
n_good        = sum(1 for p in profiles.values() if p.rag_status == "good")

try:
    n_dupes = df_page.duplicated().sum()
    dupe_pct = round(n_dupes / len(df_page) * 100, 1) if len(df_page) > 0 else 0
except Exception:
    n_dupes, dupe_pct = 0, 0.0

q_color = "#3fb950" if quality_score >= 70 else "#d29922" if quality_score >= 40 else "#f85149"
q_label = "Excellent" if quality_score >= 70 else "Acceptable" if quality_score >= 40 else "Insuffisant"

col_gauge, col_metrics = st.columns([1, 3])

with col_gauge:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=quality_score,
        title={"text": "Score Qualité Global", "font": {"color": "#c9d1d9", "size": 14}},
        number={"font": {"color": q_color, "size": 48}, "suffix": "/100"},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8b949e", "tickfont": {"color": "#8b949e"}},
            "bar": {"color": q_color, "thickness": 0.25},
            "bgcolor": "#21262d",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],  "color": "#2a1a1a"},
                {"range": [40, 70], "color": "#2a2a1a"},
                {"range": [70, 100],"color": "#1a3a1a"},
            ],
            "threshold": {"line": {"color": q_color, "width": 3}, "value": quality_score},
        },
    ))
    fig_gauge.update_layout(
        height=250, margin=dict(l=20, r=20, t=40, b=10),
        **DARK,
    )
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown(f"<div style='text-align:center;font-weight:700;color:{q_color}'>{q_label}</div>", unsafe_allow_html=True)

with col_metrics:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("📋 Lignes", f"{len(df_page):,}")
    m2.metric("📊 Colonnes", len(df_page.columns))
    m3.metric("❓ Manquants moy.", f"{avg_missing}%")
    m4.metric("🎯 Excellents RAG", n_excellent)
    m5.metric("🔁 Doublons", f"{n_dupes:,} ({dupe_pct}%)")

    # Doublons partiels sur colonne ID si présente
    id_cols = [c for c, p in profiles.items() if p.col_type == "id" and c in df_page.columns]
    if id_cols:
        try:
            col_id = id_cols[0]
            n_partial_dupes = df_page[col_id].duplicated().sum()
            if n_partial_dupes > 0:
                st.warning(f"⚠️ {n_partial_dupes:,} doublons partiels détectés sur `{col_id}`")
        except Exception:
            pass

st.markdown("---")

# ── Heatmap NaN ───────────────────────────────────────────────────────────────
st.markdown("### 🟥 Heatmap des valeurs manquantes")
st.caption("Chaque cellule = % de NaN dans une tranche de 5% du fichier. Repérez les concentrations ou les patterns aléatoires.")

try:
    N_BANDS = 20  # 20 tranches de 5%
    n = len(df_page)
    band_size = max(1, n // N_BANDS)
    cols_with_na = [c for c in df_page.columns if df_page[c].isna().any()]

    if not cols_with_na:
        st.success("✅ Aucune valeur manquante dans ce fichier !")
    else:
        show_cols = cols_with_na[:30]  # Max 30 colonnes
        heatmap_data = []
        y_labels = []
        for i in range(N_BANDS):
            start = i * band_size
            end = min(start + band_size, n)
            band = df_page.iloc[start:end]
            row = [round(band[c].isna().mean() * 100, 1) for c in show_cols]
            heatmap_data.append(row)
            y_labels.append(f"{round(start/n*100)}%")

        fig_hm = go.Figure(go.Heatmap(
            z=heatmap_data,
            x=show_cols,
            y=y_labels,
            colorscale=[[0, "#1a3a1a"], [0.01, "#21262d"], [0.5, "#d29922"], [1, "#f85149"]],
            zmin=0, zmax=100,
            colorbar=dict(title="% NaN", thickness=12, tickfont=dict(color="#c9d1d9")),
            hoverongaps=False,
            hovertemplate="Colonne: %{x}<br>Tranche: %{y}<br>NaN: %{z}%<extra></extra>",
        ))
        fig_hm.update_layout(
            height=max(300, N_BANDS * 18),
            margin=dict(l=60, r=20, t=20, b=80),
            xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
            **DARK,
        )
        st.plotly_chart(fig_hm, use_container_width=True)
        if len(cols_with_na) > 30:
            st.caption(f"⚠️ {len(cols_with_na) - 30} colonnes supplémentaires avec NaN non affichées (max 30).")
except Exception as e:
    st.error(f"Erreur heatmap NaN : {e}")

st.markdown("---")

# ── Barres manquants par colonne ──────────────────────────────────────────────
st.markdown("### 🔴 Valeurs manquantes par colonne")
try:
    missing_data = sorted(
        [(col, p.pct_missing) for col, p in profiles.items() if p.pct_missing > 0],
        key=lambda x: x[1], reverse=True,
    )
    if not missing_data:
        st.success("✅ Aucune valeur manquante !")
    else:
        missing_df = pd.DataFrame(missing_data, columns=["Colonne", "% Manquant"])
        fig_miss = go.Figure(go.Bar(
            x=missing_df["% Manquant"], y=missing_df["Colonne"], orientation="h",
            marker=dict(
                color=missing_df["% Manquant"],
                colorscale=[[0, "#3fb950"], [0.3, "#d29922"], [1.0, "#f85149"]],
                showscale=True,
                colorbar=dict(title="% manquant", thickness=12),
            ),
            text=missing_df["% Manquant"].apply(lambda x: f"{x}%"),
            textposition="outside",
        ))
        fig_miss.update_layout(
            height=max(280, len(missing_df) * 26),
            margin=dict(l=10, r=80, t=20, b=20),
            xaxis=dict(title="% de valeurs manquantes", range=[0, 115]),
            yaxis=dict(autorange="reversed"),
            **DARK,
        )
        st.plotly_chart(fig_miss, use_container_width=True)
except Exception as e:
    st.error(f"Erreur graphique manquants : {e}")

st.markdown("---")

# ── Frise chronologique ───────────────────────────────────────────────────────
date_cols = [c for c, p in profiles.items() if p.col_type == "date" and c in df_page.columns]
if date_cols:
    st.markdown("### 📅 Analyse temporelle")
    try:
        date_col = st.selectbox("Colonne date", date_cols, key="vg_date_col")
        granularity = st.radio("Granularité", ["Jour", "Semaine", "Mois", "Année"], index=2, horizontal=True, key="vg_gran")

        gran_map = {"Jour": "D", "Semaine": "W", "Mois": "ME", "Année": "YE"}
        s_dates = pd.to_datetime(df_page[date_col], errors="coerce").dropna()

        if len(s_dates) > 0:
            ts = s_dates.dt.to_period(gran_map[granularity]).value_counts().sort_index()
            ts_df = ts.reset_index()
            ts_df.columns = ["Période", "Nb lignes"]
            ts_df["Période"] = ts_df["Période"].astype(str)

            fig_ts = px.line(
                ts_df, x="Période", y="Nb lignes",
                markers=True, color_discrete_sequence=["#58a6ff"],
            )
            fig_ts.update_traces(fill="tozeroy", fillcolor="rgba(88,166,255,0.08)")
            fig_ts.update_layout(
                height=300, margin=dict(l=10, r=10, t=20, b=60),
                xaxis_tickangle=-45, **DARK,
            )
            st.plotly_chart(fig_ts, use_container_width=True)

            # Détection gaps
            if len(ts_df) > 1:
                counts = ts_df["Nb lignes"].values
                mean_c = counts.mean()
                gaps = [(ts_df["Période"].iloc[i], ts_df["Nb lignes"].iloc[i])
                        for i in range(len(ts_df))
                        if ts_df["Nb lignes"].iloc[i] < mean_c * 0.1]
                if gaps:
                    st.warning(f"⚠️ {len(gaps)} période(s) avec très peu de données : " +
                               ", ".join([f"`{p}` ({n} lignes)" for p, n in gaps[:5]]))
        else:
            st.info("Aucune date valide dans cette colonne.")
    except Exception as e:
        st.error(f"Erreur analyse temporelle : {e}")
    st.markdown("---")

# ── Tableau de profil des colonnes ────────────────────────────────────────────
st.markdown("### 📋 Profil complet des colonnes")
try:
    from utils.rag_scorer import estimate_pages
    from utils.profiler import score_balance

    lignes_par_page = float(st.session_state.get("lignes_par_page", 3))
    n_rows_page     = len(df_page)

    STATUS_EMOJI = {
        "excellent": "🟢 Excellent", "good": "🔵 Bon",
        "warning": "🟡 Attention", "bad": "🔴 Inutilisable",
    }
    table_data = []
    for col, p in sorted(profiles.items(), key=lambda x: x[1].rag_score, reverse=True):
        sem = f" ({p.semantic_type})" if p.semantic_type != "generic" else ""

        # Équilibre distribution
        equil_str = "—"
        try:
            if col in df_page.columns:
                bal = score_balance(df_page[col]) * 100
                equil_str = f"{bal:.0f}%"
        except Exception:
            pass

        # Pages moyennes estimées
        pages_moy_str = "—"
        try:
            if p.n_unique > 0 and n_rows_page > 0:
                taille_moy = n_rows_page / p.n_unique
                nb_pages, depasse = estimate_pages(int(taille_moy), lignes_par_page)
                pages_moy_str = f"{nb_pages}" + (" ⚠️" if depasse else "")
        except Exception:
            pass

        table_data.append({
            "Colonne":         col,
            "Type":            p.col_type + sem,
            "Valeurs uniques": p.n_unique,
            "% Manquant":      p.pct_missing,
            "Équilibre":       equil_str,
            "Pages moy.":      pages_moy_str,
            "Score RAG":       p.rag_score,
            "Statut":          STATUS_EMOJI.get(p.rag_status, ""),
            "Raison":          p.rag_reason,
        })
    table_df = pd.DataFrame(table_data)

    def color_score(val):
        if val >= 72:   return "background-color:#1a3a1a;color:#3fb950"
        elif val >= 48: return "background-color:#1a2a3a;color:#58a6ff"
        elif val >= 28: return "background-color:#2a2a1a;color:#d29922"
        else:           return "background-color:#2a1a1a;color:#f85149"

    styled = table_df.style.map(color_score, subset=["Score RAG"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=420)
    st.caption(
        f"💡 Score RAG ≥ 72 = excellent. "
        f"Équilibre = entropie Shannon normalisée. "
        f"Pages moy. avec {lignes_par_page:.0f} lignes/page (⚠️ = > 300 pages)."
    )
except Exception as e:
    st.error(f"Erreur tableau profil : {e}")

st.markdown("---")

# ── Export rapport HTML ───────────────────────────────────────────────────────
st.markdown("### 📄 Export")
try:
    from utils.exporter import generate_html_report
    filename = st.session_state.get("filename", "données")
    html_bytes = generate_html_report(df_page, profiles, filename)
    st.download_button(
        label="📄 Télécharger rapport qualité HTML",
        data=html_bytes,
        file_name=f"rapport_qualite_{filename.replace('.', '_')}.html",
        mime="text/html",
        use_container_width=False,
    )
except Exception as e:
    st.error(f"Erreur génération rapport : {e}")
