"""
app.py — RAG Segmentation Dashboard
Point d'entrée principal : chargement fichier, sidebar filtres rapides,
badge filtres actifs, reset, mémorisation des sélections inter-pages.
"""
import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="RAG Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS global ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label {
    color: #8b949e !important;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1rem;
}

/* Section headers */
.section-title {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #8b949e;
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #21262d;
}

/* Badge */
.badge {
    display: inline-block;
    background: #f85149;
    color: #fff !important;
    border-radius: 10px;
    padding: 1px 8px;
    font-size: 0.72rem;
    font-weight: 700;
    margin-left: 6px;
}

/* RAG badge colors */
.badge-excellent { color: #3fb950; font-weight: 600; }
.badge-good      { color: #58a6ff; font-weight: 600; }
.badge-warning   { color: #d29922; font-weight: 600; }
.badge-bad       { color: #f85149; font-weight: 600; }

/* Score bar */
.score-bar-wrap { background: #21262d; border-radius: 4px; height: 6px; }
.score-bar-fill { height: 6px; border-radius: 4px; }

/* File info box */
.file-info {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 0.6rem 0.8rem;
    font-size: 0.82rem;
    color: #8b949e;
    margin-bottom: 0.5rem;
}
.file-info strong { color: #c9d1d9; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
_defaults = {
    "df": None,
    "profiles": None,
    "filename": None,
    "load_report": [],
    "load_datetime": None,
    "df_filtered": None,
    "active_filters": {},
    "filter_presets": {},
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 RAG Dashboard")
    st.markdown("---")

    # ── Upload ────────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Charger un fichier",
        type=["csv", "xlsx", "xls"],
        help="CSV ou Excel — jusqu'à 1M de lignes",
    )

    if uploaded is not None:
        # Détecter si c'est un nouveau fichier
        is_new = (st.session_state.filename != uploaded.name)

        if is_new or st.session_state.df is None:
            with st.spinner("⏳ Chargement et analyse en cours…"):
                from utils.loader import load_file, get_file_info
                from utils.profiler import profile_dataframe

                progress = st.progress(0, text="Lecture du fichier…")
                df, load_report = load_file(uploaded)
                progress.progress(50, text="Profilage des colonnes…")
                profiles = profile_dataframe(df)
                progress.progress(100, text="Terminé !")
                progress.empty()

                st.session_state.df = df
                st.session_state.profiles = profiles
                st.session_state.filename = uploaded.name
                st.session_state.load_report = load_report
                st.session_state.load_datetime = datetime.now().strftime("%d/%m %H:%M")
                st.session_state.df_filtered = df

                # Vider tous les filtres avancés des pages (valeurs stale du fichier précédent)
                for k in [k for k in st.session_state if "_filter__" in k]:
                    del st.session_state[k]

        # Infos fichier permanent
        info = get_file_info(st.session_state.df)
        st.markdown(
            f"""<div class="file-info">
            <strong>{st.session_state.filename}</strong><br>
            {info['n_rows']:,} lignes · {info['n_cols']} col · {info['memory_mb']} Mo<br>
            <span style="color:#3fb950">✅ Chargé le {st.session_state.load_datetime}</span>
            </div>""",
            unsafe_allow_html=True,
        )

        # Rapport chargement (collapsible)
        with st.expander("📋 Rapport de chargement", expanded=False):
            for line in st.session_state.load_report:
                st.markdown(f"- {line}")

        st.markdown("---")

    # ── Filtres rapides ───────────────────────────────────────────────────────
    if st.session_state.df is not None:
        df = st.session_state.df
        profiles = st.session_state.profiles

        # Compter filtres actifs
        active_filters = st.session_state.get("active_filters", {})
        n_active = len(active_filters)
        badge_html = f'<span class="badge">{n_active}</span>' if n_active > 0 else ""
        st.markdown(
            f'<div class="section-title">🎛️ Filtres rapides {badge_html}</div>',
            unsafe_allow_html=True,
        )

        # Top 4 colonnes par score RAG
        from utils.profiler import get_rag_candidates
        cat_cols = [
            col for col, p in profiles.items()
            if p.col_type in ("categorical", "binary") and p.n_unique <= 50
        ]
        # Trier par score RAG décroissant
        cat_cols_sorted = sorted(
            cat_cols,
            key=lambda c: profiles[c].rag_score,
            reverse=True,
        )

        active_filters = {}
        for col in cat_cols_sorted[:4]:
            try:
                values = sorted(df[col].dropna().astype(str).unique().tolist())
                selected = st.multiselect(
                    col, values, default=[],
                    key=f"sidebar_filter_{col}",
                    placeholder="Toutes…",
                )
                if selected:
                    active_filters[col] = selected
            except Exception:
                pass

        st.session_state.active_filters = active_filters

        # Bouton reset
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("🔄 Réinitialiser", use_container_width=True, type="secondary"):
                for col in cat_cols_sorted[:4]:
                    key = f"sidebar_filter_{col}"
                    if key in st.session_state:
                        del st.session_state[key]
                # Reset filtres avancés de toutes les pages
                for k in [k for k in st.session_state if "_filter__" in k]:
                    del st.session_state[k]
                st.session_state.active_filters = {}
                st.rerun()

        with col_r2:
            if st.button("🗑️ Nouveau fichier", use_container_width=True, type="secondary"):
                for k in list(_defaults.keys()):
                    st.session_state[k] = _defaults[k]
                for k in [k for k in st.session_state if "_filter__" in k]:
                    del st.session_state[k]
                st.rerun()

        # Appliquer les filtres sidebar
        if active_filters:
            df_f = df.copy()
            for col, vals in active_filters.items():
                df_f = df_f[df_f[col].astype(str).isin(vals)]
            st.session_state.df_filtered = df_f
            total_before = len(df)
            pct = round(len(df_f) / total_before * 100, 1) if total_before > 0 else 0
            color = "#3fb950" if pct > 50 else "#d29922" if pct > 10 else "#f85149"
            st.markdown(
                f'<div style="font-size:0.8rem;color:{color};text-align:center;padding:4px 0">'
                f'🔍 {len(df_f):,} / {total_before:,} lignes ({pct}%)</div>',
                unsafe_allow_html=True,
            )
        else:
            st.session_state.df_filtered = df

        # ── Presets de filtres sauvegardés ────────────────────────────────────
        presets = st.session_state.get("filter_presets", {})
        if presets:
            st.markdown("---")
            st.markdown('<div class="section-title">💾 Filtres sauvegardés</div>', unsafe_allow_html=True)
            preset_name = st.selectbox("Charger un preset", [""] + list(presets.keys()), label_visibility="collapsed")
            if preset_name:
                from utils.filters import FilterEngine
                fe = FilterEngine("page_filter")
                fe.load_preset(preset_name)
                st.success(f"Preset « {preset_name} » chargé")

    else:
        st.info("👆 Chargez un fichier pour commencer")
        st.session_state.active_filters = {}
        st.session_state.df_filtered = None

    st.markdown("---")
    st.caption("v2.0 · Plotly + Streamlit · Python 3.10+")


# ── Page d'accueil ────────────────────────────────────────────────────────────
if st.session_state.df is None:
    st.markdown("# 🎯 RAG Segmentation Dashboard")
    st.markdown("### Analysez, explorez et découpez vos données pour le RAG")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("#### 🏠 Vue Générale\nQualité, santé et score global")
    with c2:
        st.markdown("#### 🗂️ Structure\nTreemap, sunburst, radar RAG")
    with c3:
        st.markdown("#### 🔬 Groupby\nCroisements N colonnes")
    with c4:
        st.markdown("#### 🎯 Simulateur\nDécoupez et exportez")

    st.markdown("---")
    c5, c6, c7 = st.columns(3)
    with c5:
        st.markdown("#### 📈 Analyse Avancée\nCorrélations, distributions, anomalies")
    with c6:
        st.markdown("#### 🤖 Assistant RAG\nRapport auto + matrice décision")
    with c7:
        st.markdown("#### 🔍 Explorateur\nNavigation ligne par ligne")

    st.markdown("---")
    c8, c9, _ = st.columns(3)
    with c8:
        st.markdown("#### 🤖 Consultation LLM\nRecommandations GPT / Gemini / Claude")
    with c9:
        st.markdown("#### 📦 Chunks JSON\nGénérez les fichiers JSON par segment")

    st.markdown("---")
    st.markdown("**👈 Commencez par charger votre fichier dans la barre latérale**")

else:
    df = st.session_state.df_filtered
    profiles = st.session_state.profiles

    st.markdown("# 🎯 RAG Segmentation Dashboard")

    # Résumé rapide
    from utils.profiler import get_rag_candidates
    from utils.exporter import _compute_quality_score

    candidates = get_rag_candidates(profiles, min_score=45)
    quality = _compute_quality_score(df, profiles)
    color_q = "#3fb950" if quality >= 70 else "#d29922" if quality >= 40 else "#f85149"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📋 Lignes", f"{len(df):,}")
    c2.metric("📊 Colonnes", len(df.columns))
    c3.metric("🎯 Candidats RAG", len(candidates))
    c4.metric("❓ Manquants moy.", f"{round(sum(p.pct_missing for p in profiles.values())/max(len(profiles),1),1)}%")
    c5.metric("⭐ Score qualité", f"{quality}/100")

    if candidates:
        st.success(f"✅ Top colonnes RAG : `{'` · `'.join(candidates[:5])}`")
    else:
        st.warning("⚠️ Aucune colonne candidate évidente — explorez la page Structure.")

    st.info("👈 Naviguez entre les pages via le menu en haut à gauche")

    if candidates:
        c_llm, c_chunks = st.columns(2)
        with c_llm:
            st.markdown("**🤖 Consultation LLM** — recommandations GPT / Gemini / Claude")
        with c_chunks:
            st.markdown("**📦 Chunks JSON** — générez vos fichiers JSON par segment")
