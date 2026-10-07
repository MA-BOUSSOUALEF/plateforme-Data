"""
pages/7_Explorateur.py
Explorateur de Données — vue tableau paginée interactive,
fiche détail d'une ligne, comparaison 2 lignes côte à côte,
stats contextuelles, export CSV/Excel de la sélection.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
from utils.profiler import safe_df_for_display

st.set_page_config(page_title="Explorateur", page_icon="🔍", layout="wide")

from utils.guards import check_data_loaded
from utils.profiler import DARK_LAYOUT as DARK

df_full, df, profiles = check_data_loaded()

st.markdown("# 🔍 Explorateur de Données")
st.markdown("Naviguez ligne par ligne, comparez des enregistrements, exportez votre sélection.")

with st.expander("💡 Comment utiliser cette page", expanded=False):
    st.markdown("""
- **Recherche globale** : filtre sur toutes les colonnes simultanément.
- **Filtres inline** : filtrez colonne par colonne dans la barre de contrôle.
- **Fiche ligne** : cliquez sur un numéro de ligne pour voir tous ses champs.
- **Comparaison** : sélectionnez 2 lignes pour les afficher côte à côte.
- **Export** : téléchargez le tableau filtré en CSV ou Excel.
    """)

# ── Filtres avancés ───────────────────────────────────────────────────────────
from utils.guards import apply_page_filters

df_page = apply_page_filters(df, profiles, "page7")

st.markdown("---")

# ── Recherche globale ─────────────────────────────────────────────────────────
search_query = st.text_input(
    "🔍 Recherche globale (filtre sur toutes les colonnes)",
    value="", placeholder="Tapez un terme pour filtrer toutes les colonnes…",
    key="expl_global_search",
)

if search_query.strip():
    try:
        mask = pd.Series([False] * len(df_page), index=df_page.index)
        for col in df_page.columns:
            mask |= df_page[col].astype(str).str.contains(search_query, case=False, na=False)
        df_page = df_page[mask]
        st.caption(f"🔍 Recherche « {search_query} » : {len(df_page):,} lignes")
    except Exception:
        pass

if len(df_page) == 0:
    st.warning("Aucun résultat pour cette recherche.")
    st.stop()

# ── Sélection colonnes à afficher ─────────────────────────────────────────────
all_cols = list(df_page.columns)
with st.expander("👁️ Colonnes affichées", expanded=False):
    shown_cols = st.multiselect(
        "Sélectionner les colonnes à afficher",
        options=all_cols,
        default=all_cols[:min(15, len(all_cols))],
        key="expl_shown_cols",
    )
    if not shown_cols:
        shown_cols = all_cols

df_display = df_page[shown_cols].copy()

# ── Pagination ────────────────────────────────────────────────────────────────
ROWS_PER_PAGE = 100
n_total = len(df_display)
n_pages = max(1, (n_total - 1) // ROWS_PER_PAGE + 1)

pc1, pc2, pc3 = st.columns([1, 2, 1])
with pc1:
    st.metric("Lignes visibles", f"{n_total:,}")
with pc2:
    page_num = st.number_input(
        f"Page (1 – {n_pages})", min_value=1, max_value=n_pages,
        value=1, step=1, key="expl_page",
    )
with pc3:
    sort_col = st.selectbox("Trier par", ["(index)"] + shown_cols, key="expl_sort_col")

# Tri
if sort_col != "(index)":
    sort_asc = st.checkbox("Croissant", value=True, key="expl_sort_asc")
    try:
        df_display = df_display.sort_values(sort_col, ascending=sort_asc)
    except Exception:
        pass

start = (page_num - 1) * ROWS_PER_PAGE
end   = min(start + ROWS_PER_PAGE, n_total)
df_slice = df_display.iloc[start:end].reset_index(drop=False)
df_slice = df_slice.rename(columns={"index": "⚓ idx"})

st.caption(f"Lignes {start+1} – {end} sur {n_total:,} · Page {page_num}/{n_pages}")
st.dataframe(safe_df_for_display(df_slice), use_container_width=True, height=450)

st.markdown("---")

# ── Vue fiche d'une ligne ─────────────────────────────────────────────────────
st.markdown("### 📋 Fiche d'une ligne")

fiche_tab1, fiche_tab2 = st.tabs(["🔍 Fiche simple", "🆚 Comparer 2 lignes"])

with fiche_tab1:
    if n_total > 0:
        _idx_list = df_page.index.tolist()
        row_idx = st.number_input(
            f"Ligne Excel ({_idx_list[0]} – {_idx_list[-1]})",
            min_value=_idx_list[0], max_value=_idx_list[-1], value=_idx_list[0], step=1,
            key="expl_row_idx",
        )
        try:
            row_data = df_page.loc[row_idx] if row_idx in df_page.index else df_page.iloc[0]
            fiche_cols = st.columns(3)
            for i, (col, val) in enumerate(row_data.items()):
                col_box = fiche_cols[i % 3]
                p = profiles.get(col)
                type_label = p.col_type if p else "?"
                col_box.markdown(
                    f"""<div style="background:#161b22;border:1px solid #21262d;border-radius:6px;
                    padding:0.5rem 0.8rem;margin-bottom:0.5rem">
                    <div style="font-size:0.7rem;color:#8b949e;text-transform:uppercase;
                    letter-spacing:0.08em">{col} <span style="color:#21262d">({type_label})</span></div>
                    <div style="font-size:0.95rem;color:#c9d1d9;font-weight:500;
                    word-break:break-word">{str(val) if pd.notna(val) else '—'}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.error(f"Erreur fiche : {e}")

with fiche_tab2:
    cc1, cc2 = st.columns(2)
    _idx_list2 = df_page.index.tolist()
    with cc1:
        row_a = st.number_input("Ligne Excel A", min_value=_idx_list2[0], max_value=_idx_list2[-1], value=_idx_list2[0], key="cmp_a")
    with cc2:
        row_b = st.number_input("Ligne Excel B", min_value=_idx_list2[0], max_value=_idx_list2[-1], value=_idx_list2[min(1, n_total-1)], key="cmp_b")

    try:
        ra = df_page.loc[row_a] if row_a in df_page.index else df_page.iloc[0]
        rb = df_page.loc[row_b] if row_b in df_page.index else df_page.iloc[0]

        cmp_cols = st.columns(2)
        with cmp_cols[0]:
            st.markdown(f"**Ligne Excel {row_a}**")
            for col in shown_cols[:20]:
                va = ra.get(col, "—")
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"padding:3px 6px;border-bottom:1px solid #21262d'>"
                    f"<span style='color:#8b949e;font-size:0.82rem'>{col}</span>"
                    f"<span style='color:#c9d1d9;font-size:0.82rem'>{str(va)[:60] if pd.notna(va) else '—'}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        with cmp_cols[1]:
            st.markdown(f"**Ligne Excel {row_b}**")
            for col in shown_cols[:20]:
                va = ra.get(col, "—")
                vb = rb.get(col, "—")
                diff = va != vb
                color = "#d29922" if diff else "#c9d1d9"
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"padding:3px 6px;border-bottom:1px solid #21262d'>"
                    f"<span style='color:#8b949e;font-size:0.82rem'>{col}</span>"
                    f"<span style='color:{color};font-size:0.82rem;font-weight:{'600' if diff else '400'}'>"
                    f"{str(vb)[:60] if pd.notna(vb) else '—'}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        st.caption("🟡 Valeurs surlignées = différentes entre les deux lignes")
    except Exception as e:
        st.error(f"Erreur comparaison : {e}")

st.markdown("---")

# ── Stats contextuelles ───────────────────────────────────────────────────────
st.markdown("### 📊 Statistiques du tableau filtré")

num_cols = [c for c, p in profiles.items() if p.col_type == "numeric" and c in df_page.columns]
if num_cols:
    try:
        sample_stat = df_page[num_cols].describe(percentiles=[0.25, 0.5, 0.75])
        sample_stat = sample_stat.round(3)
        st.dataframe(sample_stat, use_container_width=True, height=260)
    except Exception as e:
        st.error(f"Erreur stats : {e}")
else:
    st.info("Aucune colonne numérique pour les statistiques.")

st.markdown("---")

# ── Export ────────────────────────────────────────────────────────────────────
st.markdown("### 💾 Export")
exp_c1, exp_c2, exp_c3 = st.columns(3)

with exp_c1:
    csv_buf = io.StringIO()
    df_page[shown_cols].to_csv(csv_buf, index=False)
    st.download_button(
        "📥 Tableau filtré (CSV)",
        data=csv_buf.getvalue(),
        file_name="export_filtre.csv",
        mime="text/csv",
        use_container_width=True,
    )

with exp_c2:
    try:
        excel_buf = io.BytesIO()
        df_page[shown_cols].to_excel(excel_buf, index=False, engine="openpyxl")
        st.download_button(
            "📥 Tableau filtré (Excel)",
            data=excel_buf.getvalue(),
            file_name="export_filtre.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    except Exception as e:
        st.caption(f"Excel indisponible : {e}")

with exp_c3:
    # Export page courante uniquement
    csv_page_buf = io.StringIO()
    df_display.iloc[start:end].to_csv(csv_page_buf, index=False)
    st.download_button(
        f"📥 Page {page_num} uniquement (CSV)",
        data=csv_page_buf.getvalue(),
        file_name=f"page_{page_num}.csv",
        mime="text/csv",
        use_container_width=True,
    )
