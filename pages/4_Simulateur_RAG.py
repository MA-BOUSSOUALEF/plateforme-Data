"""
pages/4_Simulateur_RAG.py
Simulateur de découpe RAG — paramètre lignes/page, scan automatique enrichi,
sélecteur multi-colonnes (1-3), contrainte 300 pages, jauge score combinaison,
histogramme distribution, export ZIP + index.json.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
from utils.profiler import get_rag_candidates, simulate_rag_split, safe_fillna, safe_df_for_display
from utils.rag_scorer import estimate_pages, score_combination, scan_all_combinations

st.set_page_config(page_title="Simulateur RAG", page_icon="🎯", layout="wide")

from utils.guards import check_data_loaded
from utils.profiler import DARK_LAYOUT as DARK

_, df, profiles = check_data_loaded()

st.markdown("# 🎯 Simulateur de Découpe RAG")
st.markdown("Configurez, validez et exportez votre segmentation.")

with st.expander("💡 Comment utiliser cette page", expanded=False):
    st.markdown("""
- **Lignes/page** : calibrez la contrainte de 300 pages selon votre mise en page PDF réelle.
- **Scan automatique** : teste toutes les combinaisons candidates et classe les meilleures selon le score + contrainte pages.
- **Configuration** : choisissez 1, 2 ou 3 colonnes. Valeur par défaut = meilleure combinaison détectée.
- **Jauge** : score 0-100 de la combinaison simulée (équilibre, pages, taille segments).
- **Export ZIP** : un CSV par segment + index.json avec toutes les métadonnées.
    """)

# ── Filtres avancés ───────────────────────────────────────────────────────────
from utils.guards import apply_page_filters

df_page = apply_page_filters(df, profiles, "page4")

st.markdown("---")

candidates = get_rag_candidates(profiles, min_score=45)
if candidates:
    st.success(f"✅ **{len(candidates)} colonnes candidates** : `{'` · `'.join(candidates[:6])}`")
else:
    st.warning("⚠️ Aucune colonne candidate évidente. Sélectionnez manuellement ci-dessous.")

# ════════════════════════════════════════════════════════════
# SECTION 1 — Scan automatique des combinaisons
# ════════════════════════════════════════════════════════════
st.markdown("### 🔍 Meilleures combinaisons détectées automatiquement")

col_ratio, col_info = st.columns([1, 3])
with col_ratio:
    lignes_par_page = st.number_input(
        "📄 Lignes par page PDF",
        min_value=1, max_value=100,
        value=int(st.session_state.get("lignes_par_page", 3)),
        help="Définit combien de lignes Excel = 1 page PDF. "
             "Ajustez selon votre mise en page réelle.",
        key="lignes_par_page_input",
    )
with col_info:
    limite_lignes = int(300 * lignes_par_page)
    st.info(
        f"Avec **{lignes_par_page}** lignes/page, "
        f"la limite de 300 pages = **{limite_lignes:,} lignes** par segment."
    )
st.session_state["lignes_par_page"] = lignes_par_page

if st.button("🚀 Lancer le scan", key="scan_auto"):
    with st.spinner("Analyse de toutes les combinaisons…"):
        scan_res = scan_all_combinations(df_page, profiles, float(lignes_par_page))
    st.session_state["rag_scan_results"] = scan_res
    st.session_state["rag_scan_lpp"] = lignes_par_page  # pour détecter si lignes_par_page a changé

# Invalider le cache de résultats si lignes_par_page a changé
if st.session_state.get("rag_scan_lpp") != lignes_par_page:
    st.session_state.pop("rag_scan_results", None)

scan_res: list = st.session_state.get("rag_scan_results", [])

if scan_res:
    STATUS_EMOJI = {
        "excellent":     "✅ excellent",
        "bon":           "🔵 bon",
        "attention":     "⚠️ attention",
        "problématique": "❌ problématique",
    }

    scan_rows = []
    for i, r in enumerate(scan_res):
        scan_rows.append({
            "Rang":           i + 1,
            "Combinaison":    r["label"],
            "Nb PDFs":        r["nb_segments"],
            "Pages moy.":     r["pages_moyenne"],
            "Pages max":      r["pages_max"],
            "% hors limite":  r["pct_hors_limite"],
            "Score":          r["score"],
            "Statut":         STATUS_EMOJI.get(r["statut"], r["statut"]),
        })
    scan_df = pd.DataFrame(scan_rows)

    def _color_scan_score(val):
        if val >= 72:   return "background-color:#1a3a1a;color:#3fb950"
        elif val >= 48: return "background-color:#1a2a3a;color:#58a6ff"
        elif val >= 28: return "background-color:#2a2a1a;color:#d29922"
        else:           return "background-color:#2a1a1a;color:#f85149"

    styled_scan = scan_df.style.map(_color_scan_score, subset=["Score"])
    st.dataframe(styled_scan, use_container_width=True, hide_index=True,
                 height=min(420, (len(scan_res) + 1) * 35 + 55))

    combo_labels = [r["label"] for r in scan_res]
    sc1, sc2 = st.columns([3, 1])
    with sc1:
        selected_combo_label = st.selectbox(
            "▶ Sélectionner une combinaison à utiliser",
            options=combo_labels, key="scan_combo_select",
        )
    with sc2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✅ Appliquer", key="apply_combo", use_container_width=True):
            chosen = next((r for r in scan_res if r["label"] == selected_combo_label), None)
            if chosen:
                st.session_state["rag_selected_cols"]  = chosen["cols"]
                st.session_state["sim_cols_multi"]     = chosen["cols"]
                st.rerun()
elif not st.session_state.get("rag_scan_results"):
    st.caption("💡 Cliquez sur **Lancer le scan** pour analyser toutes les combinaisons de vos colonnes candidates.")

st.markdown("---")

# ════════════════════════════════════════════════════════════
# SECTION 2 — Configuration manuelle
# ════════════════════════════════════════════════════════════
st.markdown("### ⚙️ Configurer la découpe")

all_cols  = list(df_page.columns)
col_opts  = candidates if candidates else all_cols

# Déterminer les valeurs par défaut pour le multiselect
default_cols: list = st.session_state.get("rag_selected_cols", [])
if not default_cols and candidates:
    default_cols = candidates[:1]

valid_defaults = [c for c in default_cols if c in col_opts]

conf_c1, conf_c2 = st.columns([3, 1])
with conf_c1:
    split_cols = st.multiselect(
        "Colonnes de découpe (1 à 3)",
        options=col_opts,
        default=valid_defaults,
        max_selections=3,
        key="sim_cols_multi",
        help="Sélectionnez 1, 2 ou 3 colonnes pour la segmentation RAG.",
    )
with conf_c2:
    min_rows = st.number_input("Taille min. (lignes)", min_value=1, value=10, key="sim_minrows")

if not split_cols:
    st.info("Sélectionnez au moins 1 colonne de découpe ci-dessus.")
    st.stop()

# ── Simulation ────────────────────────────────────────────────────────────────
with st.spinner("Simulation…"):
    sim_df = simulate_rag_split(df_page, split_cols)

if len(sim_df) == 0:
    st.error("Simulation vide. Vérifiez les colonnes sélectionnées.")
    st.stop()

# ── Score de la combinaison actuelle (score_combination) ─────────────────────
try:
    combo_result = score_combination(df_page, split_cols, float(lignes_par_page))
except Exception:
    combo_result = {}

st.markdown("---")
st.markdown("### 📊 Résultat de la simulation")

n_segments  = len(sim_df)
n_too_small = int((sim_df["nb_lignes"] < min_rows).sum())
avg_size    = float(sim_df["nb_lignes"].mean())
min_size    = int(sim_df["nb_lignes"].min())
max_size    = int(sim_df["nb_lignes"].max())

pages_avg, _        = estimate_pages(int(avg_size), float(lignes_par_page))
pages_max_val, depasse_max = estimate_pages(max_size, float(lignes_par_page))

# Score et couleur pour la jauge
if combo_result:
    combo_score      = combo_result["score"]
    pct_hors         = combo_result["pct_hors_limite"]
    pct_trop_petits  = combo_result["pct_trop_petits"]
else:
    # Fallback si score_combination a échoué
    combo_score     = 0
    pct_hors        = 0.0
    pct_trop_petits = 0.0

if combo_score >= 72:
    rag_color = "#3fb950"
elif combo_score >= 48:
    rag_color = "#58a6ff"
elif combo_score >= 28:
    rag_color = "#d29922"
else:
    rag_color = "#f85149"

# ════════════════════════════════════════════════════════════
# SECTION 3 — Jauge + métriques
# ════════════════════════════════════════════════════════════
col_jg, col_met = st.columns([1, 3])

with col_jg:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=combo_score,
        title={"text": "Score combinaison", "font": {"color": "#c9d1d9", "size": 13}},
        number={"font": {"color": rag_color, "size": 44}, "suffix": "/100"},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "#8b949e"}},
            "bar":  {"color": rag_color, "thickness": 0.25},
            "bgcolor": "#21262d", "borderwidth": 0,
            "steps": [
                {"range": [0,  28], "color": "#2a1a1a"},
                {"range": [28, 48], "color": "#2a2a1a"},
                {"range": [48, 72], "color": "#1a2a3a"},
                {"range": [72, 100], "color": "#1a3a1a"},
            ],
        },
    ))
    fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=10), **DARK)
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_met:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📁 Segments",   f"{n_segments:,}")
    m2.metric("📏 Taille moy.", f"{avg_size:,.0f}")
    m3.metric("Min / Max",     f"{min_size:,} / {max_size:,}")
    m4.metric("⚠️ Trop petits", f"{n_too_small}")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("📄 Pages moy.", f"{pages_avg}")
    m6.metric("📄 Pages max",  f"{pages_max_val}")
    m7.metric("% hors limite", f"{pct_hors:.0f}%",
              delta="OK" if pct_hors == 0 else f"{pct_hors:.0f}% → ⚠️",
              delta_color="normal" if pct_hors == 0 else "inverse")
    m8.metric("% trop petits", f"{pct_trop_petits:.0f}%")

if depasse_max:
    n_hors = int((sim_df["nb_lignes"] > limite_lignes).sum())
    st.warning(
        f"⚠️ **{n_hors} segment(s)** dépassent la limite de 300 pages "
        f"({limite_lignes:,} lignes/segment avec {lignes_par_page} lignes/page)"
    )

# ── Histogramme distribution ──────────────────────────────────────────────────
st.markdown("#### 📈 Distribution des tailles de segments")
try:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=sim_df["nb_lignes"], nbinsx=min(50, n_segments),
        marker_color="#58a6ff", marker_line_color="#21262d", marker_line_width=0.5,
        name="Distribution",
        hovertemplate="<b>%{x:,} lignes</b><br>%{y} segments<extra></extra>",
    ))
    fig_hist.add_vline(
        x=min_rows, line_dash="dash", line_color="#f85149",
        annotation_text=f"Seuil min ({min_rows:,})", annotation_font_color="#f85149",
    )
    fig_hist.add_vline(
        x=avg_size, line_dash="dot", line_color="#3fb950",
        annotation_text=f"Moy. ({avg_size:,.0f})", annotation_font_color="#3fb950",
        annotation_position="top right",
    )
    # Ligne limite 300 pages
    fig_hist.add_vline(
        x=limite_lignes, line_dash="dash", line_color="#d29922", line_width=2,
        annotation_text=f"300 pages ({limite_lignes:,})", annotation_font_color="#d29922",
        annotation_position="top left",
    )
    fig_hist.update_layout(
        height=300, margin=dict(l=10, r=10, t=40, b=30),
        xaxis_title="Taille du segment (nb lignes)", yaxis_title="Nb segments",
        **DARK,
    )
    st.plotly_chart(fig_hist, use_container_width=True,
                    config={"toImageButtonOptions": {"format": "png", "filename": "distribution_segments"}})
except Exception as e:
    st.error(f"Erreur histogramme : {e}")

# ── Bar chart segments ────────────────────────────────────────────────────────
st.markdown("#### 📊 Taille par segment")
try:
    sim_df_disp = sim_df.copy()
    sim_df_disp["couleur"] = sim_df_disp["nb_lignes"].apply(
        lambda x: "Trop petit" if x < min_rows else ("Hors limite" if x > limite_lignes else "OK")
    )
    show_top = min(n_segments, 80)
    sim_top  = sim_df_disp.head(show_top)
    if n_segments > show_top:
        st.caption(f"📊 Affichage des {show_top} plus grands segments sur {n_segments} total")

    fig_sim = go.Figure()
    # OK
    ok_mask = sim_top["couleur"] == "OK"
    if ok_mask.any():
        fig_sim.add_trace(go.Bar(
            x=sim_top[ok_mask]["nb_lignes"], y=sim_top[ok_mask]["segment"],
            orientation="h", name="OK", marker_color="#58a6ff",
            text=sim_top[ok_mask].apply(
                lambda r: f"{r['nb_lignes']:,} (~{estimate_pages(int(r['nb_lignes']), float(lignes_par_page))[0]}p)",
                axis=1,
            ),
            textposition="outside",
        ))
    # Hors limite pages
    hl_mask = sim_top["couleur"] == "Hors limite"
    if hl_mask.any():
        fig_sim.add_trace(go.Bar(
            x=sim_top[hl_mask]["nb_lignes"], y=sim_top[hl_mask]["segment"],
            orientation="h", name=f"> 300 pages", marker_color="#d29922",
            text=sim_top[hl_mask].apply(
                lambda r: f"{r['nb_lignes']:,} (~{estimate_pages(int(r['nb_lignes']), float(lignes_par_page))[0]}p)",
                axis=1,
            ),
            textposition="outside",
        ))
    # Trop petits
    small_mask = sim_top["couleur"] == "Trop petit"
    if small_mask.any():
        fig_sim.add_trace(go.Bar(
            x=sim_top[small_mask]["nb_lignes"], y=sim_top[small_mask]["segment"],
            orientation="h", name=f"< {min_rows} lignes", marker_color="#f85149",
            text=sim_top[small_mask]["nb_lignes"].apply(lambda x: f"{x:,}"),
            textposition="outside",
        ))

    # Ligne seuil minimum
    fig_sim.add_vline(x=min_rows, line_dash="dash", line_color="#f85149",
                      annotation_text=f"Min ({min_rows:,})", annotation_font_color="#f85149")
    # Ligne limite 300 pages (rouge orangé)
    fig_sim.add_vline(x=limite_lignes, line_dash="dash", line_color="#d29922", line_width=2,
                      annotation_text=f"300p ({limite_lignes:,})",
                      annotation_font_color="#d29922", annotation_position="top left")

    fig_sim.update_layout(
        height=max(350, show_top * 24), barmode="overlay",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=140, t=20, b=20),
        legend=dict(orientation="h", y=1.02),
        xaxis_title="Nombre de lignes", **DARK,
    )
    st.plotly_chart(fig_sim, use_container_width=True,
                    config={"toImageButtonOptions": {"format": "png", "filename": "segments_bar"}})
except Exception as e:
    st.error(f"Erreur bar chart : {e}")

st.markdown("---")

# ── Preview segment ───────────────────────────────────────────────────────────
st.markdown("### 🔍 Aperçu d'un segment")
try:
    selected_segment = st.selectbox(
        "Segment à inspecter",
        options=sim_df["segment"].tolist(),
        format_func=lambda s: (
            f"{s}  ({sim_df[sim_df['segment']==s]['nb_lignes'].values[0]:,} lignes"
            f" ≈ {estimate_pages(int(sim_df[sim_df['segment']==s]['nb_lignes'].values[0]), float(lignes_par_page))[0]} pages)"
        ),
        key="sim_segment_select",
    )

    if selected_segment:
        seg_values = selected_segment.split(" | ")
        mask = pd.Series([True] * len(df_page), index=df_page.index)
        for col, val in zip(split_cols, seg_values):
            if col not in df_page.columns:
                continue
            col_series = df_page[col].copy()
            if hasattr(col_series, "cat"):
                col_series = col_series.astype(str)
            if val == "(non renseigné)":
                mask &= df_page[col].isna()
            else:
                mask &= col_series.fillna("(non renseigné)") == val
        seg_df = df_page[mask]

        seg_nb_lignes = len(seg_df)
        seg_nb_pages, seg_depasse = estimate_pages(seg_nb_lignes, float(lignes_par_page))

        pc1, pc2 = st.columns([3, 1])
        with pc1:
            st.caption(f"50 premières lignes sur {seg_nb_lignes:,} au total")
            st.dataframe(safe_df_for_display(seg_df.head(50)), use_container_width=True, height=300)
        with pc2:
            st.markdown("**Stats du segment**")
            st.metric("Lignes",       f"{seg_nb_lignes:,}")
            st.metric("Part du total", f"{round(seg_nb_lignes/len(df_page)*100, 1)}%")
            st.metric("📄 Pages",     f"{seg_nb_pages}",
                      delta="⚠️ > 300p" if seg_depasse else "✅ OK",
                      delta_color="inverse" if seg_depasse else "normal")
            num_cols_seg = [c for c, p in profiles.items()
                            if p.col_type == "numeric" and c in seg_df.columns]
            for nc in num_cols_seg[:3]:
                try:
                    st.metric(f"Moy. {nc[:12]}", f"{seg_df[nc].mean():,.2f}")
                except Exception:
                    pass
except Exception as e:
    st.error(f"Erreur preview : {e}")

st.markdown("---")

# ── Export ────────────────────────────────────────────────────────────────────
st.markdown("### 💾 Export")
ec1, ec2, ec3 = st.columns(3)

with ec1:
    st.download_button(
        label="📥 Rapport de découpe (CSV)",
        data=sim_df.to_csv(index=False),
        file_name=f"rapport_rag_{'_'.join(split_cols)}.csv",
        mime="text/csv", use_container_width=True,
    )

with ec2:
    try:
        if selected_segment and len(seg_df) > 0:
            fname = selected_segment[:40].replace(" | ", "_").replace("/", "-")
            st.download_button(
                label="📥 Segment sélectionné (CSV)",
                data=seg_df.to_csv(index=False),
                file_name=f"segment_{fname}.csv",
                mime="text/csv", use_container_width=True,
            )
    except Exception:
        pass

with ec3:
    if st.button("📦 Générer ZIP + index.json", use_container_width=True):
        with st.spinner(f"Génération de {n_segments} fichiers CSV…"):
            try:
                from utils.exporter import export_segments_zip
                zip_bytes = export_segments_zip(df_page, split_cols, profiles, min_rows=min_rows)
                st.download_button(
                    label="⬇️ Télécharger le ZIP",
                    data=zip_bytes,
                    file_name=f"segments_rag_{'_'.join(split_cols)}.zip",
                    mime="application/zip", use_container_width=True,
                )
            except Exception as e:
                st.error(f"Erreur génération ZIP : {e}")

st.markdown("---")
st.caption(
    f"💡 Cible RAG : 50–{limite_lignes:,} lignes par segment "
    f"(= 1–300 pages avec {lignes_par_page} lignes/page). "
    "Trop petit = manque de contexte. Trop grand = information noyée."
)
