"""
pages/8_Consultation_LLM.py
Consultation LLM — approche copier/coller + arbitre API.
1. Générez le prompt → copiez dans vos LLMs préférés
2. Collez les réponses JSON dans les 3 cases
3. Cliquez Analyser
4. Lancez l'arbitrage via API pour la recommandation finale
"""
import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Consultation LLM", page_icon="🤖", layout="wide")

from utils.guards import check_data_loaded
from utils.profiler import DARK_LAYOUT as DARK, safe_df_for_display  # noqa: F401
from utils.llm_context import (
    OPENAI_API_KEY, GOOGLE_API_KEY,
    build_context_json, build_full_prompt,
    build_arbitre_prompt, _parse_llm_json,
    call_arbitre_api,
)

_, df, profiles = check_data_loaded()


# ── Constantes d'affichage ────────────────────────────────────────────────────
LLM_LABELS       = ["LLM 1", "LLM 2", "LLM 3"]
LLM_COLORS       = ["#58a6ff", "#d29922", "#3fb950"]
LLM_PLACEHOLDERS = [
    'Collez ici la réponse JSON de GPT...',
    'Collez ici la réponse JSON de Gemini...',
    'Collez ici la réponse JSON de Claude ou autre...',
]


# ── Renderers ─────────────────────────────────────────────────────────────────

def _render_llm_response(result: dict, label: str) -> None:
    """Affiche une réponse LLM parsée en blocs HTML stylisés."""
    if not result.get("success"):
        st.error(f"❌ {label} — {result.get('error', 'Erreur inconnue')}")
        if result.get("raw"):
            with st.expander("Réponse brute"):
                st.text(result["raw"][:600])
        return

    data = result.get("data", {})
    if not data:
        st.warning(f"⚠️ {label} — réponse vide")
        return

    reco = data.get("recommandation", {})
    alt  = data.get("alternative", {})
    cols_str = " × ".join(reco.get("colonnes", ["?"]))

    # Recommandation — vert
    st.markdown(
        f"""<div style="background:#1a3a1a;border-left:3px solid #3fb950;
        border-radius:0 6px 6px 0;padding:0.8rem 1rem;margin-bottom:0.5rem">
        <div style="color:#3fb950;font-weight:700;font-size:0.75rem;
        text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.35rem">
        🎯 Recommandation</div>
        <div style="color:#f0f6fc;font-size:1rem;font-weight:700">
        {cols_str}</div>
        <div style="color:#c9d1d9;font-size:0.84rem;margin-top:0.25rem">
        {reco.get('nb_chunks', '?')} chunks ·
        ~{reco.get('taille_moyenne_lignes', '?')} lignes/chunk</div>
        <div style="color:#8b949e;font-size:0.8rem;margin-top:0.25rem;
        font-style:italic">{reco.get('description', '')}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Justification — bleu
    if data.get("justification"):
        st.markdown(
            f"""<div style="background:#161b22;border-left:3px solid #58a6ff;
            border-radius:0 6px 6px 0;padding:0.8rem 1rem;margin-bottom:0.5rem">
            <div style="color:#58a6ff;font-weight:700;font-size:0.75rem;
            text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.35rem">
            📖 Justification</div>
            <div style="color:#c9d1d9;font-size:0.86rem;line-height:1.6">
            {data['justification']}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Risques — orange
    if data.get("risques"):
        st.markdown(
            f"""<div style="background:#2a2a1a;border-left:3px solid #d29922;
            border-radius:0 6px 6px 0;padding:0.8rem 1rem;margin-bottom:0.5rem">
            <div style="color:#d29922;font-weight:700;font-size:0.75rem;
            text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.35rem">
            ⚠️ Risques</div>
            <div style="color:#c9d1d9;font-size:0.86rem;line-height:1.6">
            {data['risques']}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Alternative — gris
    if alt.get("colonnes"):
        alt_cols = " × ".join(alt.get("colonnes", []))
        st.markdown(
            f"""<div style="background:#161b22;border-left:3px solid #8b949e;
            border-radius:0 6px 6px 0;padding:0.8rem 1rem;margin-bottom:0.5rem">
            <div style="color:#8b949e;font-weight:700;font-size:0.75rem;
            text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.35rem">
            🔄 Alternative</div>
            <div style="color:#c9d1d9;font-size:0.88rem;font-weight:600">
            {alt_cols}</div>
            <div style="color:#8b949e;font-size:0.8rem;margin-top:0.2rem">
            {alt.get('nb_chunks', '?')} chunks ·
            ~{alt.get('taille_moyenne_lignes', '?')} lignes/chunk</div>
            <div style="color:#8b949e;font-size:0.8rem;font-style:italic">
            {alt.get('description', '')}</div>
            </div>""",
            unsafe_allow_html=True,
        )


def _render_arbitre_response(result: dict, model_name: str) -> None:
    """Affiche le résultat de l'arbitre via components.html pour éviter le HTML brut."""
    if not result.get("success"):
        st.error(
            f"❌ Arbitre ({model_name}) — "
            f"{result.get('error', 'Erreur inconnue')}"
        )
        if result.get("raw"):
            with st.expander("Réponse brute"):
                st.text(result["raw"][:600])
        return

    data     = result.get("data", {})
    reco     = data.get("recommandation_finale", {})
    cols_str = " × ".join(reco.get("colonnes", ["?"]))

    html_content = f"""
    <div style="
        background:linear-gradient(135deg,#1a2a3a,#1a3a2a);
        border:2px solid #58a6ff;
        border-radius:12px;
        padding:1.5rem;
        font-family:'DM Sans',sans-serif;
        color:#c9d1d9;
    ">
        <div style="color:#58a6ff;font-size:0.72rem;font-weight:700;
        text-transform:uppercase;letter-spacing:0.12em;margin-bottom:1rem">
        ⚖️ Recommandation finale — Arbitre {model_name}
        </div>

        <div style="color:#3fb950;font-size:1.15rem;font-weight:800;
        margin-bottom:0.25rem">
        🏆 {cols_str}</div>

        <div style="color:#c9d1d9;font-size:0.9rem;margin-bottom:1.2rem">
        {reco.get('nb_chunks','?')} chunks ·
        ~{reco.get('taille_moyenne_lignes','?')} lignes/chunk ·
        <em>{reco.get('description','')}</em>
        </div>

        <div style="color:#58a6ff;font-weight:700;font-size:0.72rem;
        text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.35rem">
        ⚖️ Analyse comparative</div>
        <div style="font-size:0.86rem;line-height:1.6;margin-bottom:1rem">
        {data.get('analyse_comparative','')}</div>

        <div style="color:#3fb950;font-weight:700;font-size:0.72rem;
        text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.35rem">
        💡 Justification</div>
        <div style="font-size:0.86rem;line-height:1.6;margin-bottom:1rem">
        {data.get('justification','')}</div>

        <div style="color:#d29922;font-weight:700;font-size:0.72rem;
        text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.35rem">
        📋 Plan d'action</div>
        <div style="font-size:0.86rem;line-height:1.6">
        {data.get('plan_action','')}</div>
    </div>
    """

    total_chars = sum(len(str(v)) for v in data.values())
    height      = max(400, min(900, 350 + total_chars // 3))

    components.html(html_content, height=height, scrolling=True)

    # Bouton appliquer dans Simulateur RAG (hors composant iframe)
    st.markdown("---")
    st.markdown("#### 🎯 Appliquer dans le Simulateur RAG")
    suggested  = reco.get("colonnes", [])
    apply_cols = st.multiselect(
        "Colonnes à appliquer",
        options=list(df.columns),
        default=[c for c in suggested if c in df.columns],
        key="llm_apply_cols",
    )
    if apply_cols:
        label_btn = "` × `".join(apply_cols)
        if st.button(
            f"▶ Appliquer `{label_btn}` dans le Simulateur",
            type="primary",
            key="llm_apply_btn",
        ):
            valid_cols = [c for c in apply_cols if c in df.columns]
            if valid_cols:
                st.session_state["rag_selected_cols"] = valid_cols
                st.session_state["sim_cols_multi"]    = valid_cols
                st.success(
                    f"✅ `{'` × `'.join(valid_cols)}` configuré "
                    f"— allez sur la page Simulateur RAG."
                )


# ════════════════════════════════════════════════════════════
# EN-TÊTE
# ════════════════════════════════════════════════════════════
st.markdown("# 🤖 Consultation LLM")
st.markdown(
    "Générez le prompt, consultez vos LLMs préférés, "
    "collez les réponses et lancez l'arbitrage."
)

with st.expander("💡 Comment utiliser cette page", expanded=False):
    st.markdown("""
1. **Configurez** le contexte et choisissez les lignes d'aperçu
2. **Copiez** le prompt généré (onglet "Prompt à copier")
3. **Collez-le** dans GPT / Gemini / Claude (sites officiels)
4. **Collez les réponses** JSON dans les 3 cases ci-dessous
5. Cliquez **Analyser les réponses** pour voir les résultats
6. **Lancez l'arbitrage** via API pour la recommandation finale
    """)

st.markdown("---")

# ════════════════════════════════════════════════════════════
# SECTION 1 — Configuration
# ════════════════════════════════════════════════════════════
st.markdown("### ⚙️ Configuration")

cfg1, cfg2 = st.columns([3, 1])

with cfg1:
    user_context = st.text_area(
        "📝 Contexte métier (optionnel)",
        height=110,
        placeholder=(
            "Décrivez votre fichier et l'usage du chatbot...\n"
            "Ex : Catalogue produits BTP, chatbot pour "
            "conseillers commerciaux."
        ),
        key="llm_user_context",
    )

with cfg2:
    arbitre_model = st.selectbox(
        "🎯 Modèle arbitre",
        ["GPT-4o", "Gemini"],
        key="llm_arbitre_model",
        help="LLM utilisé pour l'arbitrage final via API",
    )
    if arbitre_model == "GPT-4o":
        if OPENAI_API_KEY:
            st.success("✅ OpenAI configuré")
        else:
            st.error("❌ OPENAI_API_KEY manquante dans .env")
    else:
        if GOOGLE_API_KEY:
            st.success("✅ Google configuré")
        else:
            st.error("❌ GOOGLE_API_KEY manquante dans .env")

st.markdown("---")

# ════════════════════════════════════════════════════════════
# SECTION 2 — Aperçu et génération du prompt
# ════════════════════════════════════════════════════════════
st.markdown("### 📋 Générer le prompt")

tab_prev, tab_prompt = st.tabs(["👁️ Aperçu lignes", "📄 Prompt à copier"])

with tab_prev:
    st.caption(
        "Choisissez jusqu'à 5 lignes représentatives "
        "incluses dans le prompt."
    )
    n_total = len(df)
    row_indices = []
    rc = st.columns(5)
    for i in range(5):
        with rc[i]:
            default_idx = min(i * max(1, n_total // 5), n_total - 1)
            idx = st.number_input(
                f"Ligne {i+1}",
                min_value=0,
                max_value=max(0, n_total - 1),
                value=default_idx,
                step=1,
                key=f"llm_row_{i}",
            )
            row_indices.append(int(idx))

    sensitive_defaults = [
        c for c, p in profiles.items()
        if p.semantic_type in ("email", "phone", "siret")
        and c in df.columns
    ]
    masked_cols = st.multiselect(
        "🔒 Colonnes à masquer",
        options=list(df.columns),
        default=sensitive_defaults,
        key="llm_masked_cols",
    )

    # Construire preview_rows
    preview_rows: list = []
    seen_idx: set = set()
    for idx in row_indices:
        if idx not in seen_idx and idx < n_total:
            try:
                row = {}
                raw_row = df.iloc[idx]
                for col in df.columns:
                    val = raw_row[col]
                    if col in masked_cols:
                        row[col] = "****"
                    elif (not isinstance(val, (list, dict))
                          and pd.isna(val)):
                        row[col] = None
                    elif hasattr(val, "item"):
                        row[col] = val.item()
                    elif isinstance(val, (int, float, bool,
                                         type(None), str)):
                        row[col] = val
                    else:
                        row[col] = str(val)
                preview_rows.append(row)
                seen_idx.add(idx)
            except Exception:
                pass

    if preview_rows:
        st.dataframe(
            safe_df_for_display(pd.DataFrame(preview_rows)),
            use_container_width=True,
            height=160,
        )

with tab_prompt:
    if st.button(
        "⚙️ Générer le prompt",
        type="primary",
        key="llm_gen_prompt",
    ):
        with st.spinner("Construction du contexte JSON…"):
            try:
                lignes_par_page = float(
                    st.session_state.get("lignes_par_page", 3)
                )
                ctx = build_context_json(
                    df=df,
                    profiles=profiles,
                    user_context=user_context,
                    preview_rows=preview_rows,
                    masked_cols=masked_cols,
                    lignes_par_page=lignes_par_page,
                )
                full_prompt = build_full_prompt(ctx)
                st.session_state["llm_context_json"] = ctx
                st.session_state["llm_full_prompt"]  = full_prompt
            except Exception as e:
                st.error(f"Erreur génération : {e}")

    if "llm_full_prompt" in st.session_state:
        prompt_text = st.session_state["llm_full_prompt"]
        size_kb = len(prompt_text.encode("utf-8")) / 1024
        st.caption(f"Taille du prompt : {size_kb:.1f} Ko")

        st.text_area(
            "Prompt complet — sélectionnez tout et copiez",
            value=prompt_text,
            height=350,
            key="llm_prompt_display",
            help="Ctrl+A puis Ctrl+C pour tout copier",
        )

        st.markdown(
            """<button onclick="(function(){
            var ta = document.querySelector(
                '[data-testid=stTextArea] textarea'
            );
            if(ta){navigator.clipboard.writeText(ta.value)
                .then(function(){
                    event.target.textContent='✅ Copié !';
                    setTimeout(function(){
                        event.target.textContent='📋 Copier le prompt';
                    }, 2000);
                }).catch(function(){
                    event.target.textContent='❌ Erreur';
                });
            }})()"
            style="background:#58a6ff;color:#0d1117;
            border:none;border-radius:6px;
            padding:0.5rem 1.2rem;font-weight:700;
            cursor:pointer;margin-top:0.5rem">
            📋 Copier le prompt
            </button>""",
            unsafe_allow_html=True,
        )
        st.caption(
            "💡 Collez ce prompt dans GPT, Gemini, Claude ou tout autre LLM "
            "— puis collez la réponse JSON ci-dessous."
        )

st.markdown("---")

# ════════════════════════════════════════════════════════════
# SECTION 3 — Saisie des réponses LLM
# ════════════════════════════════════════════════════════════
st.markdown("### 📥 Collez les réponses JSON de vos LLMs")
st.caption(
    "Collez la réponse brute de chaque LLM — "
    "le dashboard parse et affiche automatiquement."
)

raw_inputs = []
for i in range(3):
    raw_inputs.append(
        st.text_area(
            f"🤖 {LLM_LABELS[i]}",
            height=130,
            placeholder=LLM_PLACEHOLDERS[i],
            key=f"llm_raw_input_{i}",
        )
    )

btn_analyze = st.button(
    "✅ Analyser les réponses",
    type="primary",
    use_container_width=True,
    key="llm_analyze",
    disabled=not any(r.strip() for r in raw_inputs),
)

if btn_analyze:
    parsed_results = []
    for i, raw in enumerate(raw_inputs):
        if raw.strip():
            parsed = _parse_llm_json(raw)
            parsed["label"] = LLM_LABELS[i]
            parsed_results.append(parsed)
        else:
            parsed_results.append({
                "success": False,
                "label":   LLM_LABELS[i],
                "skipped": True,
                "data":    {},
                "error":   "",
            })
    st.session_state["llm_parsed_results"] = parsed_results
    if "llm_result_arbitre" in st.session_state:
        del st.session_state["llm_result_arbitre"]
    st.rerun()

st.markdown("---")

# ════════════════════════════════════════════════════════════
# SECTION 4 — Affichage résultats côte à côte
# ════════════════════════════════════════════════════════════
parsed_results = st.session_state.get("llm_parsed_results", [])

if parsed_results:
    valid_results = [
        r for r in parsed_results
        if not r.get("skipped") and r.get("success")
    ]
    non_skipped = [r for r in parsed_results if not r.get("skipped")]

    st.markdown("### 📊 Résultats des LLMs")

    if non_skipped:
        cols_ui = st.columns(len(non_skipped))
        for j, (col_ui, result) in enumerate(zip(cols_ui, non_skipped)):
            with col_ui:
                color = LLM_COLORS[j % len(LLM_COLORS)]
                st.markdown(
                    f"""<div style="background:#161b22;border:1px solid #21262d;
                    border-radius:8px;padding:0.6rem 1rem;margin-bottom:0.8rem">
                    <span style="color:{color};font-size:0.75rem;font-weight:700;
                    text-transform:uppercase;letter-spacing:0.1em">
                    🤖 {result['label']}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
                _render_llm_response(result, result["label"])

    st.markdown("---")

    # ════════════════════════════════════════════════════════
    # SECTION 5 — Arbitrage via API
    # ════════════════════════════════════════════════════════
    st.markdown("### ⚖️ Arbitrage — Recommandation finale")

    n_valid = len(valid_results)

    if n_valid >= 2:
        arb_key_ok = (
            (arbitre_model == "GPT-4o" and bool(OPENAI_API_KEY)) or
            (arbitre_model == "Gemini"  and bool(GOOGLE_API_KEY))
        )

        if not arb_key_ok:
            st.warning(
                f"⚠️ Clé API manquante pour {arbitre_model}. "
                "Ajoutez-la dans votre fichier .env"
            )
        elif "llm_context_json" not in st.session_state:
            st.info(
                "ℹ️ Générez d'abord le prompt (Section 2) "
                "pour activer l'arbitrage."
            )
        else:
            st.info(
                f"✅ {n_valid} réponses valides détectées — "
                f"l'arbitre {arbitre_model} peut analyser."
            )
            btn_arb = st.button(
                f"⚖️ Lancer l'arbitrage avec {arbitre_model}",
                type="primary",
                use_container_width=True,
                key="llm_arbitrage",
            )
            if btn_arb:
                with st.spinner(
                    f"⏳ {arbitre_model} compare les propositions…"
                ):
                    try:
                        res_arb = call_arbitre_api(
                            arbitre_model=arbitre_model,
                            context_json=st.session_state["llm_context_json"],
                            responses=valid_results,
                        )
                        st.session_state["llm_result_arbitre"] = res_arb
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erreur arbitrage : {e}")

    elif n_valid == 1:
        st.info(
            "ℹ️ L'arbitrage nécessite au moins **2 réponses valides**. "
            "Ajoutez une deuxième réponse LLM."
        )
    else:
        st.info("👆 Collez au moins 2 réponses LLM valides ci-dessus pour activer l'arbitrage.")

    # ── Affichage résultat arbitre ────────────────────────────────────────────
    res_arb = st.session_state.get("llm_result_arbitre", {})
    if res_arb:
        _render_arbitre_response(res_arb, arbitre_model)
