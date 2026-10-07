"""
utils/llm_client.py
Gestion centralisée des appels LLM (GPT-4o + Gemini).
Fonctions pures uniquement — aucun widget Streamlit.
"""
import os
import json
import re
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from dotenv import load_dotenv
from google import genai as google_genai
from google.genai import types as google_types

from utils.profiler import safe_fillna

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

GEMINI_MODEL = "models/gemini-2.5-flash"
GPT_MODEL    = "gpt-4o"


# ══════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_LLM = """
Tu es un expert senior en Data Engineering et en RAG
(Retrieval-Augmented Generation).

OBJECTIF MÉTIER :
L'utilisateur veut créer un chatbot RAG basé sur un fichier
de données. Le modèle LLM utilisé dans ce chatbot est de
capacité moyenne. Le chunking doit donc être optimal :
  - Trop grand → le modèle est noyé et répond mal
  - Trop petit → le modèle manque de contexte et répond mal
  - Idéal → chaque chunk est cohérent, autonome,
    et compréhensible par un modèle moyen

Tu reçois en JSON :
  - Les statistiques complètes du fichier
  - Le profil de chaque colonne avec le nombre exact
    de segments générés pour chaque combinaison possible
  - Un aperçu de lignes réelles du fichier
  - Un contexte métier fourni par l'utilisateur

CONTRAINTES :
  - Tu n'as accès qu'aux statistiques et à l'aperçu
  - Privilégie les colonnes catégorielles à faible cardinalité
  - Évite les colonnes avec plus de 30% de valeurs manquantes
  - Chaque chunk doit contenir entre 50 et 5000 lignes
  - Chaque PDF ne doit pas dépasser 300 pages
  - Utilise les combinaisons fournies pour estimer
    les tailles réelles des segments
  - Réponds UNIQUEMENT en français

FORMAT DE RÉPONSE OBLIGATOIRE :
Réponds UNIQUEMENT avec un objet JSON valide.
Aucun texte avant ou après. Aucun bloc ```json.
JSON brut uniquement. Sois concis dans les textes.

Structure exacte :
{
  "recommandation": {
    "colonnes": ["col1", "col2"],
    "nb_chunks": <entier>,
    "taille_moyenne_lignes": <entier>,
    "description": "<phrase courte max 20 mots>"
  },
  "justification": "<paragraphe max 80 mots>",
  "risques": "<paragraphe max 60 mots>",
  "alternative": {
    "colonnes": ["col1"],
    "nb_chunks": <entier>,
    "taille_moyenne_lignes": <entier>,
    "description": "<phrase courte max 20 mots>"
  }
}
"""

ARBITRE_PROMPT = """
Tu es un expert senior en RAG et segmentation de données
pour chatbots.

CONTEXTE :
Deux LLMs (GPT et Gemini) ont analysé indépendamment
les mêmes statistiques d'un fichier de données et proposé
chacun une stratégie de chunking pour un chatbot RAG
utilisant un modèle de capacité moyenne.

TON RÔLE :
1. Analyser objectivement les deux propositions
2. Identifier convergences et divergences
3. Produire une recommandation finale optimale

RÈGLES :
  - Si les deux convergent → confirme et renforce
  - Si elles divergent → choisis la plus solide
    OU synthétise le meilleur des deux
  - Modèle moyen = chunks cohérents ni trop grands
    ni trop petits (idéal 50–2000 lignes)
  - Justifie avec les données statistiques fournies
  - Réponds UNIQUEMENT en français

FORMAT DE RÉPONSE OBLIGATOIRE :
Réponds UNIQUEMENT avec un objet JSON valide.
Aucun texte avant ou après. Aucun bloc ```json.
JSON brut uniquement. Sois concis.

Structure exacte :
{
  "analyse_comparative": "<paragraphe max 80 mots>",
  "recommandation_finale": {
    "colonnes": ["col1", "col2"],
    "nb_chunks": <entier>,
    "taille_moyenne_lignes": <entier>,
    "description": "<phrase courte max 20 mots>"
  },
  "justification": "<paragraphe max 80 mots>",
  "plan_action": "<paragraphe max 60 mots>"
}
"""


# ══════════════════════════════════════════════════════════════════
# PARSEUR JSON ROBUSTE
# ══════════════════════════════════════════════════════════════════

def _parse_llm_json(text: str) -> dict:
    """
    Parse robuste par comptage d'accolades.
    Retire les blocs ```json``` et extrait le premier objet JSON complet.
    Retourne {"success": bool, "data": dict, "error": str, "raw": str}.
    """
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*",     "", text)
    text = text.strip()

    start = text.find("{")
    if start == -1:
        return {"success": False,
                "error": "Aucun JSON trouvé", "raw": text, "data": {}}

    depth, end, in_str, escape = 0, -1, False, False
    for i, ch in enumerate(text[start:], start):
        if escape:                 escape = False; continue
        if ch == "\\" and in_str:  escape = True;  continue
        if ch == '"' and not escape:
            in_str = not in_str;   continue
        if in_str:                 continue
        if ch == "{":              depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:         end = i + 1; break

    if end == -1:
        return {"success": False,
                "error": "JSON tronqué — tokens insuffisants",
                "raw": text, "data": {}}
    try:
        return {"success": True,
                "data": json.loads(text[start:end]),
                "error": "", "raw": text[start:end]}
    except json.JSONDecodeError as e:
        return {"success": False,
                "error": f"JSON invalide : {e}",
                "raw": text[start:end], "data": {}}


# ══════════════════════════════════════════════════════════════════
# CONSTRUCTION DU CONTEXTE JSON
# ══════════════════════════════════════════════════════════════════

def _safe_val(v):
    """Convertit une valeur en type JSON-sérialisable."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, (int, float, bool, str)):
        return v
    return str(v)


def _compute_segments_for_col(
    df: pd.DataFrame,
    col: str,
    profiles: dict,
    n_rows: int,
    other_cols=None,
) -> dict:
    """
    Calcule segments_par_combinaison pour une colonne catégorielle.
    other_cols : liste des colonnes à croiser avec col.
               Si None → toutes les colonnes du df sauf col.
    """
    result = {}
    p = profiles.get(col)

    if p and p.pct_missing > 80:
        return {f"{col}_seule": {"nb_segments": "N/A", "lignes_moy": "N/A"}}

    n_unique_col = p.n_unique if p else df[col].nunique()
    if n_unique_col > 50_000:
        result[f"{col}_seule"] = {"nb_segments": ">50000", "lignes_moy": "<1"}
    else:
        lignes_moy = round(n_rows / n_unique_col) if n_unique_col > 0 else 0
        result[f"{col}_seule"] = {"nb_segments": n_unique_col, "lignes_moy": lignes_moy}

    if other_cols is None:
        other_cols = [c for c in df.columns if c != col]

    for other_col in other_cols:
        key = f"{col}_x_{other_col}"
        p_other = profiles.get(other_col)

        if p_other and p_other.pct_missing > 80:
            result[key] = {"nb_segments": "N/A", "lignes_moy": "N/A"}
            continue

        n_other = p_other.n_unique if p_other else df[other_col].nunique()
        estimated_max = n_unique_col * n_other
        if estimated_max > 50_000:
            result[key] = {"nb_segments": ">50000", "lignes_moy": "<1"}
            continue

        try:
            df_safe = safe_fillna(df[[col, other_col]])
            nb_segs = df_safe.groupby([col, other_col]).ngroups
            if nb_segs > 50_000:
                result[key] = {"nb_segments": ">50000", "lignes_moy": "<1"}
            else:
                lignes_moy = round(n_rows / nb_segs) if nb_segs > 0 else 0
                result[key] = {"nb_segments": nb_segs, "lignes_moy": lignes_moy}
        except Exception:
            result[key] = {}

    return result


def build_context_json(
    df: pd.DataFrame,
    profiles: dict,
    user_context: str,
    preview_rows: list,
    masked_cols: list,
    lignes_par_page: float,
) -> dict:
    """
    Construit le JSON de contexte complet envoyé aux LLMs.
    Retourne un dict Python JSON-sérialisable.
    """
    import streamlit as st
    from utils.rag_scorer import estimate_pages
    from utils.profiler import get_rag_candidates

    filename = st.session_state.get("filename", "inconnu")
    n_rows   = len(df)
    all_cols = list(df.columns)

    # ── Bloc A — Général ──────────────────────────────────────────────────────
    try:
        n_dupes = int(df.duplicated().sum())
    except Exception:
        n_dupes = 0

    pct_manquants_moyen = round(
        sum(p.pct_missing for p in profiles.values()) / max(len(profiles), 1), 1
    )

    bloc_a = {
        "nom_fichier":         filename,
        "nb_lignes":           n_rows,
        "nb_colonnes":         len(all_cols),
        "pct_manquants_moyen": pct_manquants_moyen,
        "nb_doublons":         n_dupes,
    }

    # ── Bloc B — Colonnes ────────────────────────────────────────────────────
    bloc_b = []
    for col in all_cols:
        p = profiles.get(col)
        if p is None:
            continue

        entry = {
            "nom":                col,
            "type":               p.col_type,
            "type_semantique":    p.semantic_type,
            "nb_valeurs_uniques": p.n_unique,
            "pct_manquants":      p.pct_missing,
        }

        if p.col_type in ("categorical", "binary"):
            top5 = []
            for tv in p.top_values[:5]:
                cnt = tv.get("count", 0)
                pct_str = f"{round(cnt / n_rows * 100, 1)}%" if n_rows > 0 else "0%"
                top5.append({"valeur": str(tv.get("value", "")), "pct": pct_str})
            entry["top_5_valeurs"] = top5
            try:
                entry["segments_par_combinaison"] = _compute_segments_for_col(
                    df=df, col=col, profiles=profiles, n_rows=n_rows,
                    other_cols=[c for c in all_cols if c != col],
                )
            except Exception:
                entry["segments_par_combinaison"] = {}

        elif p.col_type == "numeric" and p.stats:
            entry["min"]     = _safe_val(p.stats.get("min"))
            entry["max"]     = _safe_val(p.stats.get("max"))
            entry["moyenne"] = _safe_val(p.stats.get("mean"))
            entry["segments_par_combinaison"] = {}
        else:
            entry["segments_par_combinaison"] = {}

        bloc_b.append(entry)

    # ── Bloc C — Candidats RAG ───────────────────────────────────────────────
    candidate_names = get_rag_candidates(profiles, min_score=45)
    bloc_c = []

    for col in candidate_names:
        p = profiles.get(col)
        if p is None:
            continue

        n_unique = p.n_unique if p.n_unique > 0 else 1
        taille_moy_seule = round(n_rows / n_unique)
        pages_moy, alerte = estimate_pages(taille_moy_seule, lignes_par_page)

        other_candidates = [c for c in candidate_names if c != col]
        try:
            segs_cand = _compute_segments_for_col(
                df=df, col=col, profiles=profiles, n_rows=n_rows,
                other_cols=other_candidates,
            )
        except Exception:
            segs_cand = {}

        bloc_c.append({
            "colonne":                  col,
            "nb_valeurs_uniques":       p.n_unique,
            "pages_moy_seule":          pages_moy,
            "alerte_300_pages":         alerte,
            "segments_par_combinaison": segs_cand,
        })

    # ── Aperçu données ───────────────────────────────────────────────────────
    apercu = []
    for row_dict in preview_rows[:5]:
        clean_row = {}
        for k, v in row_dict.items():
            clean_row[k] = "****" if k in masked_cols else _safe_val(v)
        apercu.append(clean_row)

    return {
        "contexte_metier":      user_context,
        "bloc_a_general":       bloc_a,
        "bloc_b_colonnes":      bloc_b,
        "bloc_c_candidats_rag": bloc_c,
        "apercu_donnees":       apercu,
    }


# ══════════════════════════════════════════════════════════════════
# APPELS LLM
# ══════════════════════════════════════════════════════════════════

def call_gpt(system_prompt: str, context_json: dict) -> dict:
    """
    Appelle GPT-4o via l'API OpenAI avec response_format json_object.
    Retourne {success, data, error, raw, model}.
    """
    if not OPENAI_API_KEY:
        return {"success": False,
                "error": "Clé OPENAI_API_KEY manquante dans .env",
                "model": GPT_MODEL, "data": {}, "raw": ""}
    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content":
                    "Voici les statistiques du fichier en JSON :\n\n"
                    + json.dumps(context_json, ensure_ascii=False, indent=2)
                },
            ],
            temperature=0.3,
            max_tokens=3000,
            response_format={"type": "json_object"},
        )
        parsed = _parse_llm_json(response.choices[0].message.content)
        return {
            "success": parsed["success"],
            "data":    parsed.get("data", {}),
            "error":   parsed.get("error", ""),
            "raw":     parsed.get("raw", ""),
            "model":   GPT_MODEL,
        }
    except Exception as e:
        return {"success": False, "error": str(e),
                "model": GPT_MODEL, "data": {}, "raw": ""}


def call_gemini(system_prompt: str, context_json: dict) -> dict:
    """
    Appelle Gemini via google-genai avec response_mime_type application/json.
    Retourne {success, data, error, raw, model}.
    """
    if not GOOGLE_API_KEY:
        return {"success": False,
                "error": "Clé GOOGLE_API_KEY manquante dans .env",
                "model": GEMINI_MODEL, "data": {}, "raw": ""}
    try:
        client = google_genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=(
                "Voici les statistiques du fichier en JSON :\n\n"
                + json.dumps(context_json, ensure_ascii=False, indent=2)
            ),
            config=google_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
                max_output_tokens=3000,
                response_mime_type="application/json",
            ),
        )
        parsed = _parse_llm_json(response.text)
        return {
            "success": parsed["success"],
            "data":    parsed.get("data", {}),
            "error":   parsed.get("error", ""),
            "raw":     parsed.get("raw", ""),
            "model":   GEMINI_MODEL,
        }
    except Exception as e:
        return {"success": False, "error": str(e),
                "model": GEMINI_MODEL, "data": {}, "raw": ""}


def call_arbitre(
    arbitre_model: str,
    arbitre_prompt: str,
    context_json: dict,
    reponse_gpt: dict,
    reponse_gemini: dict,
) -> dict:
    """
    Appelle le LLM arbitre avec les deux réponses GPT et Gemini.
    reponse_gpt / reponse_gemini : dicts résultat de call_gpt / call_gemini.
    Retourne {success, data, error, raw, model}.
    """
    def _fmt(res: dict) -> str:
        if res.get("data"):
            return json.dumps(res["data"], ensure_ascii=False, indent=2)
        return res.get("raw", "Pas de réponse disponible")

    user_message = (
        "Voici les statistiques du fichier :\n"
        + json.dumps(context_json, ensure_ascii=False, indent=2)
        + "\n\n─── Proposition de GPT-4o ───\n"
        + _fmt(reponse_gpt)
        + "\n\n─── Proposition de Gemini ───\n"
        + _fmt(reponse_gemini)
    )

    if arbitre_model == "GPT-4o":
        if not OPENAI_API_KEY:
            return {"success": False, "error": "Clé OpenAI manquante",
                    "model": GPT_MODEL, "data": {}, "raw": ""}
        try:
            import openai
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": arbitre_prompt},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.3,
                max_tokens=3000,
                response_format={"type": "json_object"},
            )
            parsed = _parse_llm_json(response.choices[0].message.content)
        except Exception as e:
            return {"success": False, "error": str(e),
                    "model": GPT_MODEL, "data": {}, "raw": ""}

    else:  # Gemini
        if not GOOGLE_API_KEY:
            return {"success": False, "error": "Clé Google manquante",
                    "model": GEMINI_MODEL, "data": {}, "raw": ""}
        try:
            client = google_genai.Client(api_key=GOOGLE_API_KEY)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_message,
                config=google_types.GenerateContentConfig(
                    system_instruction=arbitre_prompt,
                    temperature=0.3,
                    max_output_tokens=3000,
                    response_mime_type="application/json",
                ),
            )
            parsed = _parse_llm_json(response.text)
        except Exception as e:
            return {"success": False, "error": str(e),
                    "model": GEMINI_MODEL, "data": {}, "raw": ""}

    return {
        "success": parsed["success"],
        "data":    parsed.get("data", {}),
        "error":   parsed.get("error", ""),
        "raw":     parsed.get("raw", ""),
        "model":   arbitre_model,
    }
