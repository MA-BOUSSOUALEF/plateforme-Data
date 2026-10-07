"""
utils/llm_context.py
Fonctions utilitaires LLM : construction du contexte JSON,
génération des prompts, parsing et appel arbitre.
Aucun widget Streamlit — fonctions pures uniquement.
"""
import os
import json
import re
import warnings
import math
warnings.filterwarnings("ignore")

import pandas as pd
from dotenv import load_dotenv

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

Tu reçois en JSON les statistiques complètes du fichier :
  - Le profil de chaque colonne
  - Le nombre exact de segments générés pour chaque
    combinaison de colonnes possible
  - Un aperçu de lignes réelles du fichier
  - Un contexte métier fourni par l'utilisateur

CONTRAINTES ABSOLUES :
  - Privilégie les colonnes catégorielles à faible cardinalité
  - Évite les colonnes avec plus de 30% de valeurs manquantes
  - Chaque chunk doit contenir entre 50 et 5000 lignes
  - Chaque PDF ne doit pas dépasser 300 pages
  - Appuie-toi sur les chiffres de segments_par_combinaison
  - Réponds UNIQUEMENT en français

FORMAT DE RÉPONSE OBLIGATOIRE :
Réponds UNIQUEMENT avec un objet JSON valide.
Aucun texte avant ou après. Aucun bloc ```json.
JSON brut uniquement. Sois concis dans les textes.

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
Plusieurs LLMs ont analysé indépendamment les mêmes
statistiques d'un fichier de données et proposé chacun
une stratégie de chunking pour un chatbot RAG utilisant
un modèle de capacité moyenne.

TON RÔLE :
1. Analyser objectivement toutes les propositions reçues
2. Identifier convergences et divergences
3. Produire une recommandation finale optimale

RÈGLES :
  - Si les LLMs convergent → confirme et renforce
  - Si ils divergent → choisis la plus solide
    OU synthétise le meilleur
  - Modèle moyen = chunks cohérents 50–2000 lignes idéal
  - Justifie avec les données statistiques
  - Réponds UNIQUEMENT en français

FORMAT DE RÉPONSE OBLIGATOIRE :
JSON brut uniquement, aucun texte autour.

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
    Parser robuste par comptage d'accolades.
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
        if escape:                 escape = False;  continue
        if ch == "\\" and in_str:  escape = True;   continue
        if ch == '"' and not escape:
            in_str = not in_str;   continue
        if in_str:                 continue
        if ch == "{":              depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:         end = i + 1; break

    if end == -1:
        return {"success": False,
                "error": "JSON tronqué", "raw": text, "data": {}}
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
    from utils.profiler import safe_fillna, get_rag_candidates
    from utils.rag_scorer import estimate_pages

    filename = st.session_state.get("filename", "?")
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
            entry["top_5_valeurs"] = [
                {
                    "valeur": str(tv.get("value", "")),
                    "pct":    f"{tv.get('count', 0) / n_rows * 100:.1f}%"
                              if n_rows > 0 else "0%",
                }
                for tv in p.top_values[:5]
            ]

            segs = {}

            # Colonne seule
            try:
                n_uniq = df[col].nunique()
                if n_uniq > 0:
                    segs[f"{col}_seule"] = {
                        "nb_segments": int(n_uniq),
                        "lignes_moy":  int(n_rows / n_uniq),
                    }
            except Exception:
                pass

            # Avec chaque autre colonne
            for autre_col in all_cols:
                if autre_col == col:
                    continue
                try:
                    p2 = profiles.get(autre_col)
                    if p2 and p2.pct_missing > 80:
                        segs[f"{col}_x_{autre_col}"] = {
                            "nb_segments": "N/A",
                            "lignes_moy":  "N/A",
                        }
                        continue

                    # Estimation rapide avant groupby
                    n_uniq_col   = p.n_unique if p.n_unique > 0 else 1
                    n_uniq_other = p2.n_unique if (p2 and p2.n_unique > 0) else df[autre_col].nunique()
                    if n_uniq_col * n_uniq_other > 50_000:
                        segs[f"{col}_x_{autre_col}"] = {
                            "nb_segments": ">50000",
                            "lignes_moy":  "<1",
                        }
                        continue

                    df_safe = safe_fillna(df[[col, autre_col]])
                    n_grp = df_safe.groupby([col, autre_col]).ngroups
                    if n_grp > 50_000:
                        segs[f"{col}_x_{autre_col}"] = {
                            "nb_segments": ">50000",
                            "lignes_moy":  "<1",
                        }
                    else:
                        segs[f"{col}_x_{autre_col}"] = {
                            "nb_segments": int(n_grp),
                            "lignes_moy":  int(n_rows / n_grp)
                                           if n_grp > 0 else 0,
                        }
                except Exception:
                    pass  # on saute silencieusement

            entry["segments_par_combinaison"] = segs

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

        n_uniq = df[col].nunique() if p.n_unique == 0 else p.n_unique
        taille_moy = int(n_rows / n_uniq) if n_uniq > 0 else 0
        pages_moy, alerte = estimate_pages(taille_moy, lignes_par_page)

        # segments_par_combinaison uniquement avec les autres candidats
        other_candidates = [c for c in candidate_names if c != col]
        segs_cand = {}

        try:
            n_uniq_col = p.n_unique if p.n_unique > 0 else 1
            if n_uniq_col <= 50_000:
                segs_cand[f"{col}_seule"] = {
                    "nb_segments": int(n_uniq_col),
                    "lignes_moy":  int(n_rows / n_uniq_col),
                }
        except Exception:
            pass

        for autre_col in other_candidates:
            try:
                p2 = profiles.get(autre_col)
                if p2 and p2.pct_missing > 80:
                    segs_cand[f"{col}_x_{autre_col}"] = {
                        "nb_segments": "N/A", "lignes_moy": "N/A",
                    }
                    continue

                n_uniq_col   = p.n_unique if p.n_unique > 0 else 1
                n_uniq_other = p2.n_unique if (p2 and p2.n_unique > 0) else df[autre_col].nunique()
                if n_uniq_col * n_uniq_other > 50_000:
                    segs_cand[f"{col}_x_{autre_col}"] = {
                        "nb_segments": ">50000", "lignes_moy": "<1",
                    }
                    continue

                df_safe = safe_fillna(df[[col, autre_col]])
                n_grp = df_safe.groupby([col, autre_col]).ngroups
                if n_grp > 50_000:
                    segs_cand[f"{col}_x_{autre_col}"] = {
                        "nb_segments": ">50000", "lignes_moy": "<1",
                    }
                else:
                    segs_cand[f"{col}_x_{autre_col}"] = {
                        "nb_segments": int(n_grp),
                        "lignes_moy":  int(n_rows / n_grp) if n_grp > 0 else 0,
                    }
            except Exception:
                pass

        bloc_c.append({
            "colonne":                  col,
            "nb_valeurs_uniques":       int(n_uniq),
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
# GÉNÉRATION DES PROMPTS
# ══════════════════════════════════════════════════════════════════

def build_full_prompt(context_json: dict) -> str:
    """Construit le prompt complet prêt à copier-coller dans n'importe quel LLM."""
    return (
        SYSTEM_PROMPT_LLM.strip()
        + "\n\n"
        + "Voici les statistiques du fichier en JSON :\n\n"
        + json.dumps(context_json, ensure_ascii=False, indent=2)
    )


def build_arbitre_prompt(
    context_json: dict,
    responses: list,  # liste de {"label": str, "data": dict}
) -> str:
    """Construit le prompt arbitre avec toutes les réponses valides."""
    parts = [ARBITRE_PROMPT.strip(), ""]
    parts.append("Voici les statistiques du fichier :")
    parts.append(json.dumps(context_json, ensure_ascii=False, indent=2))
    parts.append("")
    for i, r in enumerate(responses):
        if r.get("data"):
            parts.append(f"─── Proposition LLM {i+1} ({r['label']}) ───")
            parts.append(json.dumps(r["data"], ensure_ascii=False, indent=2))
            parts.append("")
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════
# APPEL API ARBITRE
# ══════════════════════════════════════════════════════════════════

def call_arbitre_api(
    arbitre_model: str,
    context_json: dict,
    responses: list,
) -> dict:
    """
    Seul appel API du fichier — pour l'arbitre uniquement.
    responses : liste de {"label": str, "data": dict, "success": bool}
    Retourne {success, data, error, raw}.
    """
    prompt = build_arbitre_prompt(context_json, responses)

    if arbitre_model == "GPT-4o":
        if not OPENAI_API_KEY:
            return {"success": False,
                    "error": "OPENAI_API_KEY manquante dans .env",
                    "data": {}, "raw": ""}
        try:
            import openai
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=3000,
                response_format={"type": "json_object"},
            )
            parsed = _parse_llm_json(response.choices[0].message.content)
        except Exception as e:
            return {"success": False, "error": str(e), "data": {}, "raw": ""}

    else:  # Gemini
        if not GOOGLE_API_KEY:
            return {"success": False,
                    "error": "GOOGLE_API_KEY manquante dans .env",
                    "data": {}, "raw": ""}
        try:
            import warnings
            warnings.filterwarnings("ignore")
            from google import genai as gai
            from google.genai import types as gtypes

            client = gai.Client(api_key=GOOGLE_API_KEY)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=gtypes.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=3000,
                    response_mime_type="application/json",
                ),
            )
            parsed = _parse_llm_json(response.text)
        except Exception as e:
            return {"success": False, "error": str(e), "data": {}, "raw": ""}

    return {
        "success": parsed["success"],
        "data":    parsed.get("data", {}),
        "error":   parsed.get("error", ""),
        "raw":     parsed.get("raw", ""),
    }
