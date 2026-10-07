"""
test_llm_full.py — affiche la réponse JSON complète sans troncature
"""
import os, json, re, warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

def _parse_llm_json(text):
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text).strip()
    start = text.find("{")
    if start == -1:
        return {"success": False, "error": "Aucun JSON", "raw": text}
    depth, end, in_str, escape = 0, -1, False, False
    for i, ch in enumerate(text[start:], start):
        if escape:            escape = False; continue
        if ch == "\\" and in_str: escape = True; continue
        if ch == '"' and not escape: in_str = not in_str; continue
        if in_str:            continue
        if ch == "{":         depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:   end = i + 1; break
    if end == -1:
        return {"success": False, "error": "JSON tronqué", "raw": text}
    try:
        return {"success": True, "data": json.loads(text[start:end])}
    except Exception as e:
        return {"success": False, "error": str(e), "raw": text[start:end]}

SYSTEM_PROMPT = """
Tu es un expert RAG. Réponds UNIQUEMENT en JSON brut, sans texte autour.
Structure exacte :
{
  "recommandation": {
    "colonnes": ["col1"],
    "nb_chunks": <entier>,
    "taille_moyenne_lignes": <entier>,
    "description": "<max 20 mots>"
  },
  "justification": "<max 80 mots>",
  "risques": "<max 60 mots>",
  "alternative": {
    "colonnes": ["col1"],
    "nb_chunks": <entier>,
    "taille_moyenne_lignes": <entier>,
    "description": "<max 20 mots>"
  }
}
"""

TEST_CONTEXT = {
    "bloc_a_general": {"nb_lignes": 10746, "nb_colonnes": 5},
    "bloc_b_colonnes": [
        {
            "nom": "Grande_Famille",
            "type": "categorical",
            "nb_valeurs_uniques": 6,
            "pct_manquants": 0.0,
            "segments_par_combinaison": {
                "Grande_Famille_seule":      {"nb_segments": 6,  "lignes_moy": 1791},
                "Grande_Famille_x_Zone_Geo": {"nb_segments": 48, "lignes_moy": 224}
            }
        },
        {
            "nom": "Zone_Geo",
            "type": "categorical",
            "nb_valeurs_uniques": 8,
            "pct_manquants": 1.2,
            "segments_par_combinaison": {
                "Zone_Geo_seule":            {"nb_segments": 8,  "lignes_moy": 1343},
                "Zone_Geo_x_Grande_Famille": {"nb_segments": 48, "lignes_moy": 224}
            }
        }
    ]
}

from google import genai as google_genai
from google.genai import types as google_types

client = google_genai.Client(api_key=GOOGLE_API_KEY)
response = client.models.generate_content(
    model="models/gemini-2.5-flash",
    contents="Voici les stats :\n\n" + json.dumps(TEST_CONTEXT, ensure_ascii=False, indent=2),
    config=google_types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.3,
        max_output_tokens=3000,
        response_mime_type="application/json",
    ),
)

print("=== RÉPONSE BRUTE COMPLÈTE ===")
print(response.text)
print("\n=== JSON PARSÉ ===")
parsed = _parse_llm_json(response.text)
if parsed["success"]:
    print(json.dumps(parsed["data"], ensure_ascii=False, indent=2))
    print("\n✅ JSON valide et complet")
    print(f"\nJustification complète ({len(parsed['data'].get('justification',''))} chars) :")
    print(parsed["data"].get("justification", ""))
    print(f"\nRisques complets ({len(parsed['data'].get('risques',''))} chars) :")
    print(parsed["data"].get("risques", ""))
else:
    print(f"❌ {parsed['error']}")