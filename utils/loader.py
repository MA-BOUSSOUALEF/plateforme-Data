"""
utils/loader.py
Chargement universel CSV / Excel avec :
  - Détection automatique du séparateur et de l'encodage
  - Détection automatique de la ligne d'en-tête (Excel multi-titre)
  - Nettoyage des noms de colonnes (strip, espaces, caractères spéciaux)
  - Conversion automatique des colonnes date (FR et EN)
  - Rapport de chargement lisible
  - Optimisation des types mémoire (category, downcast)
"""
import re
import pandas as pd
import streamlit as st
from io import BytesIO
from typing import Tuple, List


# ── Helpers internes ──────────────────────────────────────────────────────────

def _detect_encoding(raw: bytes) -> str:
    try:
        import chardet
        detected = chardet.detect(raw[:50_000])
        return detected.get("encoding", "utf-8") or "utf-8"
    except Exception:
        return "utf-8"


def _detect_header_row(df_raw: pd.DataFrame, max_scan: int = 6) -> int:
    """
    Heuristique : la ligne avec le plus grand nombre de chaînes non-nulles
    distinctes est la ligne d'en-tête (ignore les lignes de titre).
    """
    best_row, best_score = 0, -1
    for i in range(min(max_scan, len(df_raw))):
        row = df_raw.iloc[i]
        n_str = sum(1 for v in row if isinstance(v, str) and len(str(v).strip()) > 0)
        n_unique = row.nunique()
        score = n_str * n_unique
        if score > best_score:
            best_score = score
            best_row = i
    return best_row


def _clean_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Normalise les noms de colonnes et retourne un rapport des corrections."""
    report: List[str] = []
    new_cols: List[str] = []
    seen: dict = {}

    for col in df.columns:
        original = str(col)
        clean = original.strip()
        clean = re.sub(r'\s+', '_', clean)
        clean = re.sub(r'[^\w\-]', '_', clean)
        clean = re.sub(r'_+', '_', clean)
        clean = clean.strip('_')
        if not clean:
            clean = "col"

        # Déduplications
        base = clean
        counter = seen.get(base, 0)
        if counter > 0:
            clean = f"{base}_{counter}"
        seen[base] = counter + 1

        if clean != original:
            report.append(f"🔧 `{original}` → `{clean}`")
        new_cols.append(clean)

    df.columns = new_cols
    return df, report


def _convert_dates(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Convertit silencieusement les colonnes qui ressemblent à des dates."""
    report: List[str] = []
    for col in df.select_dtypes(include="object").columns:
        try:
            sample = df[col].dropna().astype(str).head(100)
            if len(sample) < 3:
                continue
            converted = pd.to_datetime(sample, dayfirst=True, errors="coerce")
            if converted.notna().mean() >= 0.80:
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
                report.append(f"📅 Colonne `{col}` convertie en date")
        except Exception:
            pass
    return df, report


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Réduit la mémoire RAM : category pour faible cardinalité, downcast numérique."""
    for col in df.select_dtypes(include="object").columns:
        try:
            n_unique = df[col].nunique()
            if len(df) > 0 and n_unique / len(df) < 0.5:
                df[col] = df[col].astype("category")
        except Exception:
            pass

    for col in df.select_dtypes(include=["int64"]).columns:
        try:
            df[col] = pd.to_numeric(df[col], downcast="integer")
        except Exception:
            pass

    for col in df.select_dtypes(include=["float64"]).columns:
        try:
            df[col] = pd.to_numeric(df[col], downcast="float")
        except Exception:
            pass

    return df


# ── Fonctions cachées (sans widgets) ─────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_csv_cached(file_bytes: bytes) -> Tuple[pd.DataFrame, str, str]:
    """Charge un CSV depuis les bytes bruts. Retourne (df, encoding, sep)."""
    encoding = _detect_encoding(file_bytes)
    buf = BytesIO(file_bytes)

    for sep in [";", ",", "\t", "|"]:
        try:
            buf.seek(0)
            df = pd.read_csv(buf, sep=sep, encoding=encoding, low_memory=False)
            if df.shape[1] > 1:
                df = _optimize_dtypes(df)
                return df, encoding, sep
        except Exception:
            continue

    buf.seek(0)
    df = pd.read_csv(buf, encoding=encoding, low_memory=False)
    return _optimize_dtypes(df), encoding, ","


@st.cache_data(show_spinner=False)
def _load_excel_cached(file_bytes: bytes, sheet: str) -> Tuple[pd.DataFrame, int]:
    """Charge une feuille Excel depuis les bytes bruts. Retourne (df, header_row)."""
    buf = BytesIO(file_bytes)
    df_raw = pd.read_excel(buf, sheet_name=sheet, header=None)
    header_row = _detect_header_row(df_raw)
    buf.seek(0)
    df = pd.read_excel(buf, sheet_name=sheet, header=header_row)
    # L'index reflète les vrais numéros de ligne Excel (1-based, ligne 1 = première ligne)
    # header_row + 1 = ligne Excel de l'en-tête, +1 pour passer aux données
    df.index = range(header_row + 2, header_row + 2 + len(df))
    return _optimize_dtypes(df), header_row


# ── Point d'entrée principal ──────────────────────────────────────────────────

def load_file(uploaded_file) -> Tuple[pd.DataFrame, List[str]]:
    """
    Charge CSV ou Excel. Les widgets Streamlit restent ICI (hors cache).
    Retourne (df, rapport_chargement).
    """
    report: List[str] = []
    name = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    if name.endswith(".csv"):
        df, encoding, sep = _load_csv_cached(file_bytes)
        sep_display = {";" : "point-virgule", ",": "virgule", "\t": "tabulation", "|": "pipe"}.get(sep, sep)
        report.append(f"📄 CSV — séparateur : **{sep_display}**, encodage : **{encoding}**")

    elif name.endswith((".xlsx", ".xls")):
        buf = BytesIO(file_bytes)
        try:
            xl = pd.ExcelFile(buf)
            sheet_names = xl.sheet_names
        except Exception as e:
            st.error(f"Impossible de lire le fichier Excel : {e}")
            return pd.DataFrame(), []

        if len(sheet_names) > 1:
            sheet = st.selectbox(
                "📋 Plusieurs feuilles détectées — laquelle charger ?",
                sheet_names
            )
            report.append(f"📋 Feuille chargée : **{sheet}** ({len(sheet_names)} disponibles)")
        else:
            sheet = sheet_names[0]
            report.append(f"📋 Feuille : **{sheet}**")

        df, header_row = _load_excel_cached(file_bytes, sheet)
        if header_row > 0:
            report.append(f"📊 En-têtes détectés ligne **{header_row + 1}** ({header_row} ligne(s) de titre ignorée(s))")

    else:
        st.error("Format non supporté. Utilisez CSV ou Excel (.xlsx / .xls).")
        return pd.DataFrame(), []

    # Nettoyage colonnes
    df, col_report = _clean_columns(df)
    if col_report:
        report.append(f"🔧 **{len(col_report)} colonne(s) renommée(s)** : " + ", ".join(col_report[:5]))
    else:
        report.append("✅ Noms de colonnes : aucune correction nécessaire")

    # Conversion dates
    df, date_report = _convert_dates(df)
    report.extend(date_report)

    report.append(f"✅ **{len(df):,} lignes × {len(df.columns)} colonnes** chargées avec succès")
    return df, report


def get_file_info(df: pd.DataFrame) -> dict:
    """Retourne les métadonnées de base du DataFrame."""
    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2),
        "columns": list(df.columns),
    }
