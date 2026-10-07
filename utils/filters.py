"""
utils/filters.py
FilterEngine — système de filtres dynamiques adaptatifs.

Pour chaque colonne, génère le bon widget selon son type :
  - categorical / binary (≤ 50 valeurs) → multiselect + tout sélectionner
  - categorical (> 50 valeurs)           → text_input recherche partielle
  - numeric                              → slider double (min/max)
  - date                                 → date_input range
  - text_free / id                       → text_input recherche libre

Persistance dans st.session_state, filtres sidebar + page combinés.
"""
import json
import pandas as pd
import streamlit as st
from typing import Any, Dict, List, Optional, Tuple


# ── FilterEngine ──────────────────────────────────────────────────────────────

class FilterEngine:
    """Moteur de filtres dynamiques adaptatifs."""

    def __init__(self, namespace: str = "filter"):
        """
        namespace : préfixe unique pour les clés session_state
                    (permet d'avoir sidebar + page sans collision).
        """
        self.ns = namespace

    # ── Génération des widgets ────────────────────────────────────────────────

    def build_filters(
        self,
        df: pd.DataFrame,
        profiles: dict,
        cols: Optional[List[str]] = None,
        compact: bool = False,
    ) -> None:
        """
        Affiche les widgets de filtre pour les colonnes demandées.
        cols=None → toutes les colonnes.
        compact=True → disposition sur 2 colonnes.
        """
        target_cols = cols if cols is not None else list(df.columns)
        target_cols = [c for c in target_cols if c in df.columns]

        if not target_cols:
            st.caption("Aucune colonne disponible pour les filtres.")
            return

        if compact:
            pairs = [target_cols[i:i+2] for i in range(0, len(target_cols), 2)]
            for pair in pairs:
                ncols = len(pair)
                layout_cols = st.columns(ncols)
                for idx, col in enumerate(pair):
                    with layout_cols[idx]:
                        self._widget_for_col(df, profiles, col)
        else:
            for col in target_cols:
                self._widget_for_col(df, profiles, col)

    def _widget_for_col(self, df: pd.DataFrame, profiles: dict, col: str) -> None:
        """Affiche le widget adapté à la colonne."""
        p = profiles.get(col)
        col_type = p.col_type if p else "categorical"
        key = f"{self.ns}__{col}"

        try:
            if col_type in ("categorical", "binary"):
                n_unique = df[col].nunique()
                if n_unique <= 50:
                    self._multiselect_widget(df, col, key)
                else:
                    self._text_search_widget(col, key, label=f"🔍 {col} (recherche)")

            elif col_type == "numeric":
                self._range_slider_widget(df, col, key)

            elif col_type == "date":
                self._date_range_widget(df, col, key)

            else:  # text_free, id
                self._text_search_widget(col, key, label=f"🔍 {col}")

        except Exception as e:
            st.caption(f"Filtre indisponible pour `{col}` : {e}")

    def _multiselect_widget(self, df: pd.DataFrame, col: str, key: str) -> None:
        """Multiselect avec option Tout / Rien."""
        try:
            values = sorted(df[col].dropna().astype(str).unique().tolist())
        except Exception:
            values = []

        if not values:
            return

        toggle_key = f"{key}__all"
        if toggle_key not in st.session_state:
            st.session_state[toggle_key] = True  # tout sélectionné par défaut

        col1, col2 = st.columns([4, 1])
        with col2:
            all_selected = st.checkbox("Tout", value=st.session_state[toggle_key], key=f"{toggle_key}_cb",
                                        label_visibility="collapsed")
            st.session_state[toggle_key] = all_selected

        with col1:
            default = values if all_selected else (st.session_state.get(key) or [])
            selected = st.multiselect(
                label=col,
                options=values,
                default=default if all_selected else default,
                key=key,
                placeholder="Toutes les valeurs...",
            )

    def _text_search_widget(self, col: str, key: str, label: str) -> None:
        st.text_input(label, value=st.session_state.get(key, ""), key=key,
                      placeholder="Recherche partielle…")

    def _range_slider_widget(self, df: pd.DataFrame, col: str, key: str) -> None:
        try:
            s = df[col].dropna().astype(float)
            if len(s) == 0:
                return
            col_min = float(s.min())
            col_max = float(s.max())
            if col_min == col_max:
                st.caption(f"{col} : valeur unique {col_min}")
                return

            touched_key = f"{key}__touched"

            # Toujours recaler sur la plage courante (données peuvent changer)
            step = _nice_step(col_min, col_max)
            stored = st.session_state.get(key, (col_min, col_max))
            current = (
                max(col_min, min(col_max, stored[0])),
                max(col_min, min(col_max, stored[1])),
            )
            if current[0] == current[1]:
                current = (col_min, col_max)

            def _mark_touched(k=touched_key):
                st.session_state[k] = True

            val = st.slider(
                label=col,
                min_value=col_min, max_value=col_max,
                value=current, step=step, key=key,
                format="%g",
                on_change=_mark_touched,
            )
            st.session_state[key] = val
        except Exception:
            pass

    def _date_range_widget(self, df: pd.DataFrame, col: str, key: str) -> None:
        try:
            import datetime
            s = pd.to_datetime(df[col], errors="coerce").dropna()
            if len(s) == 0:
                return
            d_min = s.min().date()
            d_max = s.max().date()

            key_start = f"{key}__start"
            key_end   = f"{key}__end"
            touched_key = f"{key}__touched"

            def _mark_date_touched(k=touched_key):
                st.session_state[k] = True

            c1, c2 = st.columns(2)
            with c1:
                st.date_input(f"{col} — depuis", value=st.session_state.get(key_start, d_min),
                              min_value=d_min, max_value=d_max, key=key_start,
                              on_change=_mark_date_touched)
            with c2:
                st.date_input(f"{col} — jusqu'à", value=st.session_state.get(key_end, d_max),
                              min_value=d_min, max_value=d_max, key=key_end,
                              on_change=_mark_date_touched)
        except Exception:
            pass

    # ── Application des filtres ───────────────────────────────────────────────

    def apply_filters(self, df: pd.DataFrame, profiles: dict,
                      cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Applique tous les filtres actifs sur df.
        Retourne le DataFrame filtré.
        """
        if df is None or len(df) == 0:
            return df

        target_cols = cols if cols is not None else list(df.columns)
        result = df.copy()

        for col in target_cols:
            if col not in df.columns:
                continue
            p = profiles.get(col)
            col_type = p.col_type if p else "categorical"
            key = f"{self.ns}__{col}"

            try:
                if col_type in ("categorical", "binary"):
                    n_unique = df[col].nunique()
                    if n_unique <= 50:
                        # N'appliquer que si l'utilisateur a décoché "Tout"
                        all_flag = st.session_state.get(f"{key}__all", True)
                        if not all_flag:
                            selected = st.session_state.get(key, [])
                            if selected:
                                result = result[result[col].astype(str).isin(selected)]
                    else:
                        text = st.session_state.get(key, "").strip()
                        if text:
                            result = result[result[col].astype(str).str.contains(text, case=False, na=False)]

                elif col_type == "numeric":
                    # N'appliquer que si l'utilisateur a explicitement bougé le slider
                    if st.session_state.get(f"{key}__touched", False):
                        val = st.session_state.get(key)
                        if val is not None and isinstance(val, (list, tuple)) and len(val) == 2:
                            lo, hi = val
                            s = pd.to_numeric(result[col], errors="coerce")
                            s_min = s.min()
                            s_max = s.max()
                            if not (pd.isna(s_min) or (lo <= s_min and hi >= s_max)):
                                result = result[(s >= lo) & (s <= hi)]

                elif col_type == "date":
                    # N'appliquer que si l'utilisateur a explicitement changé une date
                    if st.session_state.get(f"{key}__touched", False):
                        import datetime
                        d_start = st.session_state.get(f"{key}__start")
                        d_end   = st.session_state.get(f"{key}__end")
                        if d_start or d_end:
                            s = pd.to_datetime(result[col], errors="coerce")
                            if d_start:
                                result = result[s >= pd.Timestamp(d_start)]
                            if d_end:
                                result = result[s <= pd.Timestamp(d_end)]

                else:  # text_free, id
                    text = st.session_state.get(key, "").strip()
                    if text:
                        result = result[result[col].astype(str).str.contains(text, case=False, na=False)]

            except Exception:
                pass

        return result

    # ── Résumé des filtres ────────────────────────────────────────────────────

    def get_filter_summary(self, profiles: dict, cols: Optional[List[str]] = None) -> str:
        """
        Retourne un résumé lisible des filtres actifs.
        Ex : "Secteur = [A, B] · Prix entre 10 et 500"
        """
        parts: List[str] = []
        target_cols = cols if cols is not None else list(profiles.keys())

        for col in target_cols:
            p = profiles.get(col)
            col_type = p.col_type if p else "categorical"
            key = f"{self.ns}__{col}"

            try:
                if col_type in ("categorical", "binary"):
                    n_unique = p.n_unique if p else 999
                    if n_unique <= 50:
                        all_flag = st.session_state.get(f"{key}__all", True)
                        if not all_flag:
                            selected = st.session_state.get(key, [])
                            if selected:
                                vals = ", ".join(selected[:3])
                                if len(selected) > 3:
                                    vals += f" +{len(selected) - 3}"
                                parts.append(f"**{col}** = [{vals}]")
                    else:
                        text = st.session_state.get(key, "").strip()
                        if text:
                            parts.append(f"**{col}** contient \"{text}\"")

                elif col_type == "numeric":
                    if st.session_state.get(f"{key}__touched", False):
                        val = st.session_state.get(key)
                        if val and isinstance(val, (list, tuple)):
                            parts.append(f"**{col}** entre {val[0]:g} et {val[1]:g}")

                elif col_type == "date":
                    if st.session_state.get(f"{key}__touched", False):
                        d_start = st.session_state.get(f"{key}__start")
                        d_end   = st.session_state.get(f"{key}__end")
                        if d_start or d_end:
                            parts.append(f"**{col}** du {d_start or '?'} au {d_end or '?'}")

                else:
                    text = st.session_state.get(key, "").strip()
                    if text:
                        parts.append(f"**{col}** contient \"{text}\"")
            except Exception:
                pass

        return " · ".join(parts) if parts else ""

    def count_active(self, profiles: dict, cols: Optional[List[str]] = None) -> int:
        """Retourne le nombre de filtres actifs."""
        target_cols = cols if cols is not None else list(profiles.keys())
        count = 0
        for col in target_cols:
            p = profiles.get(col)
            col_type = p.col_type if p else "categorical"
            key = f"{self.ns}__{col}"
            try:
                if col_type in ("categorical", "binary"):
                    n_unique = p.n_unique if p else 999
                    if n_unique <= 50:
                        if st.session_state.get(key, []):
                            count += 1
                    else:
                        if st.session_state.get(key, "").strip():
                            count += 1
                elif col_type == "numeric":
                    val = st.session_state.get(key)
                    if val and isinstance(val, (list, tuple)):
                        count += 1
                elif col_type == "date":
                    if st.session_state.get(f"{key}__start") or st.session_state.get(f"{key}__end"):
                        count += 1
                else:
                    if st.session_state.get(key, "").strip():
                        count += 1
            except Exception:
                pass
        return count

    def reset_filters(self, profiles: dict, cols: Optional[List[str]] = None) -> None:
        """Réinitialise tous les filtres dans session_state."""
        target_cols = cols if cols is not None else list(profiles.keys())
        for col in target_cols:
            p = profiles.get(col)
            col_type = p.col_type if p else "categorical"
            key = f"{self.ns}__{col}"
            for k in [key, f"{key}__all", f"{key}__all_cb", f"{key}__touched", f"{key}__start", f"{key}__end"]:
                if k in st.session_state:
                    del st.session_state[k]

    # ── Sauvegarde de preset ──────────────────────────────────────────────────

    def save_preset(self, name: str, profiles: dict, cols: Optional[List[str]] = None) -> None:
        """Sauvegarde l'état actuel des filtres sous un nom."""
        target_cols = cols if cols is not None else list(profiles.keys())
        snapshot: Dict[str, Any] = {}
        for col in target_cols:
            key = f"{self.ns}__{col}"
            for suffix in ["", "__all", "__start", "__end"]:
                k = f"{key}{suffix}"
                if k in st.session_state:
                    snapshot[k] = str(st.session_state[k])

        presets = st.session_state.get("filter_presets", {})
        presets[name] = snapshot
        st.session_state["filter_presets"] = presets

    def load_preset(self, name: str) -> None:
        """Restaure un preset de filtres."""
        presets = st.session_state.get("filter_presets", {})
        if name not in presets:
            return
        for k, v in presets[name].items():
            st.session_state[k] = v

    def list_presets(self) -> List[str]:
        return list(st.session_state.get("filter_presets", {}).keys())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _nice_step(col_min: float, col_max: float) -> float:
    """Calcule un step agréable pour le slider."""
    r = col_max - col_min
    if r == 0:
        return 1.0
    magnitude = 10 ** (len(str(int(r))) - 2)
    return max(float(magnitude), 0.01)


def show_filter_banner(summary: str, n_filtered: int, n_total: int) -> None:
    """Affiche le bandeau de résumé des filtres actifs."""
    if not summary:
        return
    pct = round(n_filtered / n_total * 100, 1) if n_total > 0 else 0
    color = "#3fb950" if pct > 50 else "#d29922" if pct > 10 else "#f85149"
    st.markdown(
        f"""<div style="background:#161b22;border:1px solid #21262d;border-radius:6px;
        padding:0.4rem 0.8rem;margin-bottom:0.8rem;font-size:0.85rem;color:#c9d1d9">
        🎛️ Filtres actifs : {summary}
        &nbsp;·&nbsp;<span style="color:{color};font-weight:600">
        {n_filtered:,} / {n_total:,} lignes ({pct}%)</span></div>""",
        unsafe_allow_html=True,
    )
