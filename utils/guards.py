"""
utils/guards.py
Helpers centralisés pour les pages Streamlit :
  - check_data_loaded()    : vérifie que df est en session_state
  - apply_page_filters()   : encapsule le pattern FilterEngine répété dans chaque page
"""
import pandas as pd
import streamlit as st


def check_data_loaded() -> tuple:
    """
    Vérifie que les données sont chargées dans session_state.
    Si absent ou vide : affiche st.warning + st.stop().

    Retourne (df_full, df_filtered, profiles).
    df_full     = st.session_state.df          (DataFrame original complet)
    df_filtered = st.session_state.df_filtered (après filtres sidebar)
    profiles    = st.session_state.profiles    (dict[str, ColumnProfile])
    """
    if st.session_state.get("df") is None:
        st.warning("⚠️ Aucun fichier chargé. Retournez à l'accueil pour charger vos données.")
        st.stop()

    df_full: pd.DataFrame = st.session_state.df
    df_filtered: pd.DataFrame = st.session_state.df_filtered
    profiles: dict = st.session_state.profiles

    if df_filtered is None or len(df_filtered) == 0:
        st.warning("⚠️ Le DataFrame filtré est vide. Réinitialisez les filtres dans la barre latérale.")
        st.stop()

    return df_full, df_filtered, profiles


def apply_page_filters(
    df: pd.DataFrame,
    profiles: dict,
    page_key: str,
    show_reset: bool = True,
) -> pd.DataFrame:
    """
    Encapsule le pattern FilterEngine répété dans chaque page.

    Crée FilterEngine(f"{page_key}_filter"), affiche l'expander
    "🎛️ Filtres avancés", le bouton Réinitialiser, applique les filtres
    et affiche le bandeau de résumé.

    Si le résultat est vide → st.error + st.stop().
    Retourne le DataFrame filtré (df_page).

    Paramètres :
      df        : DataFrame source (df_filtered de session_state)
      profiles  : dict des profils de colonnes
      page_key  : clé de page unique, ex. "page1", "page4"
                  → namespace FilterEngine = "{page_key}_filter"
      show_reset: afficher le bouton Réinitialiser (défaut True)
    """
    from utils.filters import FilterEngine, show_filter_banner

    fe = FilterEngine(f"{page_key}_filter")

    with st.expander("🎛️ Filtres avancés", expanded=False):
        fe.build_filters(df, profiles, compact=True)
        if show_reset:
            if st.button("🔄 Réinitialiser", key=f"{page_key}_reset"):
                fe.reset_filters(profiles)
                st.rerun()

    df_page = fe.apply_filters(df, profiles)
    summary = fe.get_filter_summary(profiles)
    show_filter_banner(summary, len(df_page), len(df))

    if len(df_page) == 0:
        if len(df) > 0:
            # Filtres devenus invalides (données changées) → reset auto pour ne pas bloquer
            fe.reset_filters(profiles)
            st.warning("⚠️ Les filtres avancés ont été réinitialisés automatiquement (aucun résultat avec les critères précédents).")
            st.rerun()
        else:
            st.error("🚫 Aucune donnée disponible.")
            st.stop()

    return df_page
