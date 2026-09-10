"""
Composants d'interface Streamlit.

Tous les composants reçoivent des DataFrames déjà calculés et ne font aucun
calcul métier : cela garde la logique testable hors de Streamlit.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..features import FEATURE_LABELS

DISCLAIMER_MD = """
**⚠️ Transparence sur les limites du modèle**

- Ce pronostic est une **estimation de probabilités**, pas une prédiction du résultat.
  Un cheval estimé à 30 % **perd 7 fois sur 10**. L'incertitude est intrinsèque aux courses
  (incidents, tactique, état du terrain, forme du jour…).
- Le modèle n'utilise que les informations fournies : données incomplètes ou erronées
  ⇒ probabilités dégradées. Sans historique de résultats, il repose sur des coefficients
  *a priori* génériques, non calibrés sur votre population de courses.
- Les cotes de référence évoluent jusqu'au départ ; une « valeur » détectée peut disparaître.
- Aucun modèle ne garantit de gain. Les paris exposent à un **risque de perte financière** et
  peuvent créer une **dépendance**. Jouez de façon responsable : *Joueurs Info Service*
  **09 74 75 13 13** (appel non surtaxé). Interdit aux mineurs.
"""


def render_disclaimer(expanded: bool = False) -> None:
    with st.expander("Limites du modèle et jeu responsable", expanded=expanded):
        st.markdown(DISCLAIMER_MD)


def empty_runner_template(n: int = 8) -> pd.DataFrame:
    """Tableau vide pré-rempli pour la saisie manuelle d'une course."""
    return pd.DataFrame(
        {
            "horse": [f"Cheval {i + 1}" for i in range(n)],
            "draw": list(range(1, n + 1)),
            "odds": [None] * n,
            "musique": [""] * n,
            "jockey": [""] * n,
            "trainer": [""] * n,
            "weight_kg": [None] * n,
            "age": [None] * n,
            "days_since_last_run": [None] * n,
            "career_starts": [None] * n,
            "career_wins": [None] * n,
            "career_places": [None] * n,
            "earnings": [None] * n,
        }
    )


def render_prediction_table(pred: pd.DataFrame) -> None:
    """Tableau de classement lisible avec barres de probabilité."""
    cols = ["rank", "horse", "p_win", "p_place", "fair_odds", "odds", "market_p", "value"]
    optional = ["jockey", "trainer", "musique", "draw"]
    show = [c for c in cols if c in pred.columns]
    extra = [c for c in optional if c in pred.columns]
    table = pred[show + extra].copy()
    table = table.rename(
        columns={
            "rank": "Rang",
            "horse": "Cheval",
            "p_win": "P(victoire)",
            "p_place": "P(placé top 3)",
            "fair_odds": "Cote juste",
            "odds": "Cote marché",
            "market_p": "P(marché)",
            "value": "Écart modèle − marché",
            "jockey": "Jockey/Driver",
            "trainer": "Entraîneur",
            "musique": "Musique",
            "draw": "N°/Corde",
        }
    )
    config: Dict[str, object] = {
        "P(victoire)": st.column_config.ProgressColumn("P(victoire)", format="%.1f %%", min_value=0, max_value=100),
        "P(placé top 3)": st.column_config.ProgressColumn("P(placé top 3)", format="%.1f %%", min_value=0, max_value=100),
        "P(marché)": st.column_config.NumberColumn("P(marché)", format="%.1f %%"),
        "Écart modèle − marché": st.column_config.NumberColumn("Écart modèle − marché", format="%+.1f %%", help="Positif : le modèle estime le cheval plus probable que le marché (valeur potentielle). Négatif : le marché le surestime selon le modèle."),
        "Cote juste": st.column_config.NumberColumn("Cote juste", format="%.1f"),
        "Cote marché": st.column_config.NumberColumn("Cote marché", format="%.1f"),
    }
    # Toutes les probabilités sont affichées en pourcentage (0-100).
    for c in ["P(victoire)", "P(placé top 3)", "P(marché)", "Écart modèle − marché"]:
        if c in table.columns:
            table[c] = table[c] * 100
    st.dataframe(table, hide_index=True, use_container_width=True, column_config=config)


def render_probability_chart(pred: pd.DataFrame) -> None:
    """Barres horizontales : P(victoire) du modèle vs marché."""
    df = pred.sort_values("p_win", ascending=True)
    fig = go.Figure()
    fig.add_bar(y=df["horse"], x=df["p_win"] * 100, name="Modèle", orientation="h", marker_color="#1b7f5c")
    if df["market_p"].notna().any():
        fig.add_bar(y=df["horse"], x=df["market_p"] * 100, name="Marché (cotes)", orientation="h", marker_color="#b8c9c0")
    fig.update_layout(
        barmode="group",
        xaxis_title="Probabilité de victoire (%)",
        yaxis_title="",
        height=max(320, 28 * len(df) + 120),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_contributions_chart(pred: pd.DataFrame, features: List[str], horse: str) -> None:
    """Décomposition de la force d'un cheval par variable (waterfall simplifié)."""
    row = pred[pred["horse"] == horse]
    if row.empty:
        st.info("Cheval introuvable.")
        return
    row = row.iloc[0]
    data = pd.DataFrame(
        {
            "Variable": [FEATURE_LABELS.get(f, f) for f in features],
            "Contribution": [float(row.get(f"contrib_{f}", 0.0)) for f in features],
        }
    ).sort_values("Contribution")
    data["Sens"] = data["Contribution"].apply(lambda v: "Favorable" if v >= 0 else "Défavorable")
    fig = px.bar(
        data, x="Contribution", y="Variable", orientation="h", color="Sens",
        color_discrete_map={"Favorable": "#1b7f5c", "Défavorable": "#c0504d"},
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Contribution à la force (échelle logit)", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


def render_explanations(lines: List[str]) -> None:
    for line in lines:
        st.markdown(f"- {line}")


def render_coefficients(coefs: Dict[str, float]) -> None:
    df = pd.DataFrame(
        {"Variable": [FEATURE_LABELS.get(k, k) for k in coefs], "Coefficient β": list(coefs.values())}
    ).sort_values("Coefficient β")
    fig = px.bar(df, x="Coefficient β", y="Variable", orientation="h", color="Coefficient β", color_continuous_scale=["#c0504d", "#e8e8e8", "#1b7f5c"], color_continuous_midpoint=0)
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=30, b=10), coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Un coefficient positif signifie qu'un écart-type au-dessus de la moyenne de la course sur cette "
        "variable augmente la force du cheval. Les variables sont centrées par course et réduites globalement."
    )


def render_calibration_chart(calib: pd.DataFrame) -> None:
    if calib.empty:
        st.info("Pas assez de données pour tracer la calibration.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibration parfaite", line=dict(dash="dash", color="#999")))
    fig.add_trace(
        go.Scatter(
            x=calib["predite"], y=calib["observee"], mode="markers+lines", name="Modèle",
            marker=dict(size=(calib["effectif"] / calib["effectif"].max() * 25 + 6), color="#1b7f5c"),
            text=[f"n={e}" for e in calib["effectif"]],
        )
    )
    fig.update_layout(xaxis_title="Probabilité prédite", yaxis_title="Fréquence observée de victoire", height=380, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
