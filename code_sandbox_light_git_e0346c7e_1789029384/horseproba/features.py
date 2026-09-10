"""
Ingénierie des variables (features) pour le modèle de pronostic.

Principe général
----------------
Une course hippique est une COMPÉTITION : ce qui compte n'est pas la valeur
absolue d'un cheval mais sa valeur RELATIVE aux autres partants du jour.
C'est pourquoi la plupart des variables sont *centrées et réduites au sein de
chaque course* (fonction `within_race_standardize`). Un cheval « bon » dans une
course faible doit ressortir, et inversement.

Variables produites (toutes numériques, sans NaN à la sortie) :
    log_odds_implied     log de la probabilité implicite normalisée du marché
    form_score           score de forme récente pondéré (musique), décroissance exponentielle
    form_win_rate        proportion de victoires dans les 6 dernières sorties
    form_place_rate      proportion de places (1-3) dans les 6 dernières sorties
    dq_rate              proportion d'incidents (D/T/A) — surtout pertinent au trot
    career_win_rate      victoires / partants en carrière (lissé bayésien)
    career_place_rate    places / partants en carrière (lissé bayésien)
    log_earnings_per_start
    freshness            forme de "fraîcheur" : jours depuis dernière course transformé
    jockey_win_rate      taux de réussite historique du jockey (lissé bayésien)
    trainer_win_rate     taux de réussite historique de l'entraîneur (lissé bayésien)
    draw_rel             corde relative (0 = intérieur, 1 = extérieur) — plat uniquement
    weight_rel           poids relatif au poids moyen de la course
    age_rel              écart d'âge à la moyenne de la course

Le lissage bayésien (« shrinkage ») évite que 1 victoire en 1 course donne un taux
de 100 % : on ajoute k pseudo-observations au taux de base.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Parsing de la « musique » (forme récente au format français)
# --------------------------------------------------------------------------- #

# Une musique ressemble à "1p 3p Da 2a 0p (23) 5p". Les chiffres sont les places,
# les lettres minuscules la discipline (p=plat, a=attelé, m=monté, h=haies, s=steeple),
# les majuscules des incidents (D=disqualifié, T=tombé, A=arrêté, R=ret...).
# Les parenthèses contiennent l'année et sont ignorées.
_MUSIQUE_TOKEN = re.compile(r"(\d{1,2}|[DTARN])", re.IGNORECASE)


@dataclass
class FormSummary:
    positions: List[Optional[int]] = field(default_factory=list)  # None = incident
    n_runs: int = 0
    wins: int = 0
    places: int = 0
    incidents: int = 0
    score: float = 0.0


def parse_musique(musique: object, max_runs: int = 6, decay: float = 0.75) -> FormSummary:
    """
    Convertit une musique en résumé de forme.

    Le score de forme pondère chaque sortie récente par decay**k (k=0 pour la plus
    récente) et transforme la place en points :
        1 -> 1.0, 2 -> 0.75, 3 -> 0.6, 4 -> 0.45, 5 -> 0.35, 6+ -> décroissant, 0 -> 0.05,
        incident (D/T/A) -> 0.0
    Le résultat est normalisé par la somme des poids pour rester dans [0, 1].
    Une musique vide renvoie un score neutre (0.35 ≈ moyenne d'un partant lambda).
    """
    summary = FormSummary()
    if musique is None or (not isinstance(musique, str) and pd.isna(musique)):
        summary.score = 0.35
        return summary

    text = str(musique)
    text = re.sub(r"\(\d+\)", " ", text)  # supprime les années "(23)"
    tokens = _MUSIQUE_TOKEN.findall(text)
    if not tokens:
        summary.score = 0.35
        return summary

    tokens = tokens[:max_runs]
    weighted, weight_sum = 0.0, 0.0
    for k, tok in enumerate(tokens):
        w = decay**k
        weight_sum += w
        if tok.isdigit():
            pos = int(tok)
            summary.positions.append(pos)
            if pos == 1:
                pts, summary.wins = 1.0, summary.wins + 1
                summary.places += 1
            elif pos == 2:
                pts, summary.places = 0.75, summary.places + 1
            elif pos == 3:
                pts, summary.places = 0.6, summary.places + 1
            elif pos == 0:
                pts = 0.05  # "0" = non placé (au-delà de la 9e place)
            else:
                pts = max(0.05, 0.6 - 0.08 * (pos - 3))
        else:
            summary.positions.append(None)
            summary.incidents += 1
            pts = 0.0
        weighted += w * pts
    summary.n_runs = len(tokens)
    summary.score = weighted / weight_sum if weight_sum else 0.35
    return summary


# --------------------------------------------------------------------------- #
# Taux lissés (jockey, entraîneur, carrière)
# --------------------------------------------------------------------------- #

def shrunk_rate(successes: float, trials: float, prior_rate: float, k: float = 10.0) -> float:
    """Taux bayésien lissé : (succès + k*prior) / (essais + k)."""
    successes = 0.0 if pd.isna(successes) else float(successes)
    trials = 0.0 if pd.isna(trials) else float(trials)
    return (successes + k * prior_rate) / (trials + k)


def compute_entity_rates(
    history: Optional[pd.DataFrame], entity_col: str, prior_rate: float = 0.10, k: float = 20.0
) -> Dict[str, float]:
    """
    Calcule le taux de victoire lissé de chaque jockey/entraîneur à partir d'un
    historique de courses TERMINÉES (colonne finish_position renseignée).

    Renvoie un dict {nom_normalisé: taux}. Vide si l'historique est absent.
    """
    if history is None or entity_col not in history.columns or "finish_position" not in history.columns:
        return {}
    df = history[[entity_col, "finish_position"]].dropna()
    if df.empty:
        return {}
    df = df.assign(
        _key=df[entity_col].astype(str).str.strip().str.lower(),
        _win=(pd.to_numeric(df["finish_position"], errors="coerce") == 1).astype(float),
    )
    grouped = df.groupby("_key")["_win"].agg(["sum", "count"])
    return {
        key: shrunk_rate(row["sum"], row["count"], prior_rate, k) for key, row in grouped.iterrows()
    }


def _norm_key(value: object) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).strip().lower()


def _num_col(df: pd.DataFrame, name: str) -> pd.Series:
    """Colonne numérique float64 (NaN si absente ou invalide), toujours alignée sur df.index."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").astype("float64")
    return pd.Series(np.nan, index=df.index, dtype="float64")


# --------------------------------------------------------------------------- #
# Probabilités implicites du marché
# --------------------------------------------------------------------------- #

def implied_probabilities(odds: pd.Series) -> pd.Series:
    """
    Convertit des cotes décimales en probabilités implicites NORMALISÉES au sein
    d'une course (on retire la marge du bookmaker / le prélèvement du pari mutuel).

    Les cotes manquantes reçoivent la probabilité moyenne des cotes connues, ce qui
    est neutre : elles n'avantagent ni ne désavantagent le cheval.
    """
    o = pd.to_numeric(odds, errors="coerce")
    o = o.where(o > 1.0)  # une cote <= 1 est invalide
    raw = 1.0 / o
    if raw.notna().sum() == 0:
        return pd.Series(1.0 / len(o), index=o.index)
    raw = raw.fillna(raw.mean())
    return raw / raw.sum()


# --------------------------------------------------------------------------- #
# Construction du tableau de features
# --------------------------------------------------------------------------- #

FEATURE_COLUMNS = [
    "log_odds_implied",
    "form_score",
    "form_win_rate",
    "form_place_rate",
    "dq_rate",
    "career_win_rate",
    "career_place_rate",
    "log_earnings_per_start",
    "freshness",
    "jockey_win_rate",
    "trainer_win_rate",
    "draw_rel",
    "weight_rel",
    "age_rel",
]

# Libellés lisibles pour l'interface
FEATURE_LABELS = {
    "log_odds_implied": "Cote (probabilité du marché)",
    "form_score": "Forme récente (musique)",
    "form_win_rate": "Victoires récentes",
    "form_place_rate": "Places récentes",
    "dq_rate": "Incidents récents (D/T/A)",
    "career_win_rate": "Taux de victoire carrière",
    "career_place_rate": "Taux de place carrière",
    "log_earnings_per_start": "Gains par course",
    "freshness": "Fraîcheur (jours depuis dernière course)",
    "jockey_win_rate": "Réussite du jockey/driver",
    "trainer_win_rate": "Réussite de l'entraîneur",
    "draw_rel": "Position à la corde",
    "weight_rel": "Poids porté (relatif)",
    "age_rel": "Âge (relatif)",
}


def _freshness(days: pd.Series) -> pd.Series:
    """
    Transforme les jours depuis la dernière course en score de fraîcheur.
    Empiriquement, un délai de 15-45 jours est optimal ; un délai très court
    (< 7 j) ou très long (> 120 j, retour de repos) est pénalisant.
    Score dans [0,1], 1 = optimal. Inconnu -> 0.6 (neutre).
    """
    d = pd.to_numeric(days, errors="coerce").astype("float64")
    score = np.exp(-((np.log1p(d.clip(lower=0)) - np.log1p(28)) ** 2) / (2 * 0.9**2))
    return pd.Series(score, index=days.index).fillna(0.6)


def build_features(
    runners: pd.DataFrame,
    history: Optional[pd.DataFrame] = None,
    jockey_rates: Optional[Dict[str, float]] = None,
    trainer_rates: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Construit le tableau de features brutes (non standardisées) pour chaque partant.

    Paramètres
    ----------
    runners       : DataFrame au schéma `horseproba.schema` (une ligne par partant)
    history       : historique optionnel pour calculer les taux jockey/entraîneur
    jockey_rates  : dict pré-calculé (prioritaire sur history)
    trainer_rates : idem

    Retour : copie de `runners` enrichie des colonnes FEATURE_COLUMNS.
    """
    df = runners.copy()
    if "race_id" not in df.columns:
        df["race_id"] = "race_1"

    # --- Marché -----------------------------------------------------------
    odds = _num_col(df, "odds")
    implied = odds.groupby(df["race_id"]).transform(lambda s: implied_probabilities(s))
    df["log_odds_implied"] = np.log(implied.clip(lower=1e-4))

    # --- Forme récente ----------------------------------------------------
    musique = df["musique"] if "musique" in df.columns else pd.Series([None] * len(df), index=df.index)
    forms = [parse_musique(m) for m in musique]
    df["form_score"] = [f.score for f in forms]
    df["form_win_rate"] = [f.wins / f.n_runs if f.n_runs else 0.15 for f in forms]
    df["form_place_rate"] = [f.places / f.n_runs if f.n_runs else 0.35 for f in forms]
    df["dq_rate"] = [f.incidents / f.n_runs if f.n_runs else 0.05 for f in forms]

    # --- Carrière ---------------------------------------------------------
    starts = _num_col(df, "career_starts")
    wins = _num_col(df, "career_wins")
    places = _num_col(df, "career_places")
    earnings = _num_col(df, "earnings")
    df["career_win_rate"] = [shrunk_rate(w, s, 0.12, 8) for w, s in zip(wins, starts)]
    df["career_place_rate"] = [shrunk_rate(p, s, 0.33, 8) for p, s in zip(places, starts)]
    eps = np.log1p((earnings.fillna(0) / (starts.fillna(0) + 1)).clip(lower=0))
    # si gains inconnus pour tout le monde, la variable est constante => neutre
    df["log_earnings_per_start"] = eps.fillna(eps.median() if eps.notna().any() else 0.0)

    # --- Fraîcheur --------------------------------------------------------
    df["freshness"] = _freshness(_num_col(df, "days_since_last_run"))

    # --- Jockey / entraîneur ---------------------------------------------
    if jockey_rates is None:
        jockey_rates = compute_entity_rates(history, "jockey")
    if trainer_rates is None:
        trainer_rates = compute_entity_rates(history, "trainer")
    prior = 0.10
    jk = df["jockey"] if "jockey" in df.columns else pd.Series("", index=df.index)
    tr = df["trainer"] if "trainer" in df.columns else pd.Series("", index=df.index)
    df["jockey_win_rate"] = [jockey_rates.get(_norm_key(j), prior) for j in jk]
    df["trainer_win_rate"] = [trainer_rates.get(_norm_key(t), prior) for t in tr]

    # --- Variables relatives à la course ----------------------------------
    g = df.groupby("race_id")
    df["_draw"] = _num_col(df, "draw")
    dmin, dmax = g["_draw"].transform("min"), g["_draw"].transform("max")
    df["draw_rel"] = ((df["_draw"] - dmin) / (dmax - dmin).replace(0, np.nan)).fillna(0.5)
    # la corde n'a de sens qu'en plat
    if "discipline" in df.columns:
        not_flat = df["discipline"].astype(str).str.lower().ne("plat")
        df.loc[not_flat, "draw_rel"] = 0.5

    df["_w"] = _num_col(df, "weight_kg")
    df["weight_rel"] = (df["_w"] - g["_w"].transform("mean")).fillna(0.0)

    df["_age"] = _num_col(df, "age")
    df["age_rel"] = (df["_age"] - g["_age"].transform("mean")).fillna(0.0)

    df = df.drop(columns=["_draw", "_w", "_age"])
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(float).fillna(0.0)
    return df


def within_race_standardize(df: pd.DataFrame, columns: Iterable[str] = FEATURE_COLUMNS) -> pd.DataFrame:
    """
    Centre chaque variable par course (soustraction de la moyenne de la course).

    Justification : dans un modèle logit conditionnel, seule la DIFFÉRENCE de
    features entre partants d'une même course influence les probabilités ; centrer
    rend les coefficients plus stables et interprétables. On divise ensuite par
    l'écart-type GLOBAL de la variable (et non par course) pour conserver
    l'information « cette course est très homogène / très hétérogène ».
    """
    out = df.copy()
    cols = list(columns)
    centered = out[cols] - out.groupby("race_id")[cols].transform("mean")
    global_std = centered.std(ddof=0).replace(0, 1.0).fillna(1.0)
    out[cols] = (centered / global_std).fillna(0.0)
    return out
