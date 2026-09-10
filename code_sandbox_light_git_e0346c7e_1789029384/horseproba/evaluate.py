"""
Évaluation de la qualité des probabilités et backtest temporel.

Pourquoi évaluer des PROBABILITÉS et pas seulement le « taux de gagnants trouvés » ?
----------------------------------------------------------------------------------
Le but du modèle n'est pas de désigner un gagnant (impossible de façon fiable)
mais d'estimer correctement des probabilités. Les métriques adaptées sont :

- Log-loss (score logarithmique) sur le gagnant : −(1/N) Σ log p(gagnant).
  Comparé au log-loss du marché et du modèle uniforme (log n).
- Brier score sur la victoire : moyenne de (p_i − y_i)².
- Calibration : parmi les chevaux estimés à ~20 %, environ 20 % gagnent-ils ?
- Taux de réussite du favori du modèle (hit rate top-1) et top-3.
- ROI simulé d'une stratégie naïve « miser 1 € sur chaque cheval où
  p_modèle > p_marché + marge » — fourni À TITRE ILLUSTRATIF uniquement : le
  passé ne garantit rien et les cotes finales diffèrent des cotes probables.

Le backtest est TEMPOREL (walk-forward) : on entraîne sur les courses passées,
on prédit les suivantes, on avance. Cela évite la fuite d'information (utiliser
le futur pour prédire le passé), erreur classique qui gonfle artificiellement
les performances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .model import ConditionalLogitModel


@dataclass
class EvalReport:
    n_races: int
    logloss_model: float
    logloss_market: Optional[float]
    logloss_uniform: float
    brier_model: float
    brier_market: Optional[float]
    top1_hit_rate: float
    top3_hit_rate: float  # gagnant réel parmi les 3 premiers du modèle
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    roi_value_bets: Optional[float] = None
    n_value_bets: int = 0
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)

    def as_table(self) -> pd.DataFrame:
        rows = [
            ("Courses évaluées", f"{self.n_races}"),
            ("Log-loss modèle (↓ mieux)", f"{self.logloss_model:.4f}"),
            ("Log-loss marché", "n/d" if self.logloss_market is None else f"{self.logloss_market:.4f}"),
            ("Log-loss uniforme (1/n)", f"{self.logloss_uniform:.4f}"),
            ("Brier modèle (↓ mieux)", f"{self.brier_model:.4f}"),
            ("Brier marché", "n/d" if self.brier_market is None else f"{self.brier_market:.4f}"),
            ("Favori du modèle gagnant", f"{100 * self.top1_hit_rate:.1f} %"),
            ("Gagnant dans le top 3 du modèle", f"{100 * self.top3_hit_rate:.1f} %"),
            ("ROI simulé « value bets » (illustratif)", "n/d" if self.roi_value_bets is None else f"{100 * self.roi_value_bets:+.1f} % sur {self.n_value_bets} mises"),
        ]
        return pd.DataFrame(rows, columns=["Métrique", "Valeur"])


def _race_metrics(pred: pd.DataFrame) -> EvalReport:
    """Calcule les métriques sur un DataFrame de prédictions contenant finish_position."""
    pred = pred.copy()
    pred["is_win"] = (pd.to_numeric(pred["finish_position"], errors="coerce") == 1).astype(float)
    groups = pred.groupby("race_id", sort=False)
    valid = groups["is_win"].transform("sum") == 1
    pred = pred[valid]
    if pred.empty:
        return EvalReport(0, np.nan, None, np.nan, np.nan, None, np.nan, np.nan)

    winners = pred[pred["is_win"] == 1]
    n = len(winners)
    ll_model = -np.log(winners["p_win"].clip(lower=1e-9)).mean()
    sizes = pred.groupby("race_id")["horse"].transform("count")
    ll_unif = np.log(sizes[pred["is_win"] == 1]).mean()

    has_market = pred["market_p"].notna().all() if "market_p" in pred else False
    ll_market = -np.log(winners["market_p"].clip(lower=1e-9)).mean() if has_market else None

    brier_model = ((pred["p_win"] - pred["is_win"]) ** 2).mean()
    brier_market = ((pred["market_p"] - pred["is_win"]) ** 2).mean() if has_market else None

    top1 = (winners["rank"] == 1).mean()
    top3 = (winners["rank"] <= 3).mean()

    # Calibration par déciles de probabilité prédite
    bins = pd.cut(pred["p_win"], bins=[0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 1.0], include_lowest=True)
    calib = pred.groupby(bins, observed=True).agg(
        predite=("p_win", "mean"), observee=("is_win", "mean"), effectif=("is_win", "size")
    ).reset_index().rename(columns={"p_win": "tranche"})
    calib["tranche"] = calib["tranche"].astype(str)

    # ROI illustratif : miser 1 € si p_modèle > p_marché × 1.15 (marge 15 %)
    roi, n_bets = None, 0
    if has_market and "odds" in pred:
        odds = pd.to_numeric(pred["odds"], errors="coerce")
        bets = pred[(pred["p_win"] > pred["market_p"] * 1.15) & odds.notna()]
        n_bets = len(bets)
        if n_bets:
            returns = (bets["is_win"] * odds.loc[bets.index]).sum()
            roi = float((returns - n_bets) / n_bets)

    return EvalReport(
        n_races=n,
        logloss_model=float(ll_model),
        logloss_market=None if ll_market is None else float(ll_market),
        logloss_uniform=float(ll_unif),
        brier_model=float(brier_model),
        brier_market=None if brier_market is None else float(brier_market),
        top1_hit_rate=float(top1),
        top3_hit_rate=float(top3),
        calibration=calib,
        roi_value_bets=roi,
        n_value_bets=n_bets,
        predictions=pred,
    )


def backtest(
    history: pd.DataFrame,
    l2: float = 1.0,
    initial_train_frac: float = 0.5,
    n_folds: int = 4,
) -> EvalReport:
    """
    Backtest walk-forward :
    - trie les courses par date (ou par ordre d'apparition si pas de date) ;
    - entraîne sur les premières `initial_train_frac` courses ;
    - prédit le bloc suivant, ré-entraîne en incluant ce bloc, etc. (`n_folds` blocs).

    Renvoie un EvalReport agrégé sur toutes les prédictions hors-échantillon.
    """
    hist = history.copy()
    if "race_date" in hist.columns:
        hist["_d"] = pd.to_datetime(hist["race_date"], errors="coerce")
        order = hist.groupby("race_id")["_d"].min().sort_values(kind="stable")
    else:
        order = pd.Series(range(hist["race_id"].nunique()), index=hist["race_id"].unique())
    race_ids: List[str] = list(order.index)
    n = len(race_ids)
    if n < 10:
        raise ValueError("Au moins 10 courses terminées sont nécessaires pour un backtest.")

    start = max(5, int(n * initial_train_frac))
    fold_edges = np.linspace(start, n, n_folds + 1).astype(int)
    preds: List[pd.DataFrame] = []
    for a, b in zip(fold_edges[:-1], fold_edges[1:]):
        if b <= a:
            continue
        train_ids, test_ids = race_ids[:a], race_ids[a:b]
        train = hist[hist["race_id"].isin(train_ids)]
        test = hist[hist["race_id"].isin(test_ids)]
        model = ConditionalLogitModel(l2=l2)
        model.fit(train)
        preds.append(model.predict(test))
    if not preds:
        raise ValueError("Backtest impossible : pas assez de courses pour former des blocs de test.")
    return _race_metrics(pd.concat(preds, ignore_index=True))


def evaluate_predictions(pred: pd.DataFrame) -> EvalReport:
    """Évalue des prédictions déjà calculées (in-sample ou externes) comportant finish_position."""
    return _race_metrics(pred)
