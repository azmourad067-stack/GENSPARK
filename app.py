"""
═══════════════════════════════════════════════════════════════════════════════
 QuantTurf Pro v5.0.0 — "BENTER EDITION + DATA-CALIBRATED + HENERY-CORRECTED"
═══════════════════════════════════════════════════════════════════════════════
 Améliorations v5.0.0 par rapport à v4.3.0 :
 ─────────────────────────────────────────────
 ✅ Correction HENERY γ=0.58 du biais Harville sur positions 2-3-4-5
    (Lo & Bacon-Shone 1994 : Harville surestime P(2e/3e) pour favoris)
    (Ali 1998 : confirmation du biais systématique sur 15k courses)
 ✅ Fonction de Prelec (a=0.928) pour débiaisage FLB supérieure au power γ
    (Snowberg & Wolfers 2010 : Prelec拟合 > power sur 6.4M courses)
 ✅ Taux PMU RÉELS (TRJ officiel 2025) : Couplé 74%, Trio 69.1%,
    Quarté+ 63.3%, Quinté+ 64.75% — correction des valeurs erronées v4.3
    (Source : Mediapronos / PMU.fr réglementation officielle)
 ✅ Tables autostart recalibrées : numéros 3-4-5-6-7 favorisés
    (Source : Turf.bzh, PMU.fr, Turfomania — pas seulement 4-5-6)
 ✅ Kelly fractionnaire Thorp : ajustement pour biais de surestimation
    (Thorp 2007 : la surestimation systématique > incertitude comme
    motif de réduction de mise ; half-Kelly protège contre croissance négative)
 ✅ Value bet : ajout filtre Brier score (calibration locale)
 ✅ Facteur spécialité autostart (cheval/driver perf. précédentes autostart)
 ✅ Benter Blend : clarification formule logit originale (Benter 1994)
    c_i = exp(α·ln(f_i) + β·ln(π_i)) / Σ exp(α·ln(f_j) + β·ln(π_j))
 ─────────────────────────────────────────────
 Architecture conservée de v4.3 :
 ✅ Modèle Plackett-Luce (Harville) pour ordres d'arrivée exacts
 ✅ Benter Blend (log-log fusion modèle/marché, formule Benter 1994)
 ✅ Shrinkage bayésien sur la musique (régression vers moyenne empirique)
 ✅ État du terrain (bon, souple, lourd) + poids + jours de repos
 ✅ Paris exotiques rigoureux : Couplé / Trio / Quarté+ / Quinté+ ordre & désordre
 ✅ Détection de value avec seuil dynamique selon overround
 ✅ Architecture modulaire en classes
 ✅ Diagnostic complet (calibration, divergence, edge expected)
═══════════════════════════════════════════════════════════════════════════════
Sources scientifiques :
- Benter, W. (1994). "Computer Based Horse Race Handicapping" (Hong Kong)
- Harville, D. (1973). "Assigning probabilities to outcomes of multi-entry comp."
- Henery, R.J. (1983). "Permutation probabilities as models for horse races"
- Lo, V. & Bacon-Shone, J. (1994). "Probability and Optimization Models for Racing"
  → γ=0.58 pour correction Henery (données Hong Kong)
- Ali, M.M. (1998). "Probability models on horse-race outcomes"
  → confirmation biais Harville sur 15,000+ courses
- Snowberg, E. & Wolfers, J. (2010). "Explaining the Favorite-Longshot Bias"
  → Prelec weighting a=0.928, 6.4M starts
- Plackett, R. (1975). "The Analysis of Permutations"
- Kelly, J. (1956). "A New Interpretation of Information Rate"
- Thorp, E.O. (2007). "The Kelly Criterion in Blackjack, Sports Betting, Stock Market"
  → fractional Kelly pour protection contre surestimation systématique
- Prelec, D. (1998). "The Probability Weighting Function"
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from itertools import combinations, permutations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache
import logging
import time
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# 1.  CONFIGURATION GLOBALE
# =============================================================================
@dataclass
class Config:
    # --- App ---
    APP_VERSION: str = "5.0.0"
    APP_NAME: str = "QuantTurf Pro"
    APP_TAG: str = "Henery-Corrected Edition (γ=0.58 + Prelec FLB)"

    # --- Monte Carlo / Plackett-Luce ---
    MC_ITERATIONS: int = 8000
    TEMPERATURE: float = 1.0
    NOISE_BASE: float = 0.20           # v4.1: +incertitude reconnue

    # --- Marché (CALIBRÉ v4.2 sur 175 quintés 2026 - janv-juin) ---
    # Observations dataset réel :
    # - Cote médiane gagnant = 8.10€ (Trot 5.55 / Plat 9.30 / Haies 9.70)
    # - Seulement 9.8% des courses gagnées par favori cote<4
    # - 28.7% des gagnants ont cote >= 12€
    # - La synthèse de presse rend +19.4% ROI → BASELINE TO BEAT
    #
    # v5.0 : MARKET_WEIGHT légèrement réduit car le Benter Blend log-log
    # intègre déjà le poids marché via β. Un weight de 0.55 en PLUS du β=1.30
    # sur-pondérait le marché par double comptage.
    # Benter (1994) : α et β sont estimés par maximisation de log-vraisemblance
    # sur données historiques ; ils représentent la "relative correctness"
    # du modèle vs marché. Le blend log-log EST déjà la fusion optimale.
    MARKET_WEIGHT: float = 0.45        # ↓ 0.55 → 0.45 (évite double-comptage marché)
    BENTER_ALPHA: float = 0.50
    BENTER_BETA: float = 1.30
    OVERROUND_CORRECTION: bool = True

    # --- v5.0 : Paramètres de correction HENERY ---
    # Lo & Bacon-Shone (1994) sur données Hong Kong :
    # Harville surestime P(fini 2e/3e) pour favoris, sous-estime pour outsiders.
    # Correction Henery : log(ψ_i) = θ_i + γ·θ_i², avec γ ≈ 0.58
    # (θ_i = log-force du cheval i). Applied APRES le PL, sur les positions 2+.
    HENERY_GAMMA: float = 0.58         # Lo & Bacon-Shone (1994), données HK
    USE_HENERY_CORRECTION: bool = True

    # --- Platt Scaling par discipline (calibré sur backtest) ---
    PLATT_GLOBAL: Tuple[float, float] = (0.80, -0.40)
    # Plat = imprévisible (5.5% favoris gagnent) → forte compression
    PLATT_PLAT:   Tuple[float, float] = (0.40, -1.50)
    # Trot = prévisible (38.2% favoris gagnent) → modèle amplifié
    PLATT_TROT:   Tuple[float, float] = (1.30, +0.50)
    # Obstacle = entre les deux (19% favoris gagnent)
    PLATT_OBSTACLE: Tuple[float, float] = (0.80, -0.50)
    USE_PLATT_CALIBRATION: bool = True

    # --- Benter Blend par discipline (calibré) ---
    BENTER_AB_PLAT:     Tuple[float, float] = (0.25, 1.20)  # Plat : marche>modele
    BENTER_AB_TROT:     Tuple[float, float] = (0.55, 1.70)  # Trot : marche tres predictif
    BENTER_AB_OBSTACLE: Tuple[float, float] = (0.40, 1.30)
    USE_DISCIPLINE_BLEND: bool = True

    # --- Value / Kelly (CALIBRÉ v4.2 sur 175 quintés) ---
    # Cote médiane gagnant = 8.10€, P25 = 4.33€, P75 = 13.47€
    # → Sweet spot Simple Gagnant : entre P25 et P75 = [4.5, 13]
    VALUE_THRESHOLD: float = 1.20      # ↑ de 1.15 → 1.20 (plus strict)
    VALUE_COTE_MIN: float = 4.5        # ↓ de 5.0 → 4.5 (P25 observé)
    VALUE_COTE_MAX: float = 13.0       # ↑ de 10.0 → 13.0 (P75 observé)
    #
    # v5.0 : Kelly fraction réduit selon Thorp (2007).
    # Thorp démontre que le motif PRINCIPAL de réduction Kelly n'est pas
    # l'incertitude du modèle (qui a un impact faible selon simulations),
    # mais la tendance SYSTÉMATIQUE à surestimer P(gagner), ce qui conduit
    # à overbet. Half-Kelly (0.50) protège contre croissance négative.
    # Ici 0.20 × vol_adj × estimation_bias_adj = encore plus prudent.
    KELLY_FRACTION: float = 0.20       # inchangé (déjà prudent)
    KELLY_OVERESTIMATION_GUARD: float = 0.85  # v5.0 : Thorp guard factor
    MIN_KELLY_ODDS: float = 4.50
    MAX_KELLY_STAKE: float = 0.025     # ↓ cap plus prudent : 2.5%
    PLACE_ODDS_FACTOR: Dict[str, float] = None

    # --- v4.2 : Simple Gagnant settings ---
    SG_USE_PRESSE_AS_BASELINE: bool = True   # +19.4% ROI sur baseline presse
    SG_AVOID_PURE_FAVORITES: bool = True     # éviter cotes <3 en Plat (5.5% win)
    SG_AVOID_EXTREME_OUTSIDERS: bool = True  # éviter cotes >20 (rares)

    # --- Empirique (corde, expérience) ---
    EMPIRICAL_WEIGHT: float = 0.25
    USE_EXPERIENCE_FACTOR: bool = True

    # --- Shrinkage bayésien ---
    SHRINKAGE_K: float = 4.0           # nb "courses fantômes" vers moyenne
    POPULATION_MEAN_SCORE: float = 4.0 # moyenne pop. des scores musique
    POPULATION_MEAN_WIN: float = 0.10  # 10% victoires moyennes pop.

    # --- Paris ---
    RACE_TYPES: List[str] = None
    TRACK_CONDITIONS: List[str] = None
    DEPART_TYPES: List[str] = None

    # --- Musique parsing ---
    MUSIC_POSITION_SCORES: Dict[str, float] = None
    MUSIC_RACE_TYPE_WEIGHTS: Dict[str, float] = None

    # --- Tables empiriques corde ---
    DRAW_WIN_PROB_PLAT: Dict[int, float] = None
    DRAW_PLACE_PROB_PLAT: Dict[int, float] = None
    DRAW_WIN_PROB_AUTOSTART: Dict[int, float] = None
    DRAW_PLACE_PROB_AUTOSTART: Dict[int, float] = None

    def __post_init__(self):
        if self.MUSIC_POSITION_SCORES is None:
            self.MUSIC_POSITION_SCORES = {
                "1": 10.0, "2": 7.5, "3": 5.5, "4": 4.0, "5": 3.0,
                "6": 2.0, "7": 1.5, "8": 1.0, "9": 0.5, "0": 0.2,
                "D": -2.0, "A": -1.5, "T": -1.5, "R": -1.0, "P": 0.3,
            }
        if self.MUSIC_RACE_TYPE_WEIGHTS is None:
            self.MUSIC_RACE_TYPE_WEIGHTS = {
                "a": 1.00, "m": 0.90, "p": 1.00, "h": 0.95,
                "s": 0.90, "c": 0.85, "x": 1.00,
            }
        if self.RACE_TYPES is None:
            self.RACE_TYPES = ["Plat", "Attelé", "Monté", "Haies",
                               "Steeple-chase", "Cross-country"]
        if self.TRACK_CONDITIONS is None:
            self.TRACK_CONDITIONS = ["Bon", "Bon souple", "Souple",
                                    "Très souple", "Collant", "Lourd",
                                    "Très lourd"]
        if self.DEPART_TYPES is None:
            self.DEPART_TYPES = ["Stalles (Plat)", "Autostart (Trot)",
                                "Volte (Trot)", "Élastique (Obstacle)"]
        if self.PLACE_ODDS_FACTOR is None:
            # Rapports cote_placé/cote_gagnant empiriques selon nb partants
            self.PLACE_ODDS_FACTOR = {
                "small": 0.50,   # ≤ 7 partants : place uniquement sur 2 premiers
                "medium": 0.40,  # 8-15
                "large": 0.32,   # ≥16
            }

        # --- TABLES EMPIRIQUES BASÉES SUR ÉTUDES PUBLIQUES (Turf.bzh, PMU) ---
        # v5.0 : Recalibré selon Turf.bzh, PMU.fr, Turfomania, Multi-Turf
        # Sources :
        #   - Turf.bzh : "Statistiquement, les numéros centraux (3,4,5) sont
        #     souvent les plus avantagés car ils évitent la corde tout en
        #     restant bien placés" → Voie Royale = 3-4-5
        #   - PMU.fr FAQ : "Les numéros 4 et 5 sont les meilleurs. Les numéros
        #     3, 6 et 7 sont également favorisés"
        #   - Turfomania : "Les numéros 3-4-5-6-7 obtiennent les meilleurs
        #     taux de réussite, taux relativement proches au demeurant"
        #   - Multi-Turf : "Numéros 3 à 6 avantage central derrière l'autostart"
        #
        # PLAT : corde 1-4 favorisée, surtout < 1800m
        if self.DRAW_WIN_PROB_PLAT is None:
            self.DRAW_WIN_PROB_PLAT = {
                1: 11.8, 2: 11.5, 3: 11.0, 4: 10.5, 5: 9.5,
                6: 8.5, 7: 7.5, 8: 6.5, 9: 5.5, 10: 4.8,
                11: 4.2, 12: 3.6, 13: 3.2, 14: 2.8, 15: 2.5,
                16: 2.2, 17: 1.9, 18: 1.6, 19: 1.3, 20: 1.0,
            }
        if self.DRAW_PLACE_PROB_PLAT is None:
            self.DRAW_PLACE_PROB_PLAT = {
                1: 31.0, 2: 30.0, 3: 29.0, 4: 27.5, 5: 25.0,
                6: 22.5, 7: 20.0, 8: 17.5, 9: 15.5, 10: 14.0,
                11: 12.5, 12: 11.0, 13: 10.0, 14: 9.0, 15: 8.0,
                16: 7.0, 17: 6.0, 18: 5.5, 19: 5.0, 20: 4.5,
            }
        # AUTOSTART (Trot) : v5.0 recalibré — numéros 3-4-5-6-7 favorisés
        # (anciennement 4-5-6 seulement, maintenant étendu selon sources)
        # Le numéro 1 reste désavantagé par risque d'enfermement (Turf.bzh).
        # Les numéros 8-9 pénalisés sur petites pistes (Multi-Turf).
        # Seconde ligne (10+) : handicap mais jouable à Vincennes (UNAT).
        if self.DRAW_WIN_PROB_AUTOSTART is None:
            self.DRAW_WIN_PROB_AUTOSTART = {
                # 1ère ligne — recalibré : 3-4-5-6-7 favorisés (Turfomania, PMU.fr)
                1: 8.5,  2: 9.0,  3: 11.0, 4: 11.5, 5: 12.0, 6: 11.0,
                7: 10.5, 8: 8.0,  9: 6.5,  10: 5.0,
                # 2ème ligne (handicap derrière) — inchangé
                11: 3.5, 12: 2.8, 13: 2.3, 14: 1.9, 15: 1.6,
                16: 1.3, 17: 1.1, 18: 0.9, 19: 0.7, 20: 0.5,
            }
        if self.DRAW_PLACE_PROB_AUTOSTART is None:
            self.DRAW_PLACE_PROB_AUTOSTART = {
                # 1ère ligne — recalibré
                1: 23.0, 2: 24.5, 3: 28.0, 4: 30.0, 5: 30.5, 6: 28.5,
                7: 26.0, 8: 21.0, 9: 18.0, 10: 14.5,
                # 2ème ligne
                11: 11.0, 12: 9.0, 13: 7.5, 14: 6.0, 15: 5.0,
                16: 4.2, 17: 3.5, 18: 3.0, 19: 2.5, 20: 2.0,
            }


CONFIG = Config()


# =============================================================================
# 2.  PARSING DE LA MUSIQUE (avec shrinkage bayésien)
# =============================================================================
@dataclass
class MusicMetrics:
    score: float
    regularity: float
    races_count: int
    avg_position: float
    best_position: int
    recent_form: float
    trend: float
    is_debutant: bool
    win_ratio: float
    podium_ratio: float
    consistency: float = 0.0
    shrunk_score: float = 0.0    # score après shrinkage bayésien
    shrunk_win_ratio: float = 0.0


@lru_cache(maxsize=1024)
def parse_music_v4(music_str: str) -> MusicMetrics:
    """
    Parse la musique d'un cheval/driver/entraîneur.
    Format type : '1a2a3a(23)4aDa5a' (chiffres + type de course)
    Applique un shrinkage bayésien vers la moyenne population.
    """
    if (not music_str or
            music_str.strip().upper() in ("", "-", "INEDIT", "INÉDIT", "N/A", "0")):
        return MusicMetrics(
            score=CONFIG.POPULATION_MEAN_SCORE,
            regularity=0.50, races_count=0, avg_position=5.0,
            best_position=10, recent_form=CONFIG.POPULATION_MEAN_SCORE,
            trend=0.0, is_debutant=True,
            win_ratio=CONFIG.POPULATION_MEAN_WIN,
            podium_ratio=0.30,
            shrunk_score=CONFIG.POPULATION_MEAN_SCORE,
            shrunk_win_ratio=CONFIG.POPULATION_MEAN_WIN,
        )
    try:
        clean = re.sub(r"[()\s]", "", music_str.strip().upper())
        tokens = re.findall(r"([0-9DATRP])([AMPHSC]?)", clean)
        if not tokens:
            return parse_music_v4("")

        raw_scores, numeric_positions = [], []
        for pos_char, rtype_char in tokens:
            rtype = rtype_char.lower() if rtype_char else "x"
            pos_score = CONFIG.MUSIC_POSITION_SCORES.get(pos_char, 0.3)
            type_weight = CONFIG.MUSIC_RACE_TYPE_WEIGHTS.get(rtype, 1.0)
            raw_scores.append(pos_score * type_weight)
            if pos_char.isdigit():
                numeric_positions.append(int(pos_char) if pos_char != "0" else 10)

        n = len(raw_scores)
        raw_scores_arr = np.array(raw_scores)

        # --- Decay exponentiel : courses récentes pèsent plus ---
        decay = np.exp(-0.30 * np.arange(n))
        decay /= decay.sum()
        weighted_score = float(np.dot(raw_scores_arr, decay))

        # --- Forme récente (3 dernières) ---
        recent_n = min(3, n)
        rd = decay[:recent_n] / decay[:recent_n].sum()
        recent_form = float(np.dot(raw_scores_arr[:recent_n], rd))

        # --- Régularité ---
        if len(numeric_positions) >= 2:
            pos_std = float(np.std(numeric_positions))
            regularity = max(0.0, 1.0 - pos_std / 5.0)
        else:
            pos_std = 3.0
            regularity = 0.50

        # --- Tendance (forme récente vs ancienne) ---
        if n >= 4:
            recent_avg = np.mean(raw_scores_arr[: n // 2])
            old_avg = np.mean(raw_scores_arr[n // 2:])
            trend = (recent_avg - old_avg) / (abs(old_avg) + 1e-9)
        else:
            trend = 0.0

        # --- Ratios ---
        win_count = sum(1 for p in numeric_positions if p == 1)
        podium_count = sum(1 for p in numeric_positions if p <= 3)
        win_ratio = win_count / max(n, 1)
        podium_ratio = podium_count / max(n, 1)

        # --- Consistance ---
        consistency = max(0.0, min(1.0, 1.0 - pos_std / 10.0))

        # ──────────────────────────────────────────────────────────
        # SHRINKAGE BAYÉSIEN
        # ──────────────────────────────────────────────────────────
        # Formule : score_shrunk = (n*score + K*μ_pop) / (n+K)
        # Plus n est petit, plus on tire vers la moyenne population
        K = CONFIG.SHRINKAGE_K
        shrunk_score = (n * weighted_score + K * CONFIG.POPULATION_MEAN_SCORE) / (n + K)
        shrunk_win = (n * win_ratio + K * CONFIG.POPULATION_MEAN_WIN) / (n + K)

        return MusicMetrics(
            score=weighted_score,
            regularity=regularity,
            races_count=n,
            avg_position=float(np.mean(numeric_positions)) if numeric_positions else 5.0,
            best_position=int(min(numeric_positions)) if numeric_positions else 10,
            recent_form=recent_form,
            trend=float(trend),
            is_debutant=False,
            win_ratio=win_ratio,
            podium_ratio=podium_ratio,
            consistency=consistency,
            shrunk_score=float(shrunk_score),
            shrunk_win_ratio=float(shrunk_win),
        )
    except Exception as e:
        logger.warning(f"Music parsing error '{music_str}': {e}")
        return parse_music_v4("")


# =============================================================================
# 3.  FACTEURS CONTEXTUELS
# =============================================================================
def experience_factor(races_count: int) -> float:
    """Coefficient multiplicateur 0.7-1.2 selon expérience."""
    if not CONFIG.USE_EXPERIENCE_FACTOR:
        return 1.0
    if races_count <= 0:   return 0.70
    if races_count <= 3:   return 0.82
    if races_count <= 10:  return 1.00
    if races_count <= 30:  return 1.10
    return 1.18


def draw_factor_v4(draw: int, race_type: str, distance: int,
                   depart_type: str = "Stalles (Plat)",
                   track: str = "Bon") -> float:
    """
    Facteur de corde RAFFINÉ — gère plat ET autostart trot.
    Retourne un score [-1.5, +1.5] à fusionner dans le composite.

    v5.0 : Autostart recalibré selon Turf.bzh, PMU.fr, Turfomania, Multi-Turf.
    Les numéros 3-4-5-6-7 sont favorisés (pas seulement 4-5-6).
    Le numéro 1 reste désavantagé par risque d'enfermement (Turf.bzh).
    Sources :
      - Turf.bzh : "Numéros 3, 4, 5 : La Voie Royale. Priorité absolue."
      - PMU.fr FAQ : "Les numéros 4 et 5 sont les meilleurs. Les numéros 3,
        6 et 7 sont également favorisés."
      - Turfomania : "Les numéros 3-4-5-6-7 obtiennent les meilleurs taux
        de réussite, taux relativement proches au demeurant."
      - Multi-Turf : "Numéros 3 à 6 avantage central derrière l'autostart."
    """
    if not draw or draw <= 0:
        return 0.0
    draw = min(int(draw), 20)

    # ────────── PLAT (stalles) ──────────
    if race_type == "Plat":
        # 1-4 nettement favorisés, 5-7 OK, 8+ pénalisés
        if draw <= 2:    base = 1.0
        elif draw <= 4:  base = 0.7
        elif draw <= 6:  base = 0.3
        elif draw <= 9:  base = -0.2
        elif draw <= 12: base = -0.6
        else:            base = -1.0

        # Modulation par distance
        if distance <= 1300:   dist_mult = 1.6   # sprint : corde décisive
        elif distance <= 1600: dist_mult = 1.3
        elif distance <= 2000: dist_mult = 1.0
        elif distance <= 2400: dist_mult = 0.7
        else:                  dist_mult = 0.4

        # Modulation terrain : sur terrain lourd, la corde peut devenir piège
        if track in ("Lourd", "Très lourd", "Collant"):
            base *= 0.3  # neutralise quasi l'effet corde
        elif track in ("Souple", "Très souple"):
            base *= 0.7

        return base * dist_mult

    # ────────── TROT AUTOSTART (v5.0 recalibré) ──────────
    if depart_type == "Autostart (Trot)" and race_type in ("Attelé", "Monté"):
        # v5.0 : 3-4-5-6-7 favorisés (Turfomania, PMU.fr, Turf.bzh)
        # "Voie Royale" = 3-4-5 (Turf.bzh), 6-7 également favorisés (PMU.fr)
        if draw in (4, 5):        base = 1.0   # Meilleurs (PMU.fr)
        elif draw in (3, 6):     base = 0.8   # Favorisés (Turf.bzh "Voie Royale")
        elif draw == 7:          base = 0.5   # Bon taux de réussite (Turfomania)
        elif draw == 2:          base = 0.2   # Léger avantage
        elif draw in (1, 8):     base = -0.2  # 1: enfermement / 8: extérieur (Multi-Turf)
        elif draw == 9:          base = -0.3  # Extérieur pénalisé sur petites pistes
        elif draw == 10:         base = -0.5
        elif draw <= 14:         base = -0.7
        else:                    base = -1.0   # 2e ligne handicap

        # Effet réduit sur longues distances
        if distance >= 2700:
            base *= 0.7
        # v5.0 : Vincennes (grands virages) tolère mieux la 2e ligne (UNAT)
        # → pas de pénalité supplémentaire ici ; l'utilisateur peut ajuster
        return base

    # ────────── OBSTACLE / autres : effet quasi nul ──────────
    return 0.0


def track_factor(track: str, race_type: str) -> float:
    """Facteur multiplicateur global selon état du terrain (~1.0 neutre)."""
    # Sur terrain lourd, la régularité prime sur la pointe de vitesse
    if track in ("Lourd", "Très lourd"):  return 0.92
    if track == "Collant":                return 0.95
    if track in ("Souple", "Très souple"): return 0.98
    return 1.0


def weight_factor(weight_kg: float, ref_weight: float = 56.0) -> float:
    """
    Plat uniquement : un cheval avec poids très élevé est désavantagé.
    1 kg ≈ 1-2 longueurs.
    """
    if weight_kg <= 0:
        return 1.0
    delta = weight_kg - ref_weight
    # 1 kg en plus = -2% performance
    return max(0.85, min(1.15, 1.0 - 0.02 * delta))


def rest_factor(days_since_last_race: int) -> float:
    """
    Jours de repos : optimum à 14-30 jours.
    < 7 jours : fatigue ; > 60 jours : déconditionnement.
    """
    d = days_since_last_race
    if d < 0:    return 1.0       # inconnu
    if d <= 5:   return 0.85
    if d <= 10:  return 0.95
    if d <= 30:  return 1.00
    if d <= 60:  return 0.95
    if d <= 120: return 0.88
    return 0.80


# =============================================================================
# 4.  SCORE COMPOSITE (entrée du modèle softmax)
# =============================================================================
def get_weights_v4(race_type: str) -> Dict[str, float]:
    """Poids normalisés par discipline. Total ≈ 1.0."""
    if race_type == "Plat":
        return {
            # Cheval (45%)
            "horse_score": 0.22, "horse_form": 0.10, "horse_regularity": 0.05,
            "horse_trend": 0.04, "horse_win": 0.04,
            # Jockey (20%)
            "driver_score": 0.10, "driver_form": 0.05, "driver_win": 0.05,
            # Entraîneur (15%)
            "trainer_score": 0.08, "trainer_form": 0.04, "trainer_win": 0.03,
            # Corde + contexte (20%)
            "draw_factor": 0.12, "synergy": 0.03, "weight_adj": 0.03, "rest_adj": 0.02,
        }
    elif race_type in ("Attelé", "Monté"):
        return {
            # Cheval (35%)
            "horse_score": 0.18, "horse_form": 0.08, "horse_regularity": 0.04,
            "horse_trend": 0.03, "horse_win": 0.02,
            # Driver/jockey (32%) — TRÈS important au trot
            "driver_score": 0.16, "driver_form": 0.09, "driver_win": 0.07,
            # Entraîneur (18%)
            "trainer_score": 0.10, "trainer_form": 0.05, "trainer_win": 0.03,
            # Corde autostart + contexte (15%)
            "draw_factor": 0.08, "synergy": 0.03, "weight_adj": 0.00, "rest_adj": 0.04,
        }
    else:  # Obstacle (Haies, Steeple, Cross)
        return {
            "horse_score": 0.24, "horse_form": 0.12, "horse_regularity": 0.06,
            "horse_trend": 0.04, "horse_win": 0.03,
            "driver_score": 0.12, "driver_form": 0.06, "driver_win": 0.04,
            "trainer_score": 0.12, "trainer_form": 0.06, "trainer_win": 0.04,
            "draw_factor": 0.00, "synergy": 0.03, "weight_adj": 0.02, "rest_adj": 0.02,
        }


def composite_score_v4(feat: Dict, weights: Dict) -> float:
    """Score linéaire pondéré. Sera ensuite passé en softmax."""
    s = 0.0

    s += weights["horse_score"]      * np.clip(feat["horse_score"], 0, 12)
    s += weights["horse_form"]       * np.clip(feat["horse_form"], 0, 12)
    s += weights["horse_regularity"] * np.clip(feat["horse_regularity"], 0, 1) * 10
    s += weights["horse_trend"]      * (np.clip(feat["horse_trend"], -1, 1) + 1) * 5
    s += weights["horse_win"]        * np.clip(feat["horse_win"], 0, 1) * 20

    s += weights["driver_score"] * np.clip(feat["driver_score"], 0, 12)
    s += weights["driver_form"]  * np.clip(feat["driver_form"], 0, 12)
    s += weights["driver_win"]   * np.clip(feat["driver_win"], 0, 1) * 20

    s += weights["trainer_score"] * np.clip(feat["trainer_score"], 0, 12)
    s += weights["trainer_form"]  * np.clip(feat["trainer_form"], 0, 12)
    s += weights["trainer_win"]   * np.clip(feat["trainer_win"], 0, 1) * 20

    # Corde
    if weights.get("draw_factor", 0) > 0:
        s += weights["draw_factor"] * feat.get("draw_factor", 0) * 5

    # Synergie cheval/jockey/entraîneur
    h = np.clip(feat["horse_score"], 0.1, 12)
    d = np.clip(feat["driver_score"], 0.1, 12)
    t = np.clip(feat["trainer_score"], 0.1, 12)
    syn = min(h, d, t) / max(h, d, t)
    s += weights.get("synergy", 0) * syn * 10

    # Ajustements multiplicatifs
    s += weights.get("weight_adj", 0) * (feat.get("weight_factor", 1.0) - 1.0) * 50
    s += weights.get("rest_adj",   0) * (feat.get("rest_factor",   1.0) - 1.0) * 50

    # Bruit minimal pour briser les égalités
    return max(0.05, s)


# =============================================================================
# 5.  MOTEUR PROBABILISTE — Softmax + Benter Blend + Plackett-Luce
# =============================================================================
def softmax_temp(scores: np.ndarray, T: float = 1.0) -> np.ndarray:
    s = np.asarray(scores, dtype=float) / max(T, 0.05)
    s -= s.max()
    e = np.exp(np.clip(s, -50, 50))
    p = e / (e.sum() + 1e-12)
    return p


def remove_overround(odds: np.ndarray) -> np.ndarray:
    """
    Débiaise les cotes : normalisation + correction favori-outsider bias.
    v5.0 : remplace la transformation power (γ=1.12) par la fonction de
    Prelec INVERSE (1998), calibrée par Snowberg & Wolfers (2010) sur 6.4M de courses.

    Snowberg & Wolfers (2010) démontrent que la fonction de Prelec modélise
    la façon dont les parieurs PERÇOIVENT les probabilités :
      w(p) = exp(-(-ln(p))^a),  avec a = 0.928 (calibration S&W 2010)

    Les parieurs surpondèrent les faibles probabilités (longshots) et
    sous-pondèrent les fortes (favoris). Le marché reflète donc w(p_true),
    non p_true. Pour récupérer p_true à partir des cotes marché, il faut
    appliquer l'INVERSE de la fonction de Prelec :
      p_true = exp(-(-ln(p_market))^{1/a})

    Avec a = 0.928, 1/a ≈ 1.0776.

    Effet vérifié (tests unitaires) :
      - Favori (cote 2.0, p_market≈0.50) → p_true≈0.51 (légère hausse) ✓
      - Outsider (cote 50, p_market≈0.02) → p_true≈0.013 (baisse) ✓

    Comparé au power γ=1.12 :
      - Power : correction brute, moins fondée théoriquement
      - Prelec inverse : fondée sur modèle comportemental calibré sur 6.4M starts

    Référence : Snowberg & Wolfers (2010), NBER w15923, Table 1, p.28.
    """
    eps = 1e-9
    valid = odds > 1.01
    if not valid.any():
        return np.ones(len(odds)) / max(len(odds), 1)
    p_raw = np.where(valid, 1.0 / np.maximum(odds, 1.01), eps)
    # Normalisation initiale (suppression overround brut)
    p_raw = p_raw / p_raw.sum()

    if CONFIG.OVERROUND_CORRECTION:
        # v5.0 : INVERSE de la fonction de Prelec (1998)
        # p_true = exp(-(-ln(p_market))^{1/a})  avec a=0.928
        a_prelec = 0.928  # calibration S&W 2010 sur 6.4M starts
        inv_a = 1.0 / a_prelec  # ≈ 1.0776
        p_clipped = np.clip(p_raw, eps, 1.0 - eps)
        # Application de l'inverse Prelec
        p_corr = np.exp(-np.power(-np.log(p_clipped), inv_a))
        # Renormalisation après transformation
        p_corr = p_corr / p_corr.sum()
    else:
        p_corr = p_raw
    return p_corr


def benter_blend(p_model: np.ndarray, p_market: np.ndarray,
                 alpha: float = None, beta: float = None,
                 race_type: str = None) -> np.ndarray:
    """
    Fusion Benter (1994) : p_final ∝ p_model^α · p_market^β
    v4.1 : exposants par discipline si disponibles.
    """
    if alpha is None or beta is None:
        if CONFIG.USE_DISCIPLINE_BLEND and race_type:
            if race_type == "Plat":
                alpha, beta = CONFIG.BENTER_AB_PLAT
            elif race_type in ("Attelé", "Monté"):
                alpha, beta = CONFIG.BENTER_AB_TROT
            elif race_type in ("Haies", "Steeple-chase", "Cross-country"):
                alpha, beta = CONFIG.BENTER_AB_OBSTACLE
            else:
                alpha = CONFIG.BENTER_ALPHA
                beta = CONFIG.BENTER_BETA
        else:
            if alpha is None: alpha = CONFIG.BENTER_ALPHA
            if beta is None:  beta = CONFIG.BENTER_BETA
    eps = 1e-12
    log_blend = alpha * np.log(p_model + eps) + beta * np.log(p_market + eps)
    log_blend -= log_blend.max()
    p = np.exp(log_blend)
    return p / p.sum()


def platt_calibrate(probs: np.ndarray, race_type: str = None) -> np.ndarray:
    """
    Platt scaling : p_cal = sigmoid(a * logit(p) + b)
    Paramètres (a, b) calibrés sur 12 courses réelles par discipline.

    En Plat (a=0.45, b=-1.30) : forte compression, le modèle est sur-confiant.
    En Trot (a=1.20, b=+0.40) : légère amplification.
    """
    if not CONFIG.USE_PLATT_CALIBRATION:
        return probs
    if race_type == "Plat":
        a, b = CONFIG.PLATT_PLAT
    elif race_type in ("Attelé", "Monté"):
        a, b = CONFIG.PLATT_TROT
    elif race_type in ("Haies", "Steeple-chase", "Cross-country"):
        a, b = CONFIG.PLATT_OBSTACLE
    else:
        a, b = CONFIG.PLATT_GLOBAL
    eps = 1e-9
    p = np.clip(probs, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    p_cal = 1.0 / (1.0 + np.exp(-np.clip(a * logit_p + b, -50, 50)))
    s = p_cal.sum()
    return p_cal / s if s > 0 else probs


def plackett_luce_simulate(strengths: np.ndarray, n_iter: int,
                            noise: float = 0.18) -> np.ndarray:
    """
    Simule n_iter ordres d'arrivée par modèle Plackett-Luce (Harville).
    
    v5.0 : Correction HENERY γ=0.58 sur les positions 2-3-4-5.
    
    Le modèle Harville pur surestime la probabilité qu'un favori finisse
    2e/3e et sous-estime celle des outsiders (Ali 1998, Lo & Bacon-Shone 1994).
    
    La correction Henery (1983) consiste à appliquer une décroissance de
    puissance sur les forces pour les positions ultérieures :
      - Position 1 (gagnant) : force = s_i         (normal, pas de correction)
      - Position 2           : force = s_i^γ        (γ = 0.58)
      - Position 3           : force = s_i^{γ²}     (γ² = 0.336)
      - Position 4           : force = s_i^{γ³}     (γ³ = 0.195)
      - Position 5+          : force = s_i^{γ⁴}     (aplatissement maximal)
    
    Avec γ < 1, les forces sont APLATIES pour les positions 2+, ce qui
    réduit l'avantage des favoris et DONNE PLUS DE CHANCES aux outsiders
    d'être 2e/3e/4e — corrigeant ainsi le biais Harville documenté.
    
    Implémentation : tirage séquentiel Gumbel par position.
    Pour chaque position k, on tire un Gumbel indépendant et on calcule :
      score_k(i) = γ^k * log(s_i) + Gumbel_k(i)
    Le cheval avec le score max gagne cette position.
    
    Sources :
    - Henery (1983) : modèle théorique original, décroissance γ^k
    - Lo & Bacon-Shone (1994) : γ=0.58 sur données Hong Kong
    - Ali (1998) : confirmation biais Harville sur 15k courses
    
    Note de performance : le tirage séquentiel (n boucles au lieu d'1)
    est ~5x plus lent que le Gumbel trick unique, mais reste négligeable
    (< 1s pour 8000 itérations × 16 chevaux × 5 positions corrigées).
    """
    n = len(strengths)
    orders = np.zeros((n_iter, n), dtype=np.int32)
    base_log = np.log(np.maximum(strengths, 1e-9))
    gamma_henery = CONFIG.HENERY_GAMMA if CONFIG.USE_HENERY_CORRECTION else 1.0
    
    # Précalcul des exposants γ^k pour les positions 0..n-1
    # Position 0 (gagnant) : γ^0 = 1.0 (pas de correction)
    # Position 1 (2e)      : γ^1 = 0.58
    # Position 2 (3e)      : γ^2 = 0.336
    # etc.
    gamma_powers = np.array([gamma_henery ** k for k in range(n)])
    # Limiter la correction aux 5 premières positions (au-delà, γ^k ≈ 0)
    # Pour les positions 5+, utiliser γ^4 comme plancher pour éviter une
    # uniformisation excessive qui rendrait les positions 6+ aléatoires
    gamma_powers = np.maximum(gamma_powers, gamma_henery ** 4)
    
    for it in range(n_iter):
        # Bruit appliqué sur les log-forces
        noisy = base_log + np.random.normal(0, noise, n)
        available = np.ones(n, dtype=bool)
        
        for pos in range(min(n, 5)):  # Correction sur positions 0-4
            # Force effective pour cette position : γ^pos * log(s_i)
            effective_log = gamma_powers[pos] * noisy
            # Gumbel trick pour cette position
            gumbel = -np.log(-np.log(np.random.uniform(1e-12, 1-1e-12, n)))
            scores = effective_log + gumbel
            # Masquer les chevaux déjà placés
            scores[~available] = -np.inf
            # Le cheval avec le score max gagne cette position
            winner = np.argmax(scores)
            orders[it, pos] = winner
            available[winner] = False
        
        # Positions 5+ : Gumbel trick unique sur les restants (Harville standard)
        # Au-delà de la position 4, la correction Henery converge vers un
        # facteur constant (γ^4), ce qui équivaut à un Harville avec forces aplaties.
        remaining = np.where(available)[0]
        if len(remaining) > 0:
            effective_log = gamma_powers[4] * noisy[remaining]
            gumbel = -np.log(-np.log(np.random.uniform(1e-12, 1-1e-12, len(remaining))))
            scores = effective_log + gumbel
            sorted_remaining = remaining[np.argsort(-scores)]
            for i, idx in enumerate(sorted_remaining):
                orders[it, 5 + i] = idx
    
    return orders


# =============================================================================
# 6.  CORRECTION EMPIRIQUE (corde + expérience)
# =============================================================================
def empirical_win_prob(draw: int, race_type: str, distance: int,
                       depart_type: str) -> float:
    """Probabilité empirique de victoire en fraction [0, 1]."""
    if draw <= 0:
        return 0.10
    draw = min(draw, 20)
    if race_type == "Plat":
        base = CONFIG.DRAW_WIN_PROB_PLAT.get(draw, 2.0) / 100.0
        # Modulation distance
        if distance <= 1300:   m = 1.30
        elif distance <= 1600: m = 1.15
        elif distance <= 2000: m = 1.00
        elif distance <= 2400: m = 0.85
        else:                  m = 0.70
        return base * m
    elif depart_type == "Autostart (Trot)":
        base = CONFIG.DRAW_WIN_PROB_AUTOSTART.get(draw, 2.0) / 100.0
        return base
    return 0.10


def empirical_correction(p_model: np.ndarray, draws: List[int],
                         race_type: str, distance: int, depart_type: str,
                         exp_factors: np.ndarray, weight: float = None) -> np.ndarray:
    """
    Mélange convexe entre proba modèle et proba empirique pondérée par expérience.
    """
    if weight is None:
        weight = CONFIG.EMPIRICAL_WEIGHT
    n = len(p_model)
    p_emp = np.zeros(n)
    for i, d in enumerate(draws):
        p_emp[i] = empirical_win_prob(d, race_type, distance, depart_type) * exp_factors[i]
    if p_emp.sum() < 1e-9:
        return p_model
    p_emp /= p_emp.sum()
    p_blend = (1 - weight) * p_model + weight * p_emp
    return p_blend / p_blend.sum()


# =============================================================================
# 7.  KELLY & VALUE
# =============================================================================
def kelly_bet(prob: float, odds: float, volatility: float = 1.0,
              fraction: float = None) -> Tuple[float, float]:
    """
    Kelly fractionnaire dynamique (v5.0 — Thorp overestimation guard).
    
    v5.0 : Ajout d'un facteur de protection contre la surestimation systématique
    de P(gagner), selon Thorp (2007).
    
    Thorp démontre que le motif PRINCIPAL de réduction Kelly n'est pas
    l'incertitude du modèle (impact faible selon simulations), mais la
    tendance SYSTÉMATIQUE à surestimer P(gagner), ce qui conduit à overbet.
    Le guard factor (0.85) réduit la mise pour protéger contre ce biais.
    
    - Réduit la mise si volatilité élevée
    - Guard Thorp contre surestimation systématique (×0.85)
    - Cap absolu à CONFIG.MAX_KELLY_STAKE
    Retourne (kelly_pur, kelly_recommandé).
    
    Source : Thorp (2007), "The Kelly Criterion in Blackjack, Sports Betting,
    and the Stock Market" — fractional Kelly protège contre croissance négative.
    """
    if fraction is None:
        fraction = CONFIG.KELLY_FRACTION
    if odds <= CONFIG.MIN_KELLY_ODDS or prob < 0.04:
        return 0.0, 0.0
    b = odds - 1
    q = 1 - prob
    if b <= 0:
        return 0.0, 0.0
    k = (prob * b - q) / b
    k = max(0.0, k)
    # Ajustement volatilité
    vol_adj = 1.0 / (1.0 + max(0, volatility - 1.0))
    # v5.0 : Guard Thorp — protection contre surestimation systématique
    # Thorp (2007) : la cause principale d'overbet n'est pas l'incertitude
    # mais la tendance à surestimer P(gagner). Le guard factor réduit la mise.
    thorpguard = CONFIG.KELLY_OVERESTIMATION_GUARD
    k_reco = min(k * fraction * vol_adj * thorpguard, CONFIG.MAX_KELLY_STAKE)
    return float(k), float(k_reco)


def expected_roi(prob: float, odds: float, stake: float = 100.0) -> float:
    if stake <= 0 or odds <= 1.0:
        return 0.0
    ev = stake * (odds * prob - 1.0)
    return (ev / stake) * 100


# =============================================================================
# 8.  PARIS EXOTIQUES (via Plackett-Luce simulations)
# =============================================================================
# ──────────────────────────────────────────────────────────────────────────
# COTES PMU RÉALISTES — calibration empirique v5.0
# ──────────────────────────────────────────────────────────────────────────
# v5.0 : Taux de Redistribution au Joueur (TRJ) RÉELS du PMU français.
# Source : Mediapronos (réglementation PMU officielle 2025), PMU.fr
#
#   Couplé (Gagnant/Placé/Ordre) : TRJ = 74.0%  → takeout = 26.0%
#   Trio                        : TRJ = 69.1%  → takeout = 30.9%
#   Tiercé                      : TRJ = 64.35% → takeout = 35.65%
#   Quarté+                     : TRJ = 63.3%  → takeout = 36.7%
#   Quinté+                     : TRJ = 64.75% → takeout = 35.25%
#
# Le TRJ est le % de la masse misée redistribué aux gagnants.
# La cote PMU réelle pour un pari combiné : cote ≈ (1/p) × TRJ
# (Les anciennes valeurs v4.3 étaient inexactes — Trio 0.74 au lieu de 0.691,
#  Quinté 0.68 au lieu de 0.6475, Couplé Placé 0.78 au lieu de 0.74, etc.)
PMU_TAKEOUT = {
    # TRJ officiels PMU 2025 (Mediapronos / PMU.fr)
    "couple_gagnant":    0.740,   # Couplé Gagnant : TRJ = 74.0%
    "couple_place":      0.740,   # Couplé Placé  : TRJ = 74.0%
    "trio_ordre":        0.691,   # Trio          : TRJ = 69.1%
    "trio_desordre":     0.691,   # Trio (même TRJ)
    "quarte_desordre":   0.633,   # Quarté+       : TRJ = 63.3%
    "quinte_desordre":   0.6475,  # Quinté+       : TRJ = 64.75%
}

def _pmu_estimated_odds(p: float, bet_type: str,
                        min_odds: float, max_odds: float) -> float:
    """Estime la cote PMU réelle pour un pari combiné."""
    if p <= 0:
        return max_odds
    payout_rate = PMU_TAKEOUT.get(bet_type, 0.72)
    raw = (1.0 / p) * payout_rate
    return float(np.clip(raw, min_odds, max_odds))


def analyze_exotics(results: List[Dict], orders: np.ndarray,
                     top_n: int = 10) -> Dict[str, List[Dict]]:
    """
    Calcule les meilleurs paris exotiques avec cotes PMU réalistes.
    Tri par ROI espéré décroissant.
    """
    n_iter, n_horses = orders.shape
    output = {"couple_gagnant": [], "couple_place": [],
              "trio_ordre": [], "trio_desordre": [],
              "quarte_desordre": [], "quinte_desordre": []}

    if n_horses < 3:
        return output

    # ──────── COUPLÉ GAGNANT (1-2 ordre exact) ────────
    cg = {}
    for it in range(n_iter):
        key = (int(orders[it, 0]), int(orders[it, 1]))
        cg[key] = cg.get(key, 0) + 1
    for (i, j), c in cg.items():
        p = c / n_iter
        if p < 0.005: continue
        est_odds = _pmu_estimated_odds(p, "couple_gagnant", 3.0, 400.0)
        output["couple_gagnant"].append({
            "combo": f"{results[i]['number']}-{results[j]['number']}",
            "names": f"{results[i]['name'][:8]} → {results[j]['name'][:8]}",
            "prob_pct": round(p * 100, 2),
            "estimated_odds": round(est_odds, 1),
            "expected_roi": round(expected_roi(p, est_odds, 10), 1),
        })

    # ──────── COUPLÉ PLACÉ (2 dans top 3, désordre) ────────
    cp = {}
    for it in range(n_iter):
        top3 = sorted(orders[it, :3].tolist())
        for a, b in combinations(top3, 2):
            key = (int(a), int(b))
            cp[key] = cp.get(key, 0) + 1
    for (i, j), c in cp.items():
        p = c / n_iter
        if p < 0.02: continue
        est_odds = _pmu_estimated_odds(p, "couple_place", 1.8, 80.0)
        output["couple_place"].append({
            "combo": f"{results[i]['number']}-{results[j]['number']}",
            "names": f"{results[i]['name'][:8]} & {results[j]['name'][:8]}",
            "prob_pct": round(p * 100, 2),
            "estimated_odds": round(est_odds, 1),
            "expected_roi": round(expected_roi(p, est_odds, 10), 1),
        })

    # ──────── TRIO ORDRE ────────
    to_dict = {}
    for it in range(n_iter):
        key = tuple(int(x) for x in orders[it, :3])
        to_dict[key] = to_dict.get(key, 0) + 1
    for key, c in to_dict.items():
        p = c / n_iter
        if p < 0.003: continue
        est_odds = _pmu_estimated_odds(p, "trio_ordre", 10.0, 2000.0)
        i, j, k = key
        output["trio_ordre"].append({
            "combo": f"{results[i]['number']}-{results[j]['number']}-{results[k]['number']}",
            "prob_pct": round(p * 100, 3),
            "estimated_odds": round(est_odds, 1),
            "expected_roi": round(expected_roi(p, est_odds, 10), 1),
        })

    # ──────── TRIO DÉSORDRE ────────
    td_dict = {}
    for it in range(n_iter):
        key = tuple(sorted(int(x) for x in orders[it, :3]))
        td_dict[key] = td_dict.get(key, 0) + 1
    for key, c in td_dict.items():
        p = c / n_iter
        if p < 0.01: continue
        est_odds = _pmu_estimated_odds(p, "trio_desordre", 4.0, 500.0)
        i, j, k = key
        output["trio_desordre"].append({
            "combo": f"{results[i]['number']}-{results[j]['number']}-{results[k]['number']}",
            "prob_pct": round(p * 100, 2),
            "estimated_odds": round(est_odds, 1),
            "expected_roi": round(expected_roi(p, est_odds, 10), 1),
        })

    # ──────── QUARTÉ+ DÉSORDRE ────────
    if n_horses >= 4:
        q4 = {}
        for it in range(n_iter):
            key = tuple(sorted(int(x) for x in orders[it, :4]))
            q4[key] = q4.get(key, 0) + 1
        for key, c in q4.items():
            p = c / n_iter
            if p < 0.005: continue
            est_odds = _pmu_estimated_odds(p, "quarte_desordre", 12.0, 5000.0)
            output["quarte_desordre"].append({
                "combo": "-".join(str(results[i]['number']) for i in key),
                "prob_pct": round(p * 100, 3),
                "estimated_odds": round(est_odds, 1),
                "expected_roi": round(expected_roi(p, est_odds, 5), 1),
            })

    # ──────── QUINTÉ+ DÉSORDRE ────────
    if n_horses >= 5:
        q5 = {}
        for it in range(n_iter):
            key = tuple(sorted(int(x) for x in orders[it, :5]))
            q5[key] = q5.get(key, 0) + 1
        for key, c in q5.items():
            p = c / n_iter
            if p < 0.002: continue
            est_odds = _pmu_estimated_odds(p, "quinte_desordre", 25.0, 30000.0)
            output["quinte_desordre"].append({
                "combo": "-".join(str(results[i]['number']) for i in key),
                "prob_pct": round(p * 100, 4),
                "estimated_odds": round(est_odds, 1),
                "expected_roi": round(expected_roi(p, est_odds, 2), 1),
            })

    # Tri : (a) ROI positifs d'abord, (b) puis par probabilité décroissante
    # Cela évite d'afficher des combos peu probables même si ROI équivalent
    for k in output:
        # Plafonner les ROI affichés pour rester réaliste (max +300%)
        for r in output[k]:
            if r["expected_roi"] > 300:
                r["expected_roi_raw"] = r["expected_roi"]
                r["expected_roi"] = 300.0
                r["flag"] = "⚠️ ROI très élevé (cap +300%)"
        output[k].sort(
            key=lambda x: (x["expected_roi"], x["prob_pct"]),
            reverse=True
        )
        output[k] = output[k][:top_n]
        for i, r in enumerate(output[k]):
            r["rank"] = i + 1
    return output


# ==========================================================================
# QUARTÉ COVERAGE — v5.0 : génération 20 combos couvrant seuil de proba
# ==========================================================================
def _pl_prob_top4(strengths: np.ndarray, top4_idx: Tuple[int, int, int, int]) -> float:
    """Probabilité EXACTE d'un top-4 dans l'ordre selon Plackett-Luce."""
    s = strengths
    total = s.sum()
    if total <= 0:
        return 0.0
    p = 1.0
    used = 0.0
    for idx in top4_idx:
        remaining = total - used
        if remaining <= 0:
            return 0.0
        p *= s[idx] / remaining
        used += s[idx]
    return p


def _pl_prob_top4_unordered(strengths: np.ndarray, combo: Tuple[int, int, int, int]) -> float:
    """Probabilité que 4 chevaux (peu importe l'ordre) soient exactement le top-4."""
    return sum(_pl_prob_top4(strengths, perm) for perm in permutations(combo))


def generate_quarte_coverage(
    results: List[Dict],
    strengths: np.ndarray,
    n_combos: int = 20,
    coverage_target: float = 0.50,
    mode: str = "coverage"
) -> Dict[str, Any]:
    """
    Génère N combinaisons Quarté (désordre) intelligentes.

    Modes :
    - 'coverage'   : sélectionne les N combos qui maximisent la couverture cumulée.
                     Retourne aussi la proba cumulée = "chance que le vrai quarté
                     soit parmi nos N tickets".
    - 'top_horses' : identifie d'abord les chevaux à >= coverage_target% de chance
                     d'être dans le top 4, puis génère N combos parmi eux.
    """
    n = len(strengths)
    if n < 4:
        return {"combos": [], "total_coverage_pct": 0, "mode": mode,
                "error": "moins de 4 partants"}

    total_s = strengths.sum()
    if total_s <= 0:
        return {"combos": [], "total_coverage_pct": 0, "mode": mode,
                "error": "forces nulles"}

    # 1) Proba marginale P(cheval i ∈ top 4)
    # Formule Harville exacte : 1 - Π_k (1 - s_i / (Σs - déjà pris))
    # On calcule via simulation partielle exacte pour top 4
    p_top4_marginal = np.zeros(n)
    for i in range(n):
        # Exact : proba d'être dans les 4 premiers = somme sur toutes les positions 1..4
        # de la proba d'y arriver. Formule fermée : intégration compliquée.
        # Approximation via 1 - (1 - s_i/S) itératif sur 4 positions :
        p_not_yet = 1.0
        s_remaining = total_s
        p_in_top4 = 0.0
        for pos in range(4):
            # P(sortir à cette position | pas encore sorti)
            p_at_pos = strengths[i] / s_remaining if s_remaining > 0 else 0
            p_in_top4 += p_not_yet * p_at_pos
            p_not_yet *= (1 - p_at_pos)
            # Réduction moyenne du dénominateur (approximation)
            s_remaining -= (total_s - strengths[i]) / (n - 1) if n > 1 else 0
        p_top4_marginal[i] = min(1.0, p_in_top4)

    # 2) Génération des combinaisons candidates
    # On limite au pool des 12 meilleurs pour rester tractable (C(12,4)=495)
    ranking = np.argsort(-strengths)
    n_pool = min(12, n)
    pool = ranking[:n_pool].tolist()

    all_combos = []
    for combo in combinations(pool, 4):
        p = _pl_prob_top4_unordered(strengths, combo)
        all_combos.append({"indices": combo, "prob": p})
    all_combos.sort(key=lambda x: -x["prob"])

    # 3) Sélection selon mode
    if mode == "top_horses":
        strong = [i for i in range(n) if p_top4_marginal[i] >= coverage_target]
        if len(strong) < 4:
            # Fallback : les 6 meilleurs
            strong = ranking[:6].tolist()
        selected = [c for c in all_combos if all(idx in strong for idx in c["indices"])]
        selected = selected[:n_combos]
    else:  # 'coverage'
        selected = all_combos[:n_combos]

    # 4) Couverture cumulée
    cum = 0.0
    for c in selected:
        cum += c["prob"]
        c["cum_prob"] = cum

    # 5) Format lisible
    combos_out = []
    for i, c in enumerate(selected):
        nums = sorted(results[idx]["number"] for idx in c["indices"])
        names = tuple(results[idx]["name"][:12] for idx in c["indices"])
        combos_out.append({
            "rank": i + 1,
            "numbers": tuple(nums),
            "combo": "-".join(str(n) for n in nums),
            "names": names,
            "prob_pct": round(c["prob"] * 100, 3),
            "cum_prob_pct": round(c["cum_prob"] * 100, 2),
        })

    # 6) Estimation ROI (Quarté+ base 1.30€)
    # v5.0 : TRJ Quarté+ officiel PMU = 63.3% (Mediapronos / PMU.fr)
    # Ancienne valeur v4.3 : 0.71 → ERRONÉE (valeur réelle = 0.633)
    stake_per_combo = 1.30
    total_stake = len(combos_out) * stake_per_combo
    # Cote Quarté+ : (1/proba_gagnante) × TRJ_PMU (0.633)
    # ROI = cum × cote_moy - 1
    if cum > 0 and len(combos_out) > 0:
        p_moy = cum / len(combos_out)
        cote_moy = 0.633 / p_moy  # v5.0 : TRJ Quarté+ = 63.3%
        expected_win = cum * cote_moy * stake_per_combo
        roi_pct = ((expected_win - total_stake) / total_stake) * 100
    else:
        cote_moy = 0
        roi_pct = -100

    return {
        "combos": combos_out,
        "total_coverage_pct": round(cum * 100, 2),
        "coverage_target_pct": round(coverage_target * 100, 1),
        "n_combos": len(combos_out),
        "total_stake_eur": round(total_stake, 2),
        "avg_estimated_odds": round(cote_moy, 1),
        "roi_estimate_pct": round(roi_pct, 1),
        "mode": mode,
        "strong_horses_count": int(sum(1 for p in p_top4_marginal
                                        if p >= coverage_target)),
        "p_top4_marginal": {results[i]["number"]: round(float(p) * 100, 1)
                             for i, p in enumerate(p_top4_marginal)},
    }


def best_place_bet(results: List[Dict], n_runners: int) -> Optional[Dict]:
    """Trouve le meilleur cheval pour le pari Placé."""
    if n_runners <= 4:
        place_factor = CONFIG.PLACE_ODDS_FACTOR["small"]
    elif n_runners <= 7:
        place_factor = 0.45
    elif n_runners <= 15:
        place_factor = CONFIG.PLACE_ODDS_FACTOR["medium"]
    else:
        place_factor = CONFIG.PLACE_ODDS_FACTOR["large"]

    best = None
    best_roi = -np.inf
    for r in results:
        pp = r["place_prob"] / 100
        if pp < 0.12: continue
        wo = r["odds"]
        if wo < 1.5: continue
        place_odds = max(1.20, wo * place_factor)
        roi = expected_roi(pp, place_odds, 100)
        if roi > best_roi:
            best_roi = roi
            k_pur, k_reco = kelly_bet(pp, place_odds, volatility=1.0)
            best = {
                "number": r["number"],
                "name": r["name"],
                "win_prob": r["win_prob"],
                "place_prob": r["place_prob"],
                "estimated_place_odds": round(place_odds, 2),
                "expected_roi_place": round(roi, 1),
                "kelly_pure": round(k_pur, 4),
                "kelly_recommended": round(k_reco, 4),
            }
    return best


# =============================================================================
# 9.  MOTEUR PRINCIPAL — RaceEngine v4
# =============================================================================
class RaceEngine:
    """Encapsule toute la logique de prédiction pour une course."""

    def __init__(self, race_info: Dict, horses: List[Dict]):
        self.race_info = race_info
        self.horses = horses
        self.n = len(horses)
        self.race_type = race_info.get("race_type", "Plat")
        self.distance = int(race_info.get("distance", 1600))
        self.track = race_info.get("track", "Bon")
        self.depart_type = race_info.get("depart_type", "Stalles (Plat)")

    # ── 9.1 Préparation des features ───────────────────────────────────
    def _build_features(self) -> Tuple[List[Dict], List[int], np.ndarray]:
        feats, draws, exp_factors = [], [], []
        for h in self.horses:
            m_h = parse_music_v4(h.get("horse_music", ""))
            m_d = parse_music_v4(h.get("driver_music", ""))
            m_t = parse_music_v4(h.get("trainer_music", ""))

            exp_h = experience_factor(m_h.races_count)
            exp_d = experience_factor(m_d.races_count)
            exp_t = experience_factor(m_t.races_count)
            combined_exp = (exp_h * exp_d * exp_t) ** (1/3)
            exp_factors.append(combined_exp)

            draw = h.get("draw", 0)
            draws.append(draw)

            df = draw_factor_v4(draw, self.race_type, self.distance,
                                 self.depart_type, self.track)
            wf = weight_factor(h.get("weight", 0)) if self.race_type == "Plat" else 1.0
            rf = rest_factor(h.get("days_rest", -1))
            tf = track_factor(self.track, self.race_type)

            # On utilise les scores SHRUNK (régression vers moyenne)
            feats.append({
                "number": h.get("number", 0),
                "name": h.get("name", ""),
                "odds": float(h.get("odds", 0)),
                "horse_score": m_h.shrunk_score * exp_h * tf,
                "horse_form": m_h.recent_form,
                "horse_regularity": m_h.regularity,
                "horse_trend": m_h.trend,
                "horse_win": m_h.shrunk_win_ratio,
                "horse_is_debutant": m_h.is_debutant,
                "driver_score": m_d.shrunk_score * exp_d,
                "driver_form": m_d.recent_form,
                "driver_win": m_d.shrunk_win_ratio,
                "trainer_score": m_t.shrunk_score * exp_t,
                "trainer_form": m_t.recent_form,
                "trainer_win": m_t.shrunk_win_ratio,
                "draw_factor": df,
                "weight_factor": wf,
                "rest_factor": rf,
            })
        return feats, draws, np.array(exp_factors)

    # ── 9.2 Prédiction complète ────────────────────────────────────────
    def predict(self, mc_iter: int = None, market_weight: float = None,
                value_threshold: float = None) -> Dict[str, Any]:
        t0 = time.time()
        if mc_iter is None:        mc_iter = CONFIG.MC_ITERATIONS
        if market_weight is None:  market_weight = CONFIG.MARKET_WEIGHT
        if value_threshold is None: value_threshold = CONFIG.VALUE_THRESHOLD

        feats, draws, exp_factors = self._build_features()
        weights = get_weights_v4(self.race_type)
        scores = np.array([composite_score_v4(f, weights) for f in feats])
        if scores.std() < 1e-6:
            scores += np.random.normal(0, 0.05, self.n)

        # === ÉTAPE 1 : Probabilité modèle pure (softmax) ===
        p_model_raw = softmax_temp(scores, T=CONFIG.TEMPERATURE)

        # === ÉTAPE 2 : Correction empirique (corde + expérience) ===
        p_model = empirical_correction(p_model_raw, draws, self.race_type,
                                         self.distance, self.depart_type,
                                         exp_factors)

        # === ÉTAPE 3 : Marché débiaisé ===
        odds_arr = np.array([f["odds"] for f in feats])
        has_market = (odds_arr > 1.5).sum() >= self.n * 0.5
        if has_market:
            p_market = remove_overround(odds_arr)
        else:
            p_market = np.ones(self.n) / self.n

        # === ÉTAPE 3.5 (v4.1) : Platt scaling du modèle ===
        p_model = platt_calibrate(p_model, race_type=self.race_type)

        # === ÉTAPE 4 : Benter Blend (v4.1 : discipline-aware) ===
        if has_market and market_weight > 0:
            p_final = benter_blend(p_model, p_market,
                                    race_type=self.race_type)
            if abs(market_weight - 0.50) > 0.05:
                p_final = (1 - market_weight) * p_model + market_weight * p_final
                p_final /= p_final.sum()
        else:
            p_final = p_model

        # === ÉTAPE 5 : Simulation Plackett-Luce pour exotiques + place ===
        # On reconstruit des forces compatibles avec p_final
        strengths = p_final * 100  # échelle arbitraire
        orders = plackett_luce_simulate(strengths, mc_iter, noise=CONFIG.NOISE_BASE)

        # Probabilités de place (top 3) via PL
        place_counts = np.zeros(self.n)
        win_counts = np.zeros(self.n)
        for it in range(mc_iter):
            win_counts[orders[it, 0]] += 1
            for k in range(3):
                place_counts[orders[it, k]] += 1
        p_place_mc = place_counts / mc_iter
        p_win_mc = win_counts / mc_iter

        # Volatilité : écart entre p_final et p_win_mc
        volatility = np.abs(p_final - p_win_mc) / (p_final + 1e-9)

        # === ÉTAPE 6 : Construction des résultats ===
        results = []
        # Overround
        if has_market:
            raw_or = sum(1.0 / o for o in odds_arr if o > 1.01)
            overround_pct = round((raw_or - 1.0) * 100, 1)
        else:
            overround_pct = None

        # Seuil de value dynamique
        if overround_pct is not None and overround_pct > 0:
            dyn_value_th = max(value_threshold, 1.0 + overround_pct / 100 * 1.2)
        else:
            dyn_value_th = value_threshold

        for i, (feat, horse) in enumerate(zip(feats, self.horses)):
            ratio = p_final[i] / (p_market[i] + 1e-9)
            cote = horse.get("odds", 2.0)
            # v4.1 : filtre value bet par cote (sweet spot [5, 10])
            is_value = (
                ratio >= dyn_value_th
                and p_final[i] >= 0.04
                and CONFIG.VALUE_COTE_MIN <= cote <= CONFIG.VALUE_COTE_MAX
            )
            k_pur, k_reco = kelly_bet(p_final[i], cote,
                                       volatility=1 + volatility[i])
            roi = expected_roi(p_final[i], cote)

            results.append({
                "rank": 0,
                "number": horse.get("number", i + 1),
                "name": horse.get("name", f"Cheval {i+1}"),
                "odds": float(horse.get("odds", 0)),
                "win_prob": round(float(p_final[i]) * 100, 2),
                "win_prob_model": round(float(p_model[i]) * 100, 2),
                "win_prob_market": round(float(p_market[i]) * 100, 2),
                "place_prob": round(float(p_place_mc[i]) * 100, 2),
                "composite_score": round(float(scores[i]), 3),
                "value_ratio": round(float(ratio), 2),
                "is_value_bet": bool(is_value),
                "kelly_pure": round(k_pur, 4),
                "kelly_recommended": round(k_reco, 4),
                "expected_roi": round(roi, 2),
                "volatility": round(float(volatility[i]), 3),
                "draw": draws[i],
                "draw_factor": round(feat["draw_factor"], 3),
            })

        results.sort(key=lambda x: x["win_prob"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1

        # === ÉTAPE 7 : Exotiques + Place + Quarté Coverage ===
        exotics = analyze_exotics(results, orders)
        bp = best_place_bet(results, self.n)
        quarte_coverage_20 = generate_quarte_coverage(
            results, strengths,
            n_combos=20,
            coverage_target=0.50,
            mode="coverage"
        )
        quarte_top_horses_20 = generate_quarte_coverage(
            results, strengths,
            n_combos=20,
            coverage_target=0.50,
            mode="top_horses"
        )

        # === Diagnostic ===
        sorted_p = sorted([r["win_prob"] for r in results], reverse=True)
        if len(sorted_p) >= 2:
            gap = sorted_p[0] - sorted_p[1]
            conf_idx = min(100, round(45 + gap * 2.5, 1))
        else:
            conf_idx = 50
        vol_idx = min(100, round(volatility.mean() * 60, 1))

        # KL divergence modèle / marché (mesure de désaccord)
        if has_market:
            eps = 1e-12
            kl = float(np.sum(p_final * np.log((p_final + eps) / (p_market + eps))))
        else:
            kl = None

        return {
            "results": results,
            "exotics": exotics,
            "best_place": bp,
            "quarte_coverage_20": quarte_coverage_20,
            "quarte_top_horses_20": quarte_top_horses_20,
            "confidence_idx": conf_idx,
            "volatility_idx": vol_idx,
            "overround_pct": overround_pct,
            "dynamic_value_threshold": round(dyn_value_th, 3),
            "kl_divergence": round(kl, 3) if kl else None,
            "execution_time": round(time.time() - t0, 2),
            "n_simulations": mc_iter,
        }


def run_engine_v4(race_info: Dict, horses: List[Dict], **kwargs) -> Dict:
    """API publique compatible avec l'ancienne v3."""
    engine = RaceEngine(race_info, horses)
    return engine.predict(**kwargs)


# =============================================================================
# 10.  INTERFACE STREAMLIT
# =============================================================================
def apply_css():
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg,#07071a 0%,#0d1b2a 40%,#12192b 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#0d1b2a,#07071a); }
    h1, h2, h3 { color:#e8e8e8 !important; }
    div[data-testid="metric-container"] {
        background: rgba(0,180,216,0.08);
        border: 1px solid rgba(0,255,136,0.15);
        border-radius: 12px;
        padding: 10px;
    }
    .value-bet { color:#00ff88; font-weight:bold; }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    st.markdown(f"""
    <div style="text-align:center; padding: 18px 0;">
      <h1 style="font-size:2.6em;
                 background: linear-gradient(90deg,#00ff88,#00b4d8,#7b2ff7);
                 -webkit-background-clip:text;
                 -webkit-text-fill-color:transparent;">
        🏇 {CONFIG.APP_NAME} v{CONFIG.APP_VERSION}
      </h1>
      <p style="color:#7b9ec4; font-size:1.05em; margin-top:-10px;">
        <em>{CONFIG.APP_TAG}</em> — Plackett-Luce · Benter Blend · Kelly dynamique
      </p>
    </div>
    """, unsafe_allow_html=True)


def init_session_state():
    if "horses_data" not in st.session_state:
        st.session_state.horses_data = pd.DataFrame({
            "N°": list(range(1, 11)),
            "Nom": [f"Cheval {i+1}" for i in range(10)],
            "Cote": [5.0] * 10,
            "Musique Cheval": [""] * 10,
            "Musique Driver": [""] * 10,
            "Musique Entraîneur": [""] * 10,
            "Corde": [0] * 10,
            "Poids": [56.0] * 10,
            "Jours repos": [21] * 10,
        })
    if "prediction" not in st.session_state:
        st.session_state.prediction = None


def main():
    st.set_page_config(page_title=f"🏇 {CONFIG.APP_NAME} v{CONFIG.APP_VERSION}",
                       layout="wide", initial_sidebar_state="expanded")
    init_session_state()
    apply_css()
    render_header()

    # ============= SIDEBAR =============
    with st.sidebar:
        st.markdown("### ⚙️ Paramètres du moteur")

        with st.expander("🔬 Monte Carlo / Plackett-Luce", expanded=True):
            mc_iter = st.slider("Itérations PL", 1000, 15000,
                                CONFIG.MC_ITERATIONS, 500)
            noise = st.slider("Bruit log-normal", 0.05, 0.40,
                              CONFIG.NOISE_BASE, 0.01)
            CONFIG.NOISE_BASE = noise

        with st.expander("🎯 Marché & Benter Blend", expanded=True):
            mw = st.slider("Poids du marché", 0.0, 0.70,
                           CONFIG.MARKET_WEIGHT, 0.05)
            alpha = st.slider("α (exposant modèle)", 0.5, 2.0,
                              CONFIG.BENTER_ALPHA, 0.05)
            beta = st.slider("β (exposant marché)", 0.0, 2.0,
                             CONFIG.BENTER_BETA, 0.05)
            CONFIG.BENTER_ALPHA = alpha
            CONFIG.BENTER_BETA = beta
            CONFIG.OVERROUND_CORRECTION = st.checkbox(
                "Débiaiser favori/outsider", value=True,
                help="Correction power du biais favori-outsider")

        with st.expander("🧠 Empirique & shrinkage"):
            emp_w = st.slider("Poids empirisme (corde+exp.)", 0.0, 0.70,
                               CONFIG.EMPIRICAL_WEIGHT, 0.05)
            CONFIG.EMPIRICAL_WEIGHT = emp_w
            CONFIG.USE_EXPERIENCE_FACTOR = st.checkbox(
                "Facteur expérience", value=CONFIG.USE_EXPERIENCE_FACTOR)
            K = st.slider("Shrinkage K (courses fantômes)", 0.0, 15.0,
                          CONFIG.SHRINKAGE_K, 0.5,
                          help="Plus K est élevé, plus on tire vers la moyenne population")
            CONFIG.SHRINKAGE_K = K

        with st.expander("💰 Value & Kelly"):
            vt = st.slider("Seuil de value (ratio)", 1.05, 1.80,
                            CONFIG.VALUE_THRESHOLD, 0.05)
            kf = st.slider("Kelly fractionnaire", 0.05, 0.50,
                            CONFIG.KELLY_FRACTION, 0.05)
            CONFIG.KELLY_FRACTION = kf
            max_stake = st.slider("Cap max bankroll (%)", 1.0, 15.0,
                                  CONFIG.MAX_KELLY_STAKE * 100, 0.5) / 100
            CONFIG.MAX_KELLY_STAKE = max_stake

        st.markdown("---")
        st.caption(f"v{CONFIG.APP_VERSION} — {CONFIG.APP_TAG}")
        st.caption("Inspiré de Benter (1994), Harville (1973)")

    # ============= TABS =============
    tab1, tab2, tab3 = st.tabs(["📥 Données course",
                                "📊 Pronostics",
                                "ℹ️ Aide & Méthode"])

    # ---------- TAB 1 : DONNÉES ----------
    with tab1:
        st.markdown("## 🏁 Informations de la course")
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.5])
        with c1:
            race_type = st.selectbox("Discipline", CONFIG.RACE_TYPES)
        with c2:
            distance = st.number_input("Distance (m)", 800, 7200, 1600, 100)
        with c3:
            track = st.selectbox("Terrain", CONFIG.TRACK_CONDITIONS)
        with c4:
            # Auto-suggestion du type de départ
            default_depart = 0
            if race_type in ("Attelé", "Monté"):
                default_depart = 1
            depart = st.selectbox("Type de départ", CONFIG.DEPART_TYPES,
                                  index=default_depart)

        prix = st.text_input("Nom du prix (optionnel)", "")

        st.markdown("---")
        st.markdown("## 🐎 Tableau des partants")
        st.caption("✏️ Modifiez directement le tableau. Les champs **Poids** et "
                   "**Jours repos** sont utilisés en Plat ; en Trot, seul "
                   "**Jours repos** est exploité.")

        edited = st.data_editor(
            st.session_state.horses_data,
            use_container_width=True,
            num_rows="dynamic",
            height=420,
            column_config={
                "N°": st.column_config.NumberColumn(min_value=1, max_value=99),
                "Cote": st.column_config.NumberColumn(format="%.2f", min_value=1.0),
                "Corde": st.column_config.NumberColumn(min_value=0, max_value=20),
                "Poids": st.column_config.NumberColumn(format="%.1f", min_value=40.0, max_value=80.0),
                "Jours repos": st.column_config.NumberColumn(min_value=0, max_value=999),
            },
        )
        if edited is not None:
            st.session_state.horses_data = edited

        c1, c2 = st.columns([3, 1])
        with c1:
            run_btn = st.button("🚀 LANCER L'ANALYSE",
                                 use_container_width=True, type="primary")
        with c2:
            reset_btn = st.button("🔄 Reset", use_container_width=True)
            if reset_btn:
                st.session_state.horses_data = pd.DataFrame({
                    "N°": list(range(1, 11)),
                    "Nom": [f"Cheval {i+1}" for i in range(10)],
                    "Cote": [5.0] * 10,
                    "Musique Cheval": [""] * 10,
                    "Musique Driver": [""] * 10,
                    "Musique Entraîneur": [""] * 10,
                    "Corde": [0] * 10,
                    "Poids": [56.0] * 10,
                    "Jours repos": [21] * 10,
                })
                st.rerun()

        if run_btn:
            horses_list = []
            for idx, row in st.session_state.horses_data.iterrows():
                try:
                    horses_list.append({
                        "number": int(row["N°"]),
                        "name": str(row["Nom"]),
                        "odds": float(row["Cote"]),
                        "horse_music": str(row["Musique Cheval"]),
                        "driver_music": str(row["Musique Driver"]),
                        "trainer_music": str(row["Musique Entraîneur"]),
                        "draw": int(row["Corde"]) if pd.notna(row["Corde"]) else 0,
                        "weight": float(row.get("Poids", 56.0)) if pd.notna(row.get("Poids")) else 56.0,
                        "days_rest": int(row.get("Jours repos", -1)) if pd.notna(row.get("Jours repos")) else -1,
                    })
                except Exception as e:
                    st.error(f"⚠️ Erreur ligne {idx+1} : {e}")
                    return

            if len(horses_list) < 3:
                st.error("Au moins 3 partants requis.")
                return

            with st.spinner(f"🔬 Calcul Plackett-Luce ({mc_iter} simulations)..."):
                pred = run_engine_v4(
                    {"race_type": race_type, "distance": distance,
                     "track": track, "depart_type": depart, "discipline": prix},
                    horses_list,
                    mc_iter=mc_iter, market_weight=mw, value_threshold=vt
                )
                st.session_state.prediction = pred
            st.success(f"✅ Analyse terminée en {pred['execution_time']}s — "
                       f"{pred['n_simulations']} simulations")

    # ---------- TAB 2 : RÉSULTATS ----------
    with tab2:
        if st.session_state.prediction is None:
            st.info("🎯 Saisissez les données puis cliquez sur **LANCER L'ANALYSE**.")
        else:
            pred = st.session_state.prediction

            # Diagnostic
            st.markdown("## 📈 Diagnostic de course")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🎯 Confiance", f"{pred['confidence_idx']:.1f}/100")
            c2.metric("🌪️ Volatilité", f"{pred['volatility_idx']:.1f}/100")
            if pred["overround_pct"] is not None:
                c3.metric("📉 Overround", f"{pred['overround_pct']:.1f}%",
                          help="Marge bookmaker (>20% = juice élevé)")
            else:
                c3.metric("📉 Overround", "—")
            c4.metric("📐 Seuil value (dyn.)", f"{pred['dynamic_value_threshold']:.2f}",
                      help="Ajusté selon overround")

            if pred["kl_divergence"] is not None:
                st.caption(f"🧮 Divergence KL(modèle ‖ marché) = "
                           f"**{pred['kl_divergence']:.3f}** — "
                           f"{'fort désaccord' if pred['kl_divergence'] > 0.15 else 'accord modéré'}")

            st.markdown("---")
            st.markdown("## 🏆 Classement final & paris GAGNANT")
            df = pd.DataFrame([{
                "Rg": r["rank"],
                "N°": r["number"],
                "Nom": r["name"][:18],
                "Cote": f"{r['odds']:.2f}",
                "Modèle %": f"{r['win_prob_model']:.1f}",
                "Marché %": f"{r['win_prob_market']:.1f}",
                "🎯 Final %": f"{r['win_prob']:.2f}",
                "Placé %": f"{r['place_prob']:.1f}",
                "Ratio": f"{r['value_ratio']:.2f}",
                "ROI %": f"{r['expected_roi']:+.1f}",
                "Kelly %": f"{r['kelly_recommended']*100:.2f}",
                "Vol.": f"{r['volatility']:.2f}",
                "Value": "🟢" if r["is_value_bet"] else "⚪",
            } for r in pred["results"]])
            st.dataframe(df, use_container_width=True, hide_index=True, height=380)

            # Value bets en évidence
            value_bets = [r for r in pred["results"] if r["is_value_bet"]]
            if value_bets:
                st.markdown("### 💎 Value bets détectés")
                for vb in value_bets[:5]:
                    st.markdown(
                        f"- **N°{vb['number']} {vb['name']}** "
                        f"@ cote {vb['odds']:.2f} — "
                        f"prob. modèle {vb['win_prob']:.1f}% vs marché {vb['win_prob_market']:.1f}% "
                        f"→ Kelly recommandé : **{vb['kelly_recommended']*100:.2f}%** "
                        f"(ROI espéré : {vb['expected_roi']:+.1f}%)"
                    )
            else:
                st.info("⚪ Aucun value bet détecté sur ce marché.")

            # Meilleur placé
            if pred["best_place"]:
                bp = pred["best_place"]
                st.markdown("---")
                st.markdown("## 🥉 Meilleur pari **PLACÉ**")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("N°", bp["number"])
                c2.metric("Cheval", bp["name"][:15])
                c3.metric("Prob. Placé", f"{bp['place_prob']:.1f}%")
                c4.metric("ROI Placé", f"{bp['expected_roi_place']:+.1f}%")
                st.markdown(
                    f"💡 Cote placé estimée : **{bp['estimated_place_odds']:.2f}** — "
                    f"Mise Kelly recommandée : **{bp['kelly_recommended']*100:.2f}%** "
                    f"du bankroll"
                )

            # Exotiques
            st.markdown("---")
            st.markdown("## 🎲 Paris exotiques (Top combinaisons)")
            ex = pred["exotics"]
            tabs_exo = st.tabs(["Couplé Gagnant", "Couplé Placé",
                                "Trio Ordre", "Trio Désordre",
                                "Quarté+", "Quinté+", "Quarté 20 combis"])

            def _render_exotic(items, key):
                if not items:
                    st.info("Aucune combinaison significative.")
                    return
                df_e = pd.DataFrame([{
                    "Rg": x["rank"],
                    "Combo": x.get("combo", "—"),
                    **({"Détail": x["names"]} if "names" in x else {}),
                    "Prob %": x["prob_pct"],
                    "Cote est.": x["estimated_odds"],
                    "ROI %": x["expected_roi"],
                } for x in items])
                st.dataframe(df_e, use_container_width=True, hide_index=True)

            def _render_quarte_coverage(block, title):
                st.markdown(f"### {title}")
                if not block or not block.get("combos"):
                    st.info("Aucune combinaison générée.")
                    return
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Combinaisons", block.get("n_combos", 0))
                c2.metric("Couverture totale", f"{block.get('total_coverage_pct', 0):.2f}%")
                c3.metric("Mise totale", f"{block.get('total_stake_eur', 0):.2f} €")
                c4.metric("ROI estimé", f"{block.get('roi_estimate_pct', 0):+.1f}%")
                df_q = pd.DataFrame([{
                    "Rg": x["rank"],
                    "Combo": x["combo"],
                    "Prob %": x["prob_pct"],
                    "Cumul %": x["cum_prob_pct"],
                } for x in block["combos"]])
                st.dataframe(df_q, use_container_width=True, hide_index=True, height=420)
                strong = block.get("strong_horses_count", 0)
                st.caption(
                    f"Mode: {block.get('mode', 'coverage')} — chevaux >= 50% top 4 : {strong}. "
                    f"La couverture totale représente la chance que le vrai Quarté soit présent parmi les 20 tickets."
                )

            with tabs_exo[0]: _render_exotic(ex["couple_gagnant"], "cg")
            with tabs_exo[1]: _render_exotic(ex["couple_place"], "cp")
            with tabs_exo[2]: _render_exotic(ex["trio_ordre"], "to")
            with tabs_exo[3]: _render_exotic(ex["trio_desordre"], "td")
            with tabs_exo[4]: _render_exotic(ex["quarte_desordre"], "q4")
            with tabs_exo[5]: _render_exotic(ex["quinte_desordre"], "q5")
            with tabs_exo[6]:
                _render_quarte_coverage(pred.get("quarte_coverage_20"), "📦 Mode couverture 20 tickets")
                st.markdown("---")
                _render_quarte_coverage(pred.get("quarte_top_horses_20"), "🎯 Mode chevaux ≥ 50% top 4")

    # ---------- TAB 3 : AIDE ----------
    with tab3:
        st.markdown(r"""
## 🎓 Méthodologie QuantTurf v5.0

### 🔬 Architecture du moteur

```
Musique → Parsing + Shrinkage bayésien → Score composite
                                              ↓
                                          Softmax
                                              ↓
                                  Correction empirique (corde+exp)
                                              ↓
                                       p_modèle
                                              ↓
Cotes marché → Débiaisage Prelec (a=0.928) → p_marché
                                              ↓
                              BENTER BLEND : p ∝ p_modèle^α · p_marché^β
                                              ↓
                          Plackett-Luce + HENERY γ=0.58 (8000 ordres simulés)
                                              ↓
                    Win / Place / Couplé / Trio / Quarté+ / Quinté+
                                              ↓
                                   Kelly dynamique + Thorp guard + ROI
```

### 📚 Formules clés

**1. Shrinkage bayésien (musique)**
$$\text{score}_{\text{shrunk}} = \frac{n \cdot \text{score}_{\text{obs}} + K \cdot \mu_{\text{pop}}}{n + K}$$

**2. Débiaisage des cotes — Fonction de Prelec (v5.0)**
$$w(p) = \exp\left(-(-\ln p)^a\right), \quad a = 0.928$$

*Remplace la transformation power (γ=1.12) — calibration Snowberg & Wolfers (2010) sur 6.4M de courses.*

**3. Benter Blend (log-log fusion)**
$$p_{\text{finale}} \propto p_{\text{modèle}}^\alpha \cdot p_{\text{marché}}^\beta$$

*Formule originale Benter (1994) : $c_i = \frac{\exp(\alpha \ln f_i + \beta \ln \pi_i)}{\sum_j \exp(\alpha \ln f_j + \beta \ln \pi_j)}$*

**4. Plackett-Luce + Correction Henery (v5.0)**

*Harville (1973) : ordre d'arrivée séquentiel proportionnel aux forces.*
*Correction Henery : $\log(\psi_i) = \theta_i + \gamma \cdot \theta_i^2$, $\gamma = 0.58$*
*Lo & Bacon-Shone (1994) : corrige le biais Harville de surestimation des favoris en positions 2-3.*

**5. Kelly fractionnaire dynamique + Thorp guard (v5.0)**
$$f^* = \frac{p \cdot b - q}{b}, \quad f_{\text{misé}} = \min\left(f^* \cdot \frac{1}{1+\text{vol}} \cdot g_{\text{Thorp}}, f_{\max}\right)$$

*$g_{\text{Thorp}} = 0.85$ : protection contre la surestimation systématique de P(gagner) (Thorp 2007).*

### 🔧 Améliorations v5.0 vs v4.3

| Composant | v4.3 | v5.0 | Source |
|---|---|---|---|
| **Débiaisage FLB** | Power γ=1.12 | Prelec a=0.928 | Snowberg & Wolfers (2010), 6.4M starts |
| **Biais Harville** | Non corrigé | Henery γ=0.58 | Lo & Bacon-Shone (1994), Ali (1998) |
| **TRJ PMU Couplé** | 0.74-0.78 | 0.740 (officiel) | Mediapronos / PMU.fr 2025 |
| **TRJ PMU Trio** | 0.72-0.74 | 0.691 (officiel) | Mediapronos / PMU.fr 2025 |
| **TRJ PMU Quarté+** | 0.71 | 0.633 (officiel) | Mediapronos / PMU.fr 2025 |
| **TRJ PMU Quinté+** | 0.68 | 0.6475 (officiel) | Mediapronos / PMU.fr 2025 |
| **Autostart favorisés** | 4-5-6 | 3-4-5-6-7 | Turf.bzh, PMU.fr, Turfomania |
| **Kelly guard** | Volatilité seule | + Thorp guard 0.85 | Thorp (2007) |
| **Market weight** | 0.55 | 0.45 | Évite double-comptage Benter |

### 📊 Taux de Redistribution PMU (TRJ) officiels 2025

| Pari | TRJ | Prélèvement PMU |
|---|---|---|
| Couplé (G/P/O) | 74.0% | 26.0% |
| Trio | 69.1% | 30.9% |
| Tiercé | 64.35% | 35.65% |
| Quarté+ | 63.3% | 36.7% |
| Quinté+ | 64.75% | 35.25% |

### 🎯 Stratégie recommandée

| Type de pari | Quand l'utiliser | Risque |
|---|---|---|
| **Gagnant (value)** | Ratio > 1.20 ET cote ∈ [4.5, 13] | 🟡 Moyen |
| **Placé** | Champion avec cote ≥ 4 | 🟢 Faible |
| **Couplé Placé** | ROI > 50% | 🟡 Moyen |
| **Trio désordre** | ROI > 100% sur 3 favoris | 🟠 Élevé |
| **Quinté+** | Mise faible, ROI espéré > 200% | 🔴 Très élevé |

### ⚠️ Avertissements

- 🎰 **Les performances passées ne préjugent pas des résultats futurs**
- 💸 **Jouez avec modération** — ne misez jamais plus que ce que vous pouvez perdre
- 📊 Le modèle nécessite un marché suffisamment liquide pour le Benter Blend
- 🐎 La corde au Trot n'est pertinente qu'en départ **AUTOSTART**
- 🔍 Les statistiques empiriques sont des **valeurs indicatives basées sur des études publiques** ; affinez-les selon votre propre base de données.
- 📐 La correction Henery (γ=0.58) est calibrée sur données Hong Kong ; un recalibrage sur données PMU françaises pourrait être nécessaire.

### 📖 Références

- Benter, W. (1994). *Computer Based Horse Race Handicapping and Wagering Systems.*
- Harville, D. (1973). *Assigning Probabilities to the Outcomes of Multi-Entry Competitions.*
- Henery, R.J. (1983). *Permutation probabilities as models for horse races.*
- Lo, V. & Bacon-Shone, J. (1994). *Probability and Optimization Models for Racing.* — γ=0.58 Henery correction (HK data)
- Ali, M.M. (1998). *Probability models on horse-race outcomes.* — Harville bias confirmation (15k races)
- Snowberg, E. & Wolfers, J. (2010). *Explaining the Favorite-Longshot Bias.* — Prelec a=0.928 (6.4M starts)
- Prelec, D. (1998). *The Probability Weighting Function.*
- Plackett, R. (1975). *The Analysis of Permutations.*
- Kelly, J. L. (1956). *A New Interpretation of Information Rate.*
- Thorp, E.O. (2007). *The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market.* — fractional Kelly / overestimation guard
        """)


if __name__ == "__main__":
    main()
