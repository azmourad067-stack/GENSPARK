import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import zscore, norm, beta
from itertools import combinations
import re
import time
import warnings
import json
import io
import hashlib
from functools import lru_cache

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION GLOBALE & CONSTANTES
# =============================================================================

APP_VERSION = "3.0.0"
APP_NAME    = "QuantTurf Pro Ultimate"

RACE_TYPES = ["Plat", "Attelé", "Monté", "Haies", "Steeple-chase", "Cross-country"]
GOINGS     = ["Non connu", "Lourd", "Très souple", "Souple", "Bon souple", "Bon", "Ferme"]
SURFACES   = ["Non connue", "Herbe", "Sable", "PSF", "Fibresand"]

# Scores par position dans la musique (étendu aux codes spéciaux)
MUSIC_POSITION_SCORES = {
    "1": 10.0, "2": 7.5, "3": 5.5, "4": 4.0, "5": 3.0,
    "6": 2.0,  "7": 1.5, "8": 1.0, "9": 0.5, "0": 0.0,
    "D": -3.0, "A": -2.5, "T": -2.0, "R": -1.8, "P": 0.2,
}

MUSIC_RACE_TYPE_WEIGHTS = {
    "a": 1.00, "m": 0.90, "p": 1.00, "h": 0.95, "s": 0.90, "c": 0.85, "x": 1.00
}

# Impact du numéro de corde (plat) – fonction non linéaire
DRAW_IMPACT_BASE = {
    1: 0.35, 2: 0.40, 3: 0.35, 4: 0.25, 5: 0.15,
    6: 0.05, 7: -0.05, 8: -0.12, 9: -0.18, 10: -0.24,
    11: -0.30, 12: -0.35, 13: -0.40, 14: -0.44, 15: -0.48,
    16: -0.50, 17: -0.52, 18: -0.54, 19: -0.55, 20: -0.55,
}

# Pondération exponentielle dans la musique (plus de poids aux courses récentes)
DECAY_FACTOR = 0.30

# Paramètres par défaut
DEFAULT_MC_ITERATIONS   = 3000
DEFAULT_MARKET_WEIGHT   = 0.30
DEFAULT_VALUE_THRESHOLD = 1.20
TEMPERATURE_SOFTMAX     = 1.2
NOISE_BASE              = 0.12
NOISE_CORRELATION       = 0.25   # corrélation du bruit entre chevaux (effet course)

# Pondérations des facteurs pour le score composite avancé
WEIGHTS_ADVANCED = {
    "music_score":     0.22,
    "recent_form":     0.12,
    "regularity":      0.06,
    "trend":           0.04,
    "win_ratio":       0.05,
    "podium_ratio":    0.04,
    "earnings_factor": 0.06,
    "age_dist_factor": 0.04,
    "draw_factor":     0.06,
    "human_factor":    0.08,
    "weight_factor":   0.05,
    "going_factor":    0.04,
    "class_factor":    0.06,
    "pace_factor":     0.03,
    "specialist_factor":0.05,
}

# =============================================================================
# CLASSES ET FONCTIONS UTILITAIRES
# =============================================================================

@lru_cache(maxsize=128)
def cached_parse_music(music_str: str) -> dict:
    """Version mise en cache du parsing de musique (optimisation)."""
    return parse_music(music_str)

def parse_music(music_str: str) -> dict:
    """
    Parse la musique avec détection de :
        - poids porté (ex: 1a(57,5kg))
        - pénalité ou changement d'entraîneur (indicateurs textuels)
        - classe de la course (Groupe, Listed, Handicap)
    Retourne un dictionnaire enrichi.
    """
    if not music_str:
        return _debutant_profile()
    clean = music_str.strip().upper()
    if clean in ("", "-", "INEDIT", "INÉDIT", "N/A", "0"):
        return _debutant_profile()

    # Extraction des poids si présents entre parenthèses
    weight_pattern = r"\(([0-9,]+(?:[.,][0-9]?)?)\s*KG?\)"
    weights = re.findall(weight_pattern, clean)
    # Nettoyage des parenthèses pour le parsing des positions
    clean_no_paren = re.sub(r"\([^)]+\)", "", clean)
    tokens = re.findall(r"([0-9DATRP])([AMPHSC]?)", clean_no_paren)

    if not tokens:
        return _debutant_profile()

    raw_scores, numeric_positions, race_types_seen = [], [], []
    weight_penalty = 0.0
    for pos_char, rtype_char in tokens:
        rtype = rtype_char.lower() if rtype_char else "a"
        pos_score = MUSIC_POSITION_SCORES.get(pos_char, 0.2)
        type_weight = MUSIC_RACE_TYPE_WEIGHTS.get(rtype, 1.0)
        raw_scores.append(pos_score * type_weight)
        if pos_char.isdigit():
            numeric_positions.append(int(pos_char) if pos_char != "0" else 10)
        race_types_seen.append(rtype)

    # Application de pénalité si poids détecté (ex: porté plus que la moyenne)
    if weights:
        try:
            last_weight = float(weights[-1].replace(',', '.'))
            if last_weight > 60:
                weight_penalty = -0.5 * min(1.0, (last_weight - 60) / 10)
        except:
            pass

    n = len(raw_scores)
    decay = np.array([np.exp(-DECAY_FACTOR * i) for i in range(n)])
    decay /= decay.sum()
    weighted_score = float(np.dot(raw_scores, decay)) + weight_penalty

    recent_n = min(3, n)
    recent_scores = raw_scores[:recent_n]
    recent_decay = decay[:recent_n]
    recent_decay = recent_decay / recent_decay.sum()
    recent_form = float(np.dot(recent_scores, recent_decay))

    if len(numeric_positions) >= 2:
        pos_std = float(np.std(numeric_positions))
        regularity = max(0.0, 1.0 - pos_std / 5.0)
    else:
        regularity = 0.50

    # Tendance : amélioration récente
    if n >= 4:
        recent_avg = np.mean(raw_scores[:n//2])
        old_avg = np.mean(raw_scores[n//2:])
        trend = (recent_avg - old_avg) / (abs(old_avg) + 1e-9)
    else:
        trend = 0.0

    return {
        "score":          weighted_score,
        "regularity":     regularity,
        "races_count":    n,
        "avg_position":   float(np.mean(numeric_positions)) if numeric_positions else 5.0,
        "best_position":  int(min(numeric_positions)) if numeric_positions else 10,
        "recent_form":    recent_form,
        "trend":          float(trend),
        "is_debutant":    False,
        "win_ratio":      sum(1 for p in numeric_positions if p == 1) / max(n, 1),
        "podium_ratio":   sum(1 for p in numeric_positions if p <= 3) / max(n, 1),
        "weight_penalty": weight_penalty,
    }

def _debutant_profile() -> dict:
    return {
        "score": 2.5, "regularity": 0.45, "races_count": 0,
        "avg_position": 5.0, "best_position": 10, "recent_form": 2.5,
        "trend": 0.0, "is_debutant": True, "win_ratio": 0.0, "podium_ratio": 0.0,
        "weight_penalty": 0.0,
    }

# =============================================================================
# FACTEURS AVANCÉS
# =============================================================================

def age_distance_factor(age: int, distance: int, race_type: str) -> float:
    age = max(2, min(int(age or 4), 20))
    distance = max(800, min(int(distance or 1600), 8000))
    if race_type == "Plat":
        if age == 2:
            f = 1.0 if distance <= 1600 else 0.65
        elif age == 3:
            f = 1.05
        elif 4 <= age <= 7:
            f = 1.08
        elif age == 8:
            f = 1.00
        else:
            f = max(0.70, 1.0 - (age - 8) * 0.05)
    elif race_type in ("Attelé", "Monté"):
        if age <= 3:
            f = 0.80
        elif 4 <= age <= 9:
            f = 1.05
        elif age == 10:
            f = 1.00
        else:
            f = max(0.75, 1.0 - (age - 10) * 0.04)
    else:
        if age <= 4:
            f = 0.85
        elif 5 <= age <= 10:
            f = 1.05
        elif age == 11:
            f = 1.00
        else:
            f = max(0.72, 1.0 - (age - 11) * 0.04)
    if distance > 3000 and age >= 5:
        f *= 1.04
    return f

def draw_factor(draw: int, race_type: str, distance: int) -> float:
    if race_type != "Plat" or not draw or draw <= 0:
        return 0.0
    draw = min(int(draw), 20)
    base = DRAW_IMPACT_BASE.get(draw, -0.55)
    if distance <= 1400:
        return base * 1.60
    elif distance <= 1800:
        return base * 1.00
    else:
        return base * 0.45

def earnings_factor(earnings: float, races_count: int) -> float:
    if not earnings or earnings <= 0 or races_count <= 0:
        return 0.40
    epr = earnings / max(races_count, 1)
    log_epr = np.log1p(epr)
    return float(min(1.0, log_epr / np.log1p(15000)))

def human_factor(driver_pct: float, trainer_pct: float, driver_recent_pct: float = None, trainer_recent_pct: float = None) -> float:
    d = max(0.001, float(driver_pct or 12.0) / 100.0)
    t = max(0.001, float(trainer_pct or 12.0) / 100.0)
    dr = max(0.001, float(driver_recent_pct or d*100) / 100.0) if driver_recent_pct else d
    tr = max(0.001, float(trainer_recent_pct or t*100) / 100.0) if trainer_recent_pct else t
    combined = np.sqrt(d * t) * 0.7 + np.sqrt(dr * tr) * 0.3
    if d >= 0.25 and t >= 0.20:
        combined *= 1.25
    elif d >= 0.22 or t >= 0.18:
        combined *= 1.12
    elif d >= 0.18 or t >= 0.15:
        combined *= 1.06
    return combined

def weight_factor(weight_kg: float, distance: int, race_type: str) -> float:
    """Impact du poids porté (plat/haies). Plus le poids est élevé, plus la pénalité est forte."""
    if race_type not in ("Plat", "Haies", "Steeple-chase") or not weight_kg or weight_kg <= 0:
        return 0.0
    base = 60.0  # poids de référence
    extra = weight_kg - base
    # Pénalité par kg supplémentaire, atténuée sur les longues distances
    penalty = -0.015 * extra * (1 - 0.2 * (distance - 1600)/2400)
    return max(-0.25, min(0.1, penalty))

def going_factor(going: str, horse_going_pref: str = None) -> float:
    """Adaptation au terrain. Prend en compte une préférence éventuelle du cheval."""
    going_map = {"Lourd": -0.15, "Très souple": -0.08, "Souple": 0.0, "Bon souple": 0.05, "Bon": 0.10, "Ferme": 0.08}
    if going not in going_map:
        return 0.0
    base = going_map[going]
    if horse_going_pref and horse_going_pref.lower() in going.lower():
        base += 0.10
    return base

def class_factor(race_class: str, horse_class_rating: float) -> float:
    """
    Classe de la course : Listed, Groupe, Handicap, Maiden, Réclamer.
    horse_class_rating : note de performance dans des courses similaires (0-100).
    """
    class_bonus = {
        "Groupe I": 0.25, "Groupe II": 0.20, "Groupe III": 0.18,
        "Listed": 0.12, "Handicap": 0.05, "Maiden": -0.05, "Réclamer": -0.10
    }
    bonus = class_bonus.get(race_class, 0.0)
    # Un cheval ayant déjà performé en classe supérieure gère mieux
    if horse_class_rating > 80:
        bonus += 0.08
    elif horse_class_rating < 40:
        bonus -= 0.05
    return bonus

def pace_factor(running_style: str, expected_pace: str) -> float:
    """
    running_style : "F" (front-runner), "P" (prominent), "M" (midfield), "C" (closer)
    expected_pace : "F" (fast), "M" (moderate), "S" (slow)
    """
    # Avantage : meneur sur rythme lent, closers sur rythme rapide
    if running_style == "F" and expected_pace == "S":
        return 0.12
    elif running_style == "C" and expected_pace == "F":
        return 0.15
    elif running_style == "M" and expected_pace == "M":
        return 0.05
    elif (running_style == "F" and expected_pace == "F") or (running_style == "C" and expected_pace == "S"):
        return -0.05
    return 0.0

def specialist_factor(horse: dict, distance: int, surface: str, race_type: str) -> float:
    """Score de spécialisation distance/surface basé sur les performances passées."""
    best_dist = horse.get("best_distance", 0)
    best_surface = horse.get("best_surface", "")
    dist_diff = abs(best_dist - distance) if best_dist > 0 else 300
    if dist_diff <= 200:
        dist_bonus = 0.10
    elif dist_diff <= 400:
        dist_bonus = 0.05
    else:
        dist_bonus = -0.05 * min(1, dist_diff/1000)
    surf_bonus = 0.08 if best_surface and surface and best_surface.lower() in surface.lower() else 0.0
    return dist_bonus + surf_bonus

def market_prob(odds: float, n_runners: int) -> float:
    if not odds or odds <= 1.01:
        return 1.0 / max(n_runners, 2)
    return 1.0 / float(odds)

# =============================================================================
# FEATURE ENGINEERING COMPLET
# =============================================================================

def build_features(horse: dict, race: dict) -> dict:
    race_type = race.get("race_type", "Plat")
    distance = int(race.get("distance", 1600))
    n_runners = int(race.get("n_runners", 10))
    going = race.get("going", "Bon")
    surface = race.get("surface", "Herbe")
    race_class = race.get("race_class", "")
    expected_pace = race.get("expected_pace", "M")

    music = parse_music(horse.get("music", ""))

    # Facteurs additionnels
    w_factor = weight_factor(horse.get("weight_kg", 0), distance, race_type)
    g_factor = going_factor(going, horse.get("going_pref", None))
    c_factor = class_factor(race_class, horse.get("class_rating", 50))
    p_factor = pace_factor(horse.get("running_style", "M"), expected_pace)
    s_factor = specialist_factor(horse, distance, surface, race_type)

    # Facteur humain avec forme récente si disponible
    human = human_factor(
        horse.get("driver_win_pct", 12),
        horse.get("trainer_win_pct", 12),
        horse.get("driver_recent_pct", None),
        horse.get("trainer_recent_pct", None)
    )

    return {
        "number": horse.get("number", 0),
        "name": horse.get("name", ""),
        "odds": float(horse.get("odds", 0)),
        "music_score": music["score"],
        "recent_form": music["recent_form"],
        "regularity": music["regularity"],
        "trend": music["trend"],
        "win_ratio": music["win_ratio"],
        "podium_ratio": music["podium_ratio"],
        "races_count": music["races_count"],
        "is_debutant": music["is_debutant"],
        "age_dist_factor": age_distance_factor(horse.get("age", 4), distance, race_type),
        "draw_factor": draw_factor(horse.get("draw", 0), race_type, distance),
        "earnings_factor": earnings_factor(horse.get("earnings", 0), music["races_count"]),
        "human_factor": human,
        "weight_factor": w_factor,
        "going_factor": g_factor,
        "class_factor": c_factor,
        "pace_factor": p_factor,
        "specialist_factor": s_factor,
        "market_prob": market_prob(horse.get("odds", 0), n_runners),
        "driver_win_pct": horse.get("driver_win_pct", 12),
        "trainer_win_pct": horse.get("trainer_win_pct", 12),
        "earnings": horse.get("earnings", 0),
        "age": horse.get("age", 4),
        "sex": horse.get("sex", ""),
        "draw": horse.get("draw", 0),
        "weight_kg": horse.get("weight_kg", 0),
        "class_rating": horse.get("class_rating", 50),
    }

# =============================================================================
# NORMALISATION AVANCÉE
# =============================================================================

NORM_COLS_ADV = [
    "music_score", "recent_form", "regularity", "trend",
    "win_ratio", "podium_ratio", "earnings_factor",
    "age_dist_factor", "human_factor", "weight_factor",
    "going_factor", "class_factor", "pace_factor", "specialist_factor"
]

def normalize_features(features_list: list) -> list:
    if not features_list:
        return features_list
    df = pd.DataFrame(features_list)
    for col in NORM_COLS_ADV:
        if col not in df.columns:
            continue
        vals = df[col].values.astype(float)
        # Z-score
        std = vals.std()
        if std > 1e-9:
            df[f"{col}_z"] = (vals - vals.mean()) / std
        else:
            df[f"{col}_z"] = 0.0
        # Min-Max [0,1]
        mn, mx = vals.min(), vals.max()
        if mx - mn > 1e-9:
            df[f"{col}_norm"] = (vals - mn) / (mx - mn)
        else:
            df[f"{col}_norm"] = 0.5
    return df.to_dict("records")

# =============================================================================
# SCORE COMPOSITE AVEC POIDS DYNAMIQUES
# =============================================================================

def composite_score_advanced(feat: dict, weights: dict = None) -> float:
    if weights is None:
        weights = WEIGHTS_ADVANCED
    score = 0.0
    score += weights["music_score"] * feat.get("music_score", 3.0)
    score += weights["recent_form"] * feat.get("recent_form", 3.0)
    score += weights["regularity"] * feat.get("regularity", 0.5) * 10.0
    score += weights["trend"] * (feat.get("trend", 0.0) + 1.0) * 5.0
    score += weights["win_ratio"] * feat.get("win_ratio", 0.0) * 20.0
    score += weights["podium_ratio"] * feat.get("podium_ratio", 0.0) * 10.0
    score += weights["earnings_factor"] * feat.get("earnings_factor", 0.4) * 8.0
    score += weights["age_dist_factor"] * feat.get("age_dist_factor", 1.0) * 5.0
    score += weights["draw_factor"] * (feat.get("draw_factor", 0.0) + 1.0) * 5.0
    score += weights["human_factor"] * feat.get("human_factor", 0.12) * 18.0
    score += weights["weight_factor"] * (feat.get("weight_factor", 0.0) + 0.2) * 15.0
    score += weights["going_factor"] * (feat.get("going_factor", 0.0) + 0.1) * 12.0
    score += weights["class_factor"] * (feat.get("class_factor", 0.0) + 0.05) * 10.0
    score += weights["pace_factor"] * (feat.get("pace_factor", 0.0) + 0.05) * 8.0
    score += weights["specialist_factor"] * (feat.get("specialist_factor", 0.0) + 0.1) * 10.0
    return max(0.01, score)

def softmax(scores: np.ndarray, temperature: float = TEMPERATURE_SOFTMAX) -> np.ndarray:
    s = np.array(scores, dtype=float) / temperature
    s -= s.max()
    e = np.exp(s)
    return e / e.sum()

def logit_calibration(raw_probs: np.ndarray) -> np.ndarray:
    eps = 1e-9
    logit = np.log((raw_probs + eps) / (1 - raw_probs + eps))
    logit = logit - logit.mean() * 0.1
    calibrated = 1.0 / (1.0 + np.exp(-logit))
    return calibrated / calibrated.sum()

def bayesian_blend(model_probs: np.ndarray, market_probs: np.ndarray, market_weight: float) -> np.ndarray:
    mp = np.array(market_probs, dtype=float)
    if mp.sum() < 1e-9:
        mp = np.ones(len(model_probs)) / len(model_probs)
    else:
        mp /= mp.sum()
    eps = 1e-9
    lo_model = np.log((model_probs + eps) / (1 - model_probs + eps))
    lo_market = np.log((mp + eps) / (1 - mp + eps))
    lo_blend = (1 - market_weight) * lo_model + market_weight * lo_market
    blended = 1.0 / (1.0 + np.exp(-lo_blend))
    return blended / blended.sum()

# =============================================================================
# SIMULATION MONTE CARLO AVEC CORRÉLATION
# =============================================================================

def monte_carlo_advanced(features_list: list, weights: dict, n_iter: int = DEFAULT_MC_ITERATIONS,
                         market_weight: float = DEFAULT_MARKET_WEIGHT) -> dict:
    n = len(features_list)
    all_probs = np.zeros((n_iter, n))
    win_counts = np.zeros(n)
    base_scores = np.array([composite_score_advanced(f, weights) for f in features_list])

    # Matrice de covariance du bruit (bruit commun + spécifique)
    corr_matrix = np.full((n, n), NOISE_CORRELATION) + np.eye(n) * (1 - NOISE_CORRELATION)
    L = np.linalg.cholesky(corr_matrix)

    for it in range(n_iter):
        # Bruit spécifique par cheval
        spec_noise = np.random.normal(0, NOISE_BASE, n)
        # Bruit commun (impact de la course)
        common_noise = np.random.normal(0, NOISE_BASE * 0.8)
        # Bruit corrélé
        correlated_noise = L @ spec_noise + common_noise
        noisy = base_scores * np.exp(correlated_noise)
        for j, feat in enumerate(features_list):
            if feat.get("is_debutant", False):
                noisy[j] *= np.exp(np.random.normal(0, NOISE_BASE * 0.5))
            elif feat.get("regularity", 0.5) < 0.30:
                noisy[j] *= np.exp(np.random.normal(0, NOISE_BASE * 0.7))
        noisy = np.maximum(noisy, 0.001)
        probs = softmax(noisy)
        all_probs[it] = probs
        winner = int(np.random.choice(n, p=probs))
        win_counts[winner] += 1

    simulated_probs = win_counts / n_iter
    mean_probs = all_probs.mean(axis=0)
    std_probs = all_probs.std(axis=0)
    vol_per_horse = std_probs / (mean_probs + 1e-9)

    # Probabilités de placé (top 2) et de top 3
    place_counts = np.zeros((n_iter, n))
    top3_counts = np.zeros((n_iter, n))
    for it in range(n_iter):
        order = np.argsort(-all_probs[it])
        place_counts[it, order[0]] = 1
        if n > 1:
            place_counts[it, order[1]] = 1
        for k in range(min(3, n)):
            top3_counts[it, order[k]] = 1
    place_probs = place_counts.mean(axis=0)
    top3_probs = top3_counts.mean(axis=0)

    return {
        "simulated_probs": simulated_probs,
        "mean_probs": mean_probs,
        "std_probs": std_probs,
        "vol_per_horse": vol_per_horse,
        "place_probs": place_probs,
        "top3_probs": top3_probs,
    }

# =============================================================================
# MOTEUR PRINCIPAL
# =============================================================================

def run_engine(race_info: dict, horses: list,
               mc_iter: int = DEFAULT_MC_ITERATIONS,
               market_weight: float = DEFAULT_MARKET_WEIGHT,
               value_threshold: float = DEFAULT_VALUE_THRESHOLD) -> dict:

    n_runners = len(horses)
    race_info["n_runners"] = n_runners
    race_type = race_info.get("race_type", "Plat")

    # Construction des features
    feats = [build_features(h, race_info) for h in horses]
    feats = normalize_features(feats)

    # Poids du type de course (on peut affiner)
    weights = WEIGHTS_ADVANCED.copy()
    if race_type == "Attelé":
        weights["draw_factor"] = 0.01
        weights["weight_factor"] = 0.01
    elif race_type == "Monté":
        weights["draw_factor"] = 0.0
        weights["weight_factor"] = 0.02
    elif race_type in ("Haies", "Steeple-chase", "Cross-country"):
        weights["weight_factor"] = 0.09
        weights["draw_factor"] = 0.0

    scores = np.array([composite_score_advanced(f, weights) for f in feats])
    sm_probs = softmax(scores)
    cal_probs = logit_calibration(sm_probs)

    raw_mkt = np.array([f["market_prob"] for f in feats])
    if raw_mkt.sum() < 1e-9:
        raw_mkt = np.ones(n_runners) / n_runners
    norm_mkt = raw_mkt / raw_mkt.sum()

    has_odds = any(h.get("odds", 0) > 1.01 for h in horses)
    if has_odds:
        bayes_probs = bayesian_blend(cal_probs, norm_mkt, market_weight)
    else:
        bayes_probs = cal_probs

    mc = monte_carlo_advanced(feats, weights, n_iter=mc_iter, market_weight=market_weight)

    # Fusion bayésienne + Monte Carlo
    final_probs = 0.60 * bayes_probs + 0.40 * mc["mean_probs"]
    final_probs /= final_probs.sum()
    prob_z = zscore(final_probs)

    # Value bets et mise de Kelly
    value_flags = []
    kelly_stakes = []
    for i in range(n_runners):
        ratio = final_probs[i] / (norm_mkt[i] + 1e-9)
        is_value = (ratio >= value_threshold and final_probs[i] >= 0.03)
        value_flags.append(is_value)
        # Fraction de Kelly (sans biais)
        b = 1.0 / (norm_mkt[i] + 1e-9) - 1  # cote bookmaker nette
        p = final_probs[i]
        q = 1 - p
        if b > 0:
            kelly = (p * b - q) / b
            kelly = max(0.0, min(0.10, kelly))
        else:
            kelly = 0.0
        kelly_stakes.append(kelly)

    results = []
    for i, (feat, horse) in enumerate(zip(feats, horses)):
        ratio = final_probs[i] / (norm_mkt[i] + 1e-9) if norm_mkt[i] > 1e-9 else 1.0
        results.append({
            "rank": 0,
            "number": horse.get("number", i+1),
            "name": horse.get("name", f"Cheval {i+1}"),
            "odds": float(horse.get("odds", 0)),
            "sex": horse.get("sex", ""),
            "age": horse.get("age", 4),
            "model_prob": round(final_probs[i] * 100, 2),
            "market_prob": round(norm_mkt[i] * 100, 2),
            "place_prob": round(mc["place_probs"][i] * 100, 2),
            "top3_prob": round(mc["top3_probs"][i] * 100, 2),
            "composite_score": round(scores[i], 4),
            "music_score": round(feat.get("music_score", 0.0), 2),
            "recent_form": round(feat.get("recent_form", 0.0), 2),
            "regularity": round(feat.get("regularity", 0.0), 2),
            "trend": round(feat.get("trend", 0.0), 3),
            "win_ratio": round(feat.get("win_ratio", 0.0), 3),
            "podium_ratio": round(feat.get("podium_ratio", 0.0), 3),
            "human_factor": round(feat.get("human_factor", 0.0), 4),
            "earnings_factor": round(feat.get("earnings_factor", 0.0), 3),
            "draw_factor": round(feat.get("draw_factor", 0.0), 3),
            "weight_factor": round(feat.get("weight_factor", 0.0), 3),
            "going_factor": round(feat.get("going_factor", 0.0), 3),
            "class_factor": round(feat.get("class_factor", 0.0), 3),
            "pace_factor": round(feat.get("pace_factor", 0.0), 3),
            "specialist_factor": round(feat.get("specialist_factor", 0.0), 3),
            "value_ratio": round(ratio, 2),
            "is_value_bet": value_flags[i],
            "kelly_stake": round(kelly_stakes[i] * 100, 1),
            "is_debutant": feat.get("is_debutant", False),
            "mc_std": round(mc["std_probs"][i] * 100, 2),
            "prob_z": round(prob_z[i], 3),
            "driver_win_pct": feat.get("driver_win_pct", 12),
            "trainer_win_pct": feat.get("trainer_win_pct", 12),
            "earnings": feat.get("earnings", 0),
        })

    results.sort(key=lambda x: x["model_prob"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    bases = results[:2]
    outsiders_pool = [r for r in results[2:] if r["model_prob"] > 2.0]
    outsiders_pool.sort(key=lambda x: x["value_ratio"], reverse=True)
    outsiders = outsiders_pool[:3]

    top6 = [r["number"] for r in results[:min(6, n_runners)]]
    trio_combos = list(combinations(top6, 3))[:12]
    top8 = [r["number"] for r in results[:min(8, n_runners)]]
    quinte_combos = list(combinations(top8, 5))[:12]

    sorted_p = sorted([r["model_prob"] for r in results], reverse=True)
    if len(sorted_p) >= 2:
        gap = sorted_p[0] - sorted_p[1]
        conf_idx = min(100.0, round(45.0 + gap * 2.0, 1))
    else:
        conf_idx = 50.0
    avg_vol = float(mc["vol_per_horse"].mean())
    vol_idx = min(100.0, round(avg_vol * 55.0, 1))

    if has_odds:
        raw_overround = sum(1.0 / h["odds"] for h in horses if h.get("odds", 0) > 1.01)
        overround_pct = round((raw_overround - 1.0) * 100, 1)
    else:
        overround_pct = None

    return {
        "results": results,
        "bases": bases,
        "outsiders": outsiders,
        "trio_combos": trio_combos,
        "quinte_combos": quinte_combos,
        "confidence_idx": conf_idx,
        "volatility_idx": vol_idx,
        "overround_pct": overround_pct,
        "weights": weights,
        "mc": mc,
        "has_odds": has_odds,
    }

# =============================================================================
# ANALYSE TEXTUELLE AUTOMATIQUE (ENRICHIE)
# =============================================================================

def generate_analysis(pred: dict, race: dict) -> str:
    results = pred["results"]
    bases = pred["bases"]
    outsiders = pred["outsiders"]
    conf = pred["confidence_idx"]
    vol = pred["volatility_idx"]
    rt = race.get("race_type", "Plat")
    dist = race.get("distance", 1600)
    nr = race.get("n_runners", len(results))
    going = race.get("going", "Non connu")
    surface = race.get("surface", "Non connue")

    lines = []
    lines.append(f"## 📊 Analyse Quantitative Pro — {rt} — {dist}m — {nr} partants — {going} / {surface}\n")
    lines.append("---\n")

    if conf > 72:
        conf_txt = "**Course nettement hiérarchisée** – favori modèle très solide."
    elif conf > 56:
        conf_txt = "**Course de difficulté intermédiaire** – plusieurs candidats sérieux."
    else:
        conf_txt = "**Course très ouverte** – la hiérarchie est floue, outsiders à surveiller."

    if vol > 62:
        vol_txt = "Volatilité **très élevée** => forte incertitude, mise de précaution."
    elif vol > 38:
        vol_txt = "Volatilité **modérée** – le modèle identifie une structure lisible."
    else:
        vol_txt = "Volatilité **faible** – la course est stable, les favoris ont du crédit."

    lines.append(f"{conf_txt}\n\n{vol_txt}\n\n")

    if bases:
        lines.append("### ⭐ Bases ultra‑solides\n")
        for b in bases:
            vsign = " 🟢 *Value bet*" if b["is_value_bet"] else ""
            lines.append(
                f"- **N°{b['number']} — {b['name']}** : proba modèle **{b['model_prob']}%** | marché {b['market_prob']}% | "
                f"score comp. {b['composite_score']:.4f} | forme {b['recent_form']:.2f}/10 | Kelly recommandé : {b['kelly_stake']}%\n"
            )
        lines.append("\n")

    if outsiders:
        lines.append("### 💎 Outsiders à value potentielle\n")
        for o in outsiders:
            if o["value_ratio"] > 1.35:
                sig = "🔥 **Value forte**"
            elif o["value_ratio"] > 1.18:
                sig = "✅ **Value modérée**"
            else:
                sig = "⚠️ *Value marginale*"
            lines.append(
                f"- **N°{o['number']} — {o['name']}** : {sig} | modèle {o['model_prob']}% vs marché {o['market_prob']}% "
                f"(ratio {o['value_ratio']}x) | cote {o['odds']} | Kelly {o['kelly_stake']}%\n"
            )
        lines.append("\n")

    lines.append("### 📈 Analyse du marché & recommandations\n")
    value_bets = [r for r in results if r["is_value_bet"]]
    if value_bets:
        lines.append("- **Value bets détectés (sous‑cotés)** : " + ", ".join(f"N°{r['number']} {r['name']}" for r in value_bets[:4]) + "\n")
    if pred.get("overround_pct") is not None:
        lines.append(f"- **Overround bookmaker** : {pred['overround_pct']}%\n")
    lines.append("\n### 🎲 Stratégie de mise (Kelly fractionné)\n")
    for r in results[:4]:
        if r["kelly_stake"] > 0:
            lines.append(f"- **N°{r['number']} {r['name']}** : miser **{r['kelly_stade']}%** du bankroll\n")

    lines.append("\n### 🏆 Classement probabiliste top 5\n")
    for r in results[:5]:
        lines.append(f"| {r['rank']} | N°{r['number']} {r['name']} | {r['model_prob']}% | marché {r['market_prob']}% | cote {r['odds']} |\n")
    lines.append("\n### ⚙️ Méthodologie avancée\n")
    lines.append(
        "Modèle composite (15+ facteurs) : parsing musical avancé (poids, classes), gains, forme humaine, spécialiste distance/surface, "
        "adaptation au terrain, rythme de course, indice de classe. "
        "Calibration : Softmax → Logit (Platt) → Bayes log‑odds → Monte Carlo corrélé (3000 itérations). "
        "Détection de value bet avec critère de Kelly fractionné.\n"
    )
    lines.append("\n⚠️ *Outil d'aide à la décision. Pariez de façon responsable.*")
    return "".join(lines)

# =============================================================================
# GRAPHIQUES PLOTLY (rajout de carte de rythme et tendance)
# =============================================================================

DARK_BG = "rgba(10,10,26,0)"
GRID_CLR = "rgba(255,255,255,0.08)"
TEXT_CLR = "#e0e0e0"
GREEN = "#00ff88"
RED = "#ff4d6d"
BLUE = "#4cc9f0"
PURPLE = "#a29bfe"
ORANGE = "#ff9f43"
PALETTE = [GREEN, ORANGE, BLUE, PURPLE, RED, "#ffe66d", "#fd79a8", "#6c5ce7"]

def fig_probabilities(results: list) -> go.Figure:
    df = pd.DataFrame(results).sort_values("model_prob", ascending=True)
    colors = [GREEN if r else (RED if s else BLUE) for r, s in zip(df["is_value_bet"], df["model_prob"] < df["market_prob"]*0.78)]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[f"N°{r} — {n}" for r, n in zip(df["number"], df["name"])],
        x=df["model_prob"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0.5, color="rgba(255,255,255,0.2)")),
        text=[f"  {p:.1f}%" for p in df["model_prob"]],
        textposition="outside",
        name="Probabilité Modèle"
    ))
    if df["market_prob"].sum() > 1:
        fig.add_trace(go.Scatter(
            y=[f"N°{r} — {n}" for r, n in zip(df["number"], df["name"])],
            x=df["market_prob"],
            mode="markers",
            marker=dict(symbol="diamond", size=10, color=ORANGE),
            name="Probabilité Marché"
        ))
    fig.update_layout(
        title="Probabilités de victoire — Modèle Quantitatif",
        xaxis_title="Probabilité (%)",
        yaxis_title="",
        height=max(420, len(results)*48),
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_CLR),
        margin=dict(l=180, r=80, t=60, b=40)
    )
    return fig

def fig_trend_performance(results: list) -> go.Figure:
    """Graphique de tendance récente (forme et score composite)"""
    df = pd.DataFrame(results).sort_values("rank")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["rank"],
        y=df["recent_form"],
        mode="lines+markers",
        name="Forme récente (dernières 3 courses)",
        line=dict(color=GREEN, width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df["rank"],
        y=df["composite_score"] / df["composite_score"].max() * 10,
        mode="lines+markers",
        name="Score composite normalisé",
        line=dict(color=ORANGE, width=2, dash="dot")
    ))
    fig.update_layout(
        title="Tendance de performance (forme récente vs score global)",
        xaxis_title="Classement modèle",
        yaxis_title="Note (/10)",
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_CLR)
    )
    return fig

def fig_radar(results: list) -> go.Figure:
    top = results[:min(5, len(results))]
    cats = ["Musique", "Forme", "Régularité", "Humain", "Spécialiste", "Classe", "Terrain"]
    fig = go.Figure()
    for i, h in enumerate(top):
        vals = [
            min(10.0, h["music_score"]),
            min(10.0, h["recent_form"]),
            h["regularity"] * 10.0,
            min(10.0, h["human_factor"] * 80.0),
            min(10.0, (h["specialist_factor"] + 1) * 5),
            min(10.0, (h["class_factor"] + 0.1) * 50),
            min(10.0, (h["going_factor"] + 0.1) * 50),
        ]
        c = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            name=f"N°{h['number']} {h['name'][:14]}",
            line=dict(color=c, width=2),
            fill="toself",
            fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.10)"
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,10])),
        title="Profil multi‑critères (Top 5)",
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_CLR),
        height=470
    )
    return fig

def fig_mc_distribution(results: list, mc: dict) -> go.Figure:
    fig = go.Figure()
    for i, r in enumerate(results[:6]):
        mu = r["model_prob"] / 100
        sigma = r["mc_std"] / 100
        if sigma < 1e-6:
            continue
        x = np.linspace(max(0, mu - 4*sigma), min(1, mu + 4*sigma), 200)
        y = norm.pdf(x, mu, sigma)
        fig.add_trace(go.Scatter(
            x=x*100, y=y,
            mode="lines",
            name=f"N°{r['number']} {r['name'][:12]}",
            fill="toself",
            line=dict(color=PALETTE[i % len(PALETTE)], width=2)
        ))
    fig.update_layout(
        title="Distribution Monte Carlo des probabilités (Top 6)",
        xaxis_title="Probabilité (%)",
        yaxis_title="Densité",
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_CLR)
    )
    return fig

# =============================================================================
# INTERFACE STREAMLIT AVEC IMPORT/EXPORT
# =============================================================================

def apply_css() -> None:
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #07071a 0%, #0d1b2a 40%, #12192b 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1b2a, #07071a); border-right: 1px solid rgba(0,255,136,0.12); }
    [data-testid="metric-container"] { background: rgba(13,27,42,0.85); border: 1px solid rgba(0,255,136,0.18); border-radius: 10px; padding: 10px 14px; }
    .stButton > button { background: linear-gradient(135deg, #00c896, #00b4d8); color: #000; font-weight: 700; border-radius: 10px; width: 100%; }
    .stButton > button:hover { box-shadow: 0 0 24px rgba(0,200,150,0.55); }
    h1, h2, h3, h4 { color: #e8e8e8 !important; }
    .horse-card { background: linear-gradient(135deg, rgba(13,27,42,0.92), rgba(18,25,43,0.92)); border: 1px solid rgba(255,255,255,0.10); border-radius: 10px; padding: 14px 18px; margin: 6px 0; }
    .horse-card-value { border: 1.5px solid #00ff88; box-shadow: 0 0 18px rgba(0,255,136,0.22); }
    </style>
    """, unsafe_allow_html=True)

def render_header() -> None:
    st.markdown(f"""
    <div style="text-align:center; padding: 22px 0 8px 0;">
        <h1 style="font-size:2.6em; background: linear-gradient(90deg,#00ff88,#00b4d8,#a29bfe); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">🏇 {APP_NAME}</h1>
        <p style="color:#6b7fa3;">Moteur de prédiction quantitatif — Bayes · Monte Carlo · Value Bet (Kelly)</p>
        <p style="color:#3a4a5a;">v{APP_VERSION}</p>
    </div>
    """, unsafe_allow_html=True)

def sidebar_config() -> tuple:
    with st.sidebar:
        st.markdown("### ⚙️ Configuration avancée")
        mc_iter = st.slider("Itérations Monte Carlo", 1000, 10000, DEFAULT_MC_ITERATIONS, 500)
        mw = st.slider("Poids bayésien du marché", 0.0, 0.7, DEFAULT_MARKET_WEIGHT, 0.05)
        vt = st.slider("Seuil Value Bet (ratio)", 1.05, 1.80, DEFAULT_VALUE_THRESHOLD, 0.05)
        st.markdown("---")
        st.markdown("### 📁 Import/Export")
        uploaded_file = st.file_uploader("Importer une course (JSON)", type=["json"])
        if uploaded_file:
            data = json.load(uploaded_file)
            st.session_state["imported_race"] = data
            st.success("Course importée ! Utilisez l'onglet saisie pour charger.")
        if st.button("Exporter la course en JSON"):
            if "prediction" in st.session_state:
                export = {"race_info": st.session_state["race_info"], "horses": st.session_state["horses_input"]}
                st.download_button("Télécharger JSON", data=json.dumps(export, indent=2), file_name="course.json")
        st.markdown("---")
        st.markdown("### 📖 Légende enrichie")
        st.markdown("🟢 Value bet (Kelly) · 🔴 Surpayé · ⭐ Base · 💎 Outsider · 📈 Tendance haussière · ⚖️ Poids · 🌾 Terrain")
    return mc_iter, mw, vt

def tab_saisie(race_types_list: list) -> tuple:
    st.markdown("## 🏁 Informations de la course")
    if "imported_race" in st.session_state:
        imp = st.session_state["imported_race"]
        st.info("Données importées détectées. Vous pouvez modifier ci-dessous.")
        race_info_default = imp.get("race_info", {})
        horses_default = imp.get("horses", [])
    else:
        race_info_default = {}
        horses_default = []

    with st.form("race_form"):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            race_type = st.selectbox("Type", race_types_list, index=race_types_list.index(race_info_default.get("race_type", "Plat")) if race_info_default.get("race_type") in race_types_list else 0)
        with c2:
            distance = st.number_input("Distance (m)", 800, 7200, race_info_default.get("distance", 1600), 100)
        with c3:
            going = st.selectbox("Terrain", GOINGS, index=GOINGS.index(race_info_default.get("going", "Bon")) if race_info_default.get("going") in GOINGS else 5)
        with c4:
            surface = st.selectbox("Surface", SURFACES, index=SURFACES.index(race_info_default.get("surface", "Herbe")) if race_info_default.get("surface") in SURFACES else 1)
        with c5:
            race_class = st.selectbox("Classe", ["Maiden", "Réclamer", "Handicap", "Listed", "Groupe III", "Groupe II", "Groupe I"], index=3)
        with c6:
            expected_pace = st.selectbox("Rythme attendu", ["Lent", "Modéré", "Rapide"], index=1)
        st.markdown("---")
        st.markdown("## 🐎 Saisie des partants")
        n_horses = st.number_input("Nombre de partants", 2, 20, len(horses_default) if horses_default else 10)

        horses_input = []
        for i in range(int(n_horses)):
            with st.expander(f"🐎 Partant {i+1}", expanded=(i<2)):
                cols = st.columns(5)
                default = horses_default[i] if i < len(horses_default) else {}
                with cols[0]:
                    num = st.number_input("N°", 1, 30, default.get("number", i+1), key=f"num_{i}")
                    name = st.text_input("Nom", default.get("name", f"Cheval {i+1}"), key=f"name_{i}")
                with cols[1]:
                    age = st.number_input("Âge", 2, 20, default.get("age", 4), key=f"age_{i}")
                    sex = st.selectbox("Sexe", ["H","F","G","M","E"], index=["H","F","G","M","E"].index(default.get("sex","H")), key=f"sex_{i}")
                with cols[2]:
                    odds = st.number_input("Cote", 0.0, 999.0, default.get("odds", 5.0), 0.5, key=f"odds_{i}")
                    weight = st.number_input("Poids (kg)", 0.0, 80.0, default.get("weight_kg", 0.0), 0.5, key=f"weight_{i}")
                with cols[3]:
                    driver = st.number_input("% Driver", 0.0, 100.0, default.get("driver_win_pct", 12.0), 0.5, key=f"drv_{i}")
                    trainer = st.number_input("% Entraîneur", 0.0, 100.0, default.get("trainer_win_pct", 12.0), 0.5, key=f"trn_{i}")
                with cols[4]:
                    music = st.text_input("Musique", default.get("music", ""), key=f"mus_{i}")
                    draw = st.number_input("Corde", 0, 30, default.get("draw", 0), key=f"draw_{i}")
                # Advanced fields (collapsed)
                with st.expander("🔧 Paramètres avancés (spécialiste, etc.)"):
                    best_dist = st.number_input("Meilleure distance (m)", 0, 8000, default.get("best_distance", 0), key=f"bestd_{i}")
                    best_surf = st.selectbox("Meilleure surface", SURFACES, index=SURFACES.index(default.get("best_surface", "Non connue")) if default.get("best_surface") in SURFACES else 0, key=f"bestsurf_{i}")
                    going_pref = st.selectbox("Terrain préféré", GOINGS, index=GOINGS.index(default.get("going_pref", "Bon")) if default.get("going_pref") in GOINGS else 5, key=f"goingpref_{i}")
                    running_style = st.selectbox("Style de course", ["F", "P", "M", "C"], index=["F","P","M","C"].index(default.get("running_style","M")), key=f"style_{i}")
                    class_rating = st.slider("Rating classe (0-100)", 0, 100, default.get("class_rating", 50), key=f"classr_{i}")
                horses_input.append({
                    "number": num, "name": name, "age": age, "sex": sex,
                    "odds": odds, "weight_kg": weight,
                    "driver_win_pct": driver, "trainer_win_pct": trainer,
                    "music": music, "draw": draw,
                    "best_distance": best_dist, "best_surface": best_surf,
                    "going_pref": going_pref, "running_style": running_style,
                    "class_rating": class_rating,
                })

        submitted = st.form_submit_button("🚀 ANALYSER LA COURSE", use_container_width=True)
        race_info = {
            "race_type": race_type, "distance": distance, "going": going, "surface": surface,
            "race_class": race_class, "expected_pace": expected_pace.lower()[0],
            "n_runners": n_horses
        }
        return race_info, horses_input, n_horses, submitted

def tab_results(pred: dict, race: dict) -> None:
    results = pred["results"]
    st.markdown("## 📊 Indicateurs globaux")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric("🎯 Confiance", f"{pred['confidence_idx']}/100")
    with k2:
        st.metric("📈 Volatilité", f"{pred['volatility_idx']}/100")
    with k3:
        st.metric("🐎 Partants", len(results))
    with k4:
        st.metric("💎 Value bets", sum(1 for r in results if r["is_value_bet"]))
    with k5:
        top = results[0] if results else {}
        st.metric("🏆 Favori", f"N°{top.get('number','-')}", f"{top.get('model_prob',0):.1f}%")
    with k6:
        st.metric("📉 Overround", f"{pred.get('overround_pct','N/A')}%")
    st.markdown("---")
    st.markdown("## 🏆 Classement probabiliste complet")
    df_disp = pd.DataFrame(results)[["rank","number","name","age","odds","model_prob","market_prob","place_prob","value_ratio","is_value_bet","kelly_stake"]]
    df_disp.columns = ["Rg","N°","Nom","Âge","Cote","P.Modele%","P.Marché%","P.Place%","Ratio","Value","Kelly%"]
    st.dataframe(df_disp.style.apply(lambda row: ["background-color:rgba(0,255,136,0.10)"]*len(row) if row["Value"] else [""]*len(row), axis=1), use_container_width=True, hide_index=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_probabilities(results), use_container_width=True)
    with col2:
        st.plotly_chart(fig_trend_performance(results), use_container_width=True)
    st.plotly_chart(fig_radar(results), use_container_width=True)
    st.plotly_chart(fig_mc_distribution(results, pred["mc"]), use_container_width=True)
    st.markdown("---")
    st.markdown("## 🧠 Analyse automatique")
    st.markdown(generate_analysis(pred, race))

def main() -> None:
    st.set_page_config(page_title=f"🏇 {APP_NAME}", layout="wide", initial_sidebar_state="expanded")
    apply_css()
    render_header()
    mc_iter, market_w, value_t = sidebar_config()
    tab1, tab2 = st.tabs(["📥 Saisie des données", "📊 Analyse & Prédictions"])
    with tab1:
        race_info, horses_input, _, submitted = tab_saisie(RACE_TYPES)
        if submitted:
            if len(horses_input) < 2:
                st.error("Au moins 2 partants requis.")
                return
            with st.spinner("Analyse en cours…"):
                pred = run_engine(race_info, horses_input, mc_iter=mc_iter, market_weight=market_w, value_threshold=value_t)
                st.session_state["prediction"] = pred
                st.session_state["race_info"] = race_info
                st.session_state["horses_input"] = horses_input
                st.success("Analyse terminée ! Rendez-vous dans l'onglet 'Analyse & Prédictions'.")
    with tab2:
        if "prediction" not in st.session_state:
            st.info("Aucune course analysée. Saisissez les données et cliquez sur 'ANALYSER'.")
        else:
            tab_results(st.session_state["prediction"], st.session_state["race_info"])

if __name__ == "__main__":
    main()
