import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import zscore, norm
from itertools import combinations, permutations
import re
import time
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION GLOBALE & CONSTANTES
# =============================================================================

APP_VERSION = "2.0.0"
APP_NAME    = "QuantTurf Pro"

RACE_TYPES = ["Plat", "Attelé", "Monté", "Haies", "Steeple-chase", "Cross-country"]

# Scores par position dans la musique
MUSIC_POSITION_SCORES: dict = {
    "1": 10.0, "2": 7.5, "3": 5.5, "4": 4.0, "5": 3.0,
    "6": 2.0,  "7": 1.5, "8": 1.0, "9": 0.5, "0": 0.2,
    "D": -2.0, "A": -1.5, "T": -1.5, "R": -1.0, "P": 0.3,
}

# Pondération par type de course dans la musique
MUSIC_RACE_TYPE_WEIGHTS: dict = {
    "a": 1.00,  # Attelé
    "m": 0.90,  # Monté
    "p": 1.00,  # Plat
    "h": 0.95,  # Haies
    "s": 0.90,  # Steeple
    "c": 0.85,  # Cross
    "x": 1.00,  # Inconnu / générique
}

# Impact du numéro de corde (plat) — pondéré par distance
DRAW_IMPACT_BASE: dict = {
    1: 0.35, 2: 0.40, 3: 0.35, 4: 0.25, 5: 0.15,
    6: 0.05, 7: -0.05, 8: -0.12, 9: -0.18, 10: -0.24,
    11: -0.30, 12: -0.35, 13: -0.40, 14: -0.44, 15: -0.48,
    16: -0.50, 17: -0.52, 18: -0.54, 19: -0.55, 20: -0.55,
}

DEFAULT_MC_ITERATIONS   = 2000
DEFAULT_MARKET_WEIGHT   = 0.35
DEFAULT_VALUE_THRESHOLD = 1.15
TEMPERATURE_SOFTMAX     = 1.5
NOISE_BASE              = 0.15

# =============================================================================
# SECTION 1 — PARSING DE LA MUSIQUE
# =============================================================================

def parse_music(music_str: str) -> dict:
    """
    Parse la musique d'un cheval (format PMU/turf français).
    Retourne un dictionnaire de métriques numériques.

    Format attendu : séquences position+type  ex: "1a2p3a0m(4a2a)"
    - Chiffre  = position (0 = non classé >9)
    - Lettre   = type de course (a/m/p/h/s/c)
    - D/A/T/R  = codes spéciaux (disq./abandonné/tombe/refusé)
    """
    if not music_str:
        return _debutant_profile()
    clean = music_str.strip().upper()
    if clean in ("", "-", "INEDIT", "INÉDIT", "N/A", "0"):
        return _debutant_profile()

    # Supprime parenthèses et espaces
    clean = re.sub(r"[() ]", "", clean)

    # Tokenisation : un token = position (chiffre ou code) + type optionnel
    tokens = re.findall(r"([0-9DATRP])([AMPHSC]?)", clean)

    if not tokens:
        return _debutant_profile()

    raw_scores, numeric_positions, race_types_seen = [], [], []

    for pos_char, rtype_char in tokens:
        rtype = rtype_char.lower() if rtype_char else "a"
        pos_score  = MUSIC_POSITION_SCORES.get(pos_char, 0.3)
        type_weight = MUSIC_RACE_TYPE_WEIGHTS.get(rtype, 1.0)
        raw_scores.append(pos_score * type_weight)
        if pos_char.isdigit():
            numeric_positions.append(int(pos_char) if pos_char != "0" else 10)
        race_types_seen.append(rtype)

    n = len(raw_scores)

    # Pondération exponentielle décroissante (course[0] = la plus récente)
    decay = np.array([np.exp(-0.30 * i) for i in range(n)])
    decay /= decay.sum()

    weighted_score = float(np.dot(raw_scores, decay))

    # Forme récente (3 dernières courses)
    recent_n      = min(3, n)
    recent_scores = raw_scores[:recent_n]
    recent_decay  = decay[:recent_n]
    recent_decay  = recent_decay / recent_decay.sum()
    recent_form   = float(np.dot(recent_scores, recent_decay))

    # Régularité (1 - std/5 normalisée)
    if len(numeric_positions) >= 2:
        pos_std   = float(np.std(numeric_positions))
        regularity = max(0.0, 1.0 - pos_std / 5.0)
    else:
        regularity = 0.50

    # Tendance (amélioration récente vs ancienne)
    if n >= 4:
        recent_avg = np.mean(raw_scores[:n // 2])
        old_avg    = np.mean(raw_scores[n // 2:])
        trend      = (recent_avg - old_avg) / (abs(old_avg) + 1e-9)
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
    }


def _debutant_profile() -> dict:
    return {
        "score": 3.0, "regularity": 0.50, "races_count": 0,
        "avg_position": 5.0, "best_position": 10, "recent_form": 3.0,
        "trend": 0.0, "is_debutant": True, "win_ratio": 0.0, "podium_ratio": 0.0,
    }

# =============================================================================
# SECTION 2 — FEATURE ENGINEERING
# =============================================================================

def age_distance_factor(age: int, distance: int, race_type: str) -> float:
    age      = max(2, min(int(age or 4), 20))
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
    else:   # Obstacles
        if age <= 4:
            f = 0.85
        elif 5 <= age <= 10:
            f = 1.05
        elif age == 11:
            f = 1.00
        else:
            f = max(0.72, 1.0 - (age - 11) * 0.04)

    # Bonus distance longue pour chevaux expérimentés
    if distance > 3000 and age >= 5:
        f *= 1.04
    return f


def draw_factor(draw: int, race_type: str, distance: int) -> float:
    if race_type != "Plat" or not draw or draw <= 0:
        return 0.0
    draw = min(int(draw), 20)
    base = DRAW_IMPACT_BASE.get(draw, -0.55)
    # Distance courte (<1400m) → impact accru
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
    # Transformation log pour gérer les outliers
    log_epr = np.log1p(epr)
    # Normalisation : moyenne ~2500€/course → log(2501)≈7.8
    return float(min(1.0, log_epr / np.log1p(15000)))


def human_factor(driver_pct: float, trainer_pct: float) -> float:
    d = max(0.001, float(driver_pct or 12.0) / 100.0)
    t = max(0.001, float(trainer_pct or 12.0) / 100.0)
    # Moyenne géométrique pour l'effet combiné
    combined = float(np.sqrt(d * t))
    # Bonus connexion élite
    if d >= 0.25 and t >= 0.20:
        combined *= 1.25
    elif d >= 0.22 or t >= 0.18:
        combined *= 1.12
    elif d >= 0.18 or t >= 0.15:
        combined *= 1.06
    return combined


def market_prob(odds: float, n_runners: int) -> float:
    if not odds or odds <= 1.01:
        return 1.0 / max(n_runners, 2)
    return 1.0 / float(odds)

# =============================================================================
# SECTION 3 — CONSTRUCTION DU VECTEUR DE FEATURES
# =============================================================================

def build_features(horse: dict, race: dict) -> dict:
    race_type  = race.get("race_type", "Plat")
    distance   = int(race.get("distance", 1600))
    n_runners  = int(race.get("n_runners", 10))

    music = parse_music(horse.get("music", ""))

    return {
        # identité
        "number":      horse.get("number", 0),
        "name":        horse.get("name", ""),
        "odds":        float(horse.get("odds", 0)),
        # features brutes
        "music_score": music["score"],
        "recent_form": music["recent_form"],
        "regularity":  music["regularity"],
        "trend":       music["trend"],
        "win_ratio":   music["win_ratio"],
        "podium_ratio":music["podium_ratio"],
        "races_count": music["races_count"],
        "is_debutant": music["is_debutant"],
        "age_dist_factor": age_distance_factor(horse.get("age", 4), distance, race_type),
        "draw_factor":     draw_factor(horse.get("draw", 0), race_type, distance),
        "earnings_factor": earnings_factor(horse.get("earnings", 0), music["races_count"]),
        "human_factor":    human_factor(horse.get("driver_win_pct", 12), horse.get("trainer_win_pct", 12)),
        "market_prob":     market_prob(horse.get("odds", 0), n_runners),
        # stockage original pour UI
        "driver_win_pct":  horse.get("driver_win_pct", 12),
        "trainer_win_pct": horse.get("trainer_win_pct", 12),
        "earnings":        horse.get("earnings", 0),
        "age":             horse.get("age", 4),
        "sex":             horse.get("sex", ""),
        "draw":            horse.get("draw", 0),
    }

# =============================================================================
# SECTION 4 — NORMALISATION
# =============================================================================

NORM_COLS = [
    "music_score", "recent_form", "regularity", "trend",
    "win_ratio", "podium_ratio", "earnings_factor",
    "age_dist_factor", "human_factor",
]


def normalize_features(features_list: list) -> list:
    if not features_list:
        return features_list
    df = pd.DataFrame(features_list)

    for col in NORM_COLS:
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
# SECTION 5 — POIDS DYNAMIQUES PAR TYPE DE COURSE
# =============================================================================

def race_weights(race_type: str) -> dict:
    if race_type == "Plat":
        return {
            "music_score":    0.28,
            "recent_form":    0.18,
            "regularity":     0.07,
            "trend":          0.04,
            "win_ratio":      0.05,
            "podium_ratio":   0.04,
            "earnings_factor":0.08,
            "age_dist_factor":0.07,
            "draw_factor":    0.09,
            "human_factor":   0.10,
        }
    elif race_type == "Attelé":
        return {
            "music_score":    0.30,
            "recent_form":    0.20,
            "regularity":     0.09,
            "trend":          0.05,
            "win_ratio":      0.06,
            "podium_ratio":   0.04,
            "earnings_factor":0.08,
            "age_dist_factor":0.04,
            "draw_factor":    0.01,
            "human_factor":   0.13,
        }
    elif race_type == "Monté":
        return {
            "music_score":    0.27,
            "recent_form":    0.18,
            "regularity":     0.09,
            "trend":          0.05,
            "win_ratio":      0.06,
            "podium_ratio":   0.04,
            "earnings_factor":0.08,
            "age_dist_factor":0.04,
            "draw_factor":    0.00,
            "human_factor":   0.19,
        }
    else:  # Haies / Steeple / Cross
        return {
            "music_score":    0.26,
            "recent_form":    0.20,
            "regularity":     0.11,
            "trend":          0.04,
            "win_ratio":      0.05,
            "podium_ratio":   0.05,
            "earnings_factor":0.08,
            "age_dist_factor":0.06,
            "draw_factor":    0.00,
            "human_factor":   0.15,
        }

# =============================================================================
# SECTION 6 — SCORE COMPOSITE
# =============================================================================

def composite_score(feat: dict, weights: dict) -> float:
    score  = weights["music_score"]    * feat.get("music_score", 3.0)
    score += weights["recent_form"]    * feat.get("recent_form", 3.0)
    score += weights["regularity"]     * feat.get("regularity", 0.5) * 10.0
    score += weights["trend"]          * (feat.get("trend", 0.0) + 1.0) * 5.0
    score += weights["win_ratio"]      * feat.get("win_ratio", 0.0) * 20.0
    score += weights["podium_ratio"]   * feat.get("podium_ratio", 0.0) * 10.0
    score += weights["earnings_factor"]* feat.get("earnings_factor", 0.4) * 8.0
    score += weights["age_dist_factor"]* feat.get("age_dist_factor", 1.0) * 5.0
    score += weights["draw_factor"]    * (feat.get("draw_factor", 0.0) + 1.0) * 5.0
    score += weights["human_factor"]   * feat.get("human_factor", 0.12) * 18.0
    return max(0.01, score)

# =============================================================================
# SECTION 7 — SOFTMAX
# =============================================================================

def softmax(scores: np.ndarray, temperature: float = TEMPERATURE_SOFTMAX) -> np.ndarray:
    s = np.array(scores, dtype=float) / temperature
    s -= s.max()
    e = np.exp(s)
    return e / e.sum()

# =============================================================================
# SECTION 8 — RÉGRESSION LOGISTIQUE (LOGIT TRANSFORM)
# =============================================================================

def logit_calibration(raw_probs: np.ndarray) -> np.ndarray:
    """Calibration des probabilités via transformation logit (Platt scaling simplifié)."""
    eps   = 1e-9
    logit = np.log((raw_probs + eps) / (1 - raw_probs + eps))
    # Recentrage léger
    logit = logit - logit.mean() * 0.1
    calibrated = 1.0 / (1.0 + np.exp(-logit))
    return calibrated / calibrated.sum()

# =============================================================================
# SECTION 9 — AJUSTEMENT BAYÉSIEN
# =============================================================================

def bayesian_blend(model_probs: np.ndarray, market_probs: np.ndarray,
                   market_weight: float) -> np.ndarray:
    """
    Fusion bayésienne log-odds :
    log_odds_final = (1-w)*log_odds_model + w*log_odds_market
    """
    mp  = np.array(market_probs, dtype=float)
    if mp.sum() < 1e-9:
        mp = np.ones(len(model_probs)) / len(model_probs)
    else:
        mp /= mp.sum()

    eps = 1e-9
    lo_model  = np.log((model_probs + eps) / (1 - model_probs + eps))
    lo_market = np.log((mp + eps) / (1 - mp + eps))

    lo_blend  = (1 - market_weight) * lo_model + market_weight * lo_market
    blended   = 1.0 / (1.0 + np.exp(-lo_blend))
    return blended / blended.sum()

# =============================================================================
# SECTION 10 — SIMULATION MONTE CARLO
# =============================================================================

def monte_carlo(features_list: list, weights: dict, n_iter: int = DEFAULT_MC_ITERATIONS,
                market_weight: float = DEFAULT_MARKET_WEIGHT) -> dict:
    """
    Simulation Monte Carlo complète :
    - Bruit stochastique adapté à la régularité / débutant
    - Agrégation des probabilités de victoire
    - Calcul de la variance et de l'indice de volatilité par cheval
    """
    n   = len(features_list)
    all_probs   = np.zeros((n_iter, n))
    win_counts  = np.zeros(n)

    base_scores = np.array([composite_score(f, weights) for f in features_list])

    for it in range(n_iter):
        noisy = base_scores.copy()
        for j, feat in enumerate(features_list):
            ns = NOISE_BASE
            if feat.get("is_debutant", False):
                ns *= 2.20
            elif feat.get("regularity", 0.5) < 0.30:
                ns *= 1.60
            elif feat.get("regularity", 0.5) > 0.80:
                ns *= 0.70
            # Bruit multiplicatif log-normal
            noisy[j] = base_scores[j] * np.exp(np.random.normal(0, ns))
        noisy = np.maximum(noisy, 0.001)
        probs = softmax(noisy)
        all_probs[it] = probs
        winner = int(np.random.choice(n, p=probs))
        win_counts[winner] += 1

    simulated_probs = win_counts / n_iter
    mean_probs      = all_probs.mean(axis=0)
    std_probs       = all_probs.std(axis=0)
    vol_per_horse   = std_probs / (mean_probs + 1e-9)

    # Distribution du top-2 (placé)
    place_counts = np.zeros((n_iter, n))
    for it in range(n_iter):
        order = np.argsort(-all_probs[it])
        place_counts[it, order[0]] = 1
        if n > 1:
            place_counts[it, order[1]] = 1
    place_probs = place_counts.mean(axis=0)

    return {
        "simulated_probs": simulated_probs,
        "mean_probs":      mean_probs,
        "std_probs":       std_probs,
        "vol_per_horse":   vol_per_horse,
        "place_probs":     place_probs,
    }

# =============================================================================
# SECTION 11 — MOTEUR PRINCIPAL
# =============================================================================

def run_engine(race_info: dict, horses: list,
               mc_iter: int        = DEFAULT_MC_ITERATIONS,
               market_weight: float = DEFAULT_MARKET_WEIGHT,
               value_threshold: float = DEFAULT_VALUE_THRESHOLD) -> dict:

    n_runners = len(horses)
    race_info["n_runners"] = n_runners
    race_type = race_info.get("race_type", "Plat")

    # ── Features ──────────────────────────────────────────────
    feats = [build_features(h, race_info) for h in horses]
    feats = normalize_features(feats)

    # ── Poids ─────────────────────────────────────────────────
    weights = race_weights(race_type)

    # ── Scores composites ─────────────────────────────────────
    scores = np.array([composite_score(f, weights) for f in feats])

    # ── Softmax ───────────────────────────────────────────────
    sm_probs = softmax(scores)

    # ── Calibration logit ─────────────────────────────────────
    cal_probs = logit_calibration(sm_probs)

    # ── Probabilités marché ───────────────────────────────────
    raw_mkt = np.array([f["market_prob"] for f in feats])
    if raw_mkt.sum() < 1e-9:
        raw_mkt = np.ones(n_runners) / n_runners
    norm_mkt = raw_mkt / raw_mkt.sum()

    # ── Ajustement bayésien ───────────────────────────────────
    has_odds = any(h.get("odds", 0) > 1.01 for h in horses)
    if has_odds:
        bayes_probs = bayesian_blend(cal_probs, norm_mkt, market_weight)
    else:
        bayes_probs = cal_probs

    # ── Monte Carlo ───────────────────────────────────────────
    mc = monte_carlo(feats, weights, n_iter=mc_iter, market_weight=market_weight)

    # ── Fusion finale (Bayésien 55% + MC 45%) ─────────────────
    final_probs = 0.55 * bayes_probs + 0.45 * mc["mean_probs"]
    final_probs /= final_probs.sum()

    # ── Z-score des probabilités finales ──────────────────────
    prob_z = zscore(final_probs)

    # ── Value bets ────────────────────────────────────────────
    value_flags = []
    for i in range(n_runners):
        ratio = final_probs[i] / (norm_mkt[i] + 1e-9)
        value_flags.append(ratio >= value_threshold and final_probs[i] >= 0.04)

    # ── Construction des résultats ────────────────────────────
    results = []
    for i, (feat, horse) in enumerate(zip(feats, horses)):
        ratio = final_probs[i] / (norm_mkt[i] + 1e-9) if norm_mkt[i] > 1e-9 else 1.0
        results.append({
            "rank":           0,
            "number":         horse.get("number", i + 1),
            "name":           horse.get("name", f"Cheval {i+1}"),
            "odds":           float(horse.get("odds", 0)),
            "sex":            horse.get("sex", ""),
            "age":            horse.get("age", 4),
            "model_prob":     round(float(final_probs[i]) * 100, 2),
            "market_prob":    round(float(norm_mkt[i]) * 100, 2),
            "place_prob":     round(float(mc["place_probs"][i]) * 100, 2),
            "composite_score":round(float(scores[i]), 4),
            "music_score":    round(feat.get("music_score", 0.0), 2),
            "recent_form":    round(feat.get("recent_form", 0.0), 2),
            "regularity":     round(feat.get("regularity", 0.0), 2),
            "trend":          round(feat.get("trend", 0.0), 3),
            "win_ratio":      round(feat.get("win_ratio", 0.0), 3),
            "podium_ratio":   round(feat.get("podium_ratio", 0.0), 3),
            "human_factor":   round(feat.get("human_factor", 0.0), 4),
            "earnings_factor":round(feat.get("earnings_factor", 0.0), 3),
            "draw_factor":    round(feat.get("draw_factor", 0.0), 3),
            "value_ratio":    round(float(ratio), 2),
            "is_value_bet":   value_flags[i],
            "is_debutant":    feat.get("is_debutant", False),
            "mc_std":         round(float(mc["std_probs"][i]) * 100, 2),
            "prob_z":         round(float(prob_z[i]), 3),
            "driver_win_pct": feat.get("driver_win_pct", 12),
            "trainer_win_pct":feat.get("trainer_win_pct", 12),
            "earnings":       feat.get("earnings", 0),
        })

    # Tri par probabilité modèle décroissante
    results.sort(key=lambda x: x["model_prob"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    # ── Bases ultra-solides (top 2) ───────────────────────────
    bases = results[:2]

    # ── Outsiders à value (rang 3+, meilleur ratio) ───────────
    outsiders_pool = [r for r in results[2:] if r["model_prob"] > 2.5]
    outsiders_pool.sort(key=lambda x: x["value_ratio"], reverse=True)
    outsiders = outsiders_pool[:3]

    # ── Combinaisons Trio (meilleurs 6, 10 combos) ────────────
    top6 = [r["number"] for r in results[:min(6, n_runners)]]
    trio_combos = list(combinations(top6, 3))[:10]

    # ── Combinaisons Quinté (meilleurs 8, 10 combos) ──────────
    top8 = [r["number"] for r in results[:min(8, n_runners)]]
    quinte_combos = list(combinations(top8, 5))[:10]

    # ── Indice de confiance ───────────────────────────────────
    sorted_p = sorted([r["model_prob"] for r in results], reverse=True)
    if len(sorted_p) >= 2:
        gap        = sorted_p[0] - sorted_p[1]
        conf_idx   = min(100.0, round(45.0 + gap * 2.2, 1))
    else:
        conf_idx = 50.0

    # ── Indice de volatilité ──────────────────────────────────
    avg_vol  = float(mc["vol_per_horse"].mean())
    vol_idx  = min(100.0, round(avg_vol * 55.0, 1))

    # ── Overround ─────────────────────────────────────────────
    if has_odds:
        raw_overround = sum(1.0 / h["odds"] for h in horses if h.get("odds", 0) > 1.01)
        overround_pct = round((raw_overround - 1.0) * 100, 1)
    else:
        overround_pct = None

    return {
        "results":        results,
        "bases":          bases,
        "outsiders":      outsiders,
        "trio_combos":    trio_combos,
        "quinte_combos":  quinte_combos,
        "confidence_idx": conf_idx,
        "volatility_idx": vol_idx,
        "overround_pct":  overround_pct,
        "weights":        weights,
        "mc":             mc,
        "has_odds":       has_odds,
    }

# =============================================================================
# SECTION 12 — ANALYSE TEXTUELLE AUTOMATIQUE
# =============================================================================

def generate_analysis(pred: dict, race: dict) -> str:
    results   = pred["results"]
    bases     = pred["bases"]
    outsiders = pred["outsiders"]
    conf      = pred["confidence_idx"]
    vol       = pred["volatility_idx"]
    rt        = race.get("race_type", "Plat")
    dist      = race.get("distance", 1600)
    nr        = race.get("n_runners", len(results))
    lines     = []

    # Titre
    lines.append(f"## 📊 Analyse Quantitative — {rt} — {dist}m — {nr} partants\n")
    lines.append("---\n")

    # Lecture de course
    if conf > 72:
        conf_txt = "**Course nettement hiérarchisée** — le favori modèle est solidement identifié."
    elif conf > 56:
        conf_txt = "**Course de difficulté intermédiaire** — quelques candidats sérieux émergent."
    else:
        conf_txt = "**Course très ouverte** — la hiérarchie est incertaine, les outsiders ont leur chance."

    if vol > 62:
        vol_txt = "La volatilité stochastique est **très élevée**, synonyme de forte incertitude."
    elif vol > 38:
        vol_txt = "La volatilité est **modérée** — quelques aléas possibles mais le cadre reste lisible."
    else:
        vol_txt = "La volatilité est **faible** — le modèle identifie une course structurellement stable."

    lines.append(f"{conf_txt}\n\n{vol_txt}\n\n")

    # Bases
    if bases:
        lines.append("### ⭐ Bases recommandées\n")
        for b in bases:
            vsign = " 🟢 *Value bet confirmé*" if b["is_value_bet"] else ""
            lines.append(
                f"- **N°{b['number']} — {b['name']}** : probabilité modèle **{b['model_prob']}%**"
                f" | marché {b['market_prob']}% | score composite {b['composite_score']:.4f}"
                f" | forme récente {b['recent_form']:.2f}/10 | régularité {b['regularity']:.2f}"
                f"{vsign}\n"
            )
        lines.append("\n")

    # Outsiders
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
                f"- **N°{o['number']} — {o['name']}** : {sig} | "
                f"modèle {o['model_prob']}% vs marché {o['market_prob']}% "
                f"(ratio {o['value_ratio']}x) — cote {o['odds']}\n"
            )
        lines.append("\n")

    # Analyse marché
    lines.append("### 📈 Analyse du marché\n")
    overbet   = [r for r in results if r["model_prob"] < r["market_prob"] * 0.75 and r["market_prob"] > 5]
    value_bets = [r for r in results if r["is_value_bet"]]

    if overbet:
        nms = ", ".join(f"N°{r['number']} *{r['name']}*" for r in overbet[:4])
        lines.append(f"- **Chevaux surcotés par le marché (surpariés)** : {nms}\n")
    if value_bets:
        nms = ", ".join(f"N°{r['number']} *{r['name']}*" for r in value_bets[:4])
        lines.append(f"- **Value bets détectés (sous-cotés)** : {nms}\n")
    if pred.get("overround_pct") is not None:
        lines.append(f"- **Overround bookmaker estimé** : {pred['overround_pct']}%\n")
    lines.append("\n")

    # Top 5 rapide
    lines.append("### 🏆 Classement probabiliste top 5\n")
    for r in results[:5]:
        mc_range = f"±{r['mc_std']:.1f}%"
        debutant = " *(débutant)*" if r["is_debutant"] else ""
        lines.append(
            f"| **{r['rank']}** | N°{r['number']} {r['name']}{debutant} | "
            f"{r['model_prob']}% | {r['market_prob']}% | cote {r['odds']} | MC {mc_range} |\n"
        )
    lines.append("\n")

    # Note méthodologique
    lines.append("### ⚙️ Méthodologie\n")
    lines.append(
        "Modèle composite fondé sur : **parsing de la musique** (pondération exponentielle décroissante, "
        "régularité, tendance, ratio victoire/podium), **facteurs humains** (driver × entraîneur, "
        "interaction multiplicative), **gains/expérience** (log-transform), **adaptation âge/distance**, "
        "**numéro de corde** (plat uniquement). "
        "Probabilités calibrées via **Softmax → Logit (Platt) → Ajustement bayésien log-odds → "
        f"Monte Carlo ({DEFAULT_MC_ITERATIONS} itérations)**. "
        f"**Indice de confiance : {conf}/100** | **Indice de volatilité : {vol}/100**\n"
    )
    lines.append("\n⚠️ *Outil d'aide à la décision uniquement. Pariez de manière responsable.*\n")

    return "".join(lines)

# =============================================================================
# SECTION 13 — GRAPHIQUES PLOTLY
# =============================================================================

DARK_BG   = "rgba(10,10,26,0)"
GRID_CLR  = "rgba(255,255,255,0.08)"
TEXT_CLR  = "#e0e0e0"
GREEN     = "#00ff88"
RED       = "#ff4d6d"
BLUE      = "#4cc9f0"
PURPLE    = "#a29bfe"
ORANGE    = "#ff9f43"
PALETTE   = [GREEN, ORANGE, BLUE, PURPLE, RED, "#ffe66d", "#fd79a8", "#6c5ce7"]


def fig_probabilities(results: list) -> go.Figure:
    df = pd.DataFrame(results).sort_values("model_prob", ascending=True)
    colors = [GREEN if r else (RED if s else BLUE)
              for r, s in zip(df["is_value_bet"], df["model_prob"] < df["market_prob"] * 0.78)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[f"N°{r} — {n}" for r, n in zip(df["number"], df["name"])],
        x=df["model_prob"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0.5, color="rgba(255,255,255,0.2)")),
        text=[f"  {p:.1f}%" for p in df["model_prob"]],
        textposition="outside",
        textfont=dict(color=TEXT_CLR, size=12),
        name="Probabilité Modèle",
        hovertemplate="<b>%{y}</b><br>Probabilité: %{x:.2f}%<extra></extra>",
    ))
    # Marqueurs marché
    if df["market_prob"].sum() > 1:
        fig.add_trace(go.Scatter(
            y=[f"N°{r} — {n}" for r, n in zip(df["number"], df["name"])],
            x=df["market_prob"],
            mode="markers",
            marker=dict(symbol="diamond", size=10, color=ORANGE,
                        line=dict(width=1, color="white")),
            name="Probabilité Marché",
            hovertemplate="<b>%{y}</b><br>Marché: %{x:.2f}%<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text="Probabilités de victoire — Modèle Quantitatif", font=dict(size=15, color=TEXT_CLR)),
        xaxis=dict(title="Probabilité (%)", gridcolor=GRID_CLR, color=TEXT_CLR),
        yaxis=dict(color=TEXT_CLR, tickfont=dict(size=11)),
        height=max(420, len(results) * 48),
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_CLR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=180, r=80, t=60, b=40),
    )
    return fig


def fig_model_vs_market(results: list) -> go.Figure:
    df    = pd.DataFrame(results)
    mx    = max(df["market_prob"].max(), df["model_prob"].max()) * 1.15
    fig   = go.Figure()

    # Diagonale parité
    fig.add_trace(go.Scatter(
        x=[0, mx], y=[0, mx],
        mode="lines", name="Parité",
        line=dict(color="rgba(255,255,255,0.25)", dash="dash", width=1),
    ))
    # Zone value
    fig.add_shape(type="rect", x0=0, y0=0, x1=mx, y1=mx,
                  fillcolor="rgba(0,255,136,0.03)", line=dict(width=0))

    for _, row in df.iterrows():
        if row["is_value_bet"]:
            color, sym, sz = GREEN, "star", 18
        elif row["model_prob"] < row["market_prob"] * 0.78:
            color, sym, sz = RED, "x", 14
        else:
            color, sym, sz = BLUE, "circle", 13

        fig.add_trace(go.Scatter(
            x=[row["market_prob"]], y=[row["model_prob"]],
            mode="markers+text",
            marker=dict(size=sz, color=color, symbol=sym,
                        line=dict(width=1, color="rgba(255,255,255,0.5)")),
            text=[f"  N°{int(row['number'])}"],
            textposition="middle right",
            textfont=dict(color=TEXT_CLR, size=10),
            showlegend=False,
            hovertemplate=(
                f"<b>N°{int(row['number'])} — {row['name']}</b><br>"
                f"Marché: {row['market_prob']:.2f}%<br>"
                f"Modèle: {row['model_prob']:.2f}%<br>"
                f"Ratio: {row['value_ratio']:.2f}x<extra></extra>"
            ),
        ))

    fig.add_annotation(x=mx * 0.72, y=mx * 0.92,
                       text="▲ Zone VALUE (modèle > marché)",
                       showarrow=False, font=dict(color=GREEN, size=11))
    fig.add_annotation(x=mx * 0.72, y=mx * 0.08,
                       text="▼ Zone SURPAYÉ (marché > modèle)",
                       showarrow=False, font=dict(color=RED, size=11))

    fig.update_layout(
        title=dict(text="Modèle vs Marché — Détection de Value Bet", font=dict(size=15, color=TEXT_CLR)),
        xaxis=dict(title="Probabilité Marché (%)", gridcolor=GRID_CLR, color=TEXT_CLR, range=[0, mx]),
        yaxis=dict(title="Probabilité Modèle (%)", gridcolor=GRID_CLR, color=TEXT_CLR, range=[0, mx]),
        height=500,
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_CLR),
    )
    return fig


def fig_radar(results: list) -> go.Figure:
    top = results[:min(5, len(results))]
    cats = ["Musique", "Forme", "Régularité", "Humain", "Tendance", "Podiums", "Score"]

    fig = go.Figure()
    for i, h in enumerate(top):
        vals = [
            min(10.0, h["music_score"]),
            min(10.0, h["recent_form"]),
            h["regularity"] * 10.0,
            min(10.0, h["human_factor"] * 80.0),
            min(10.0, (h["trend"] + 1.0) * 5.0),
            h["podium_ratio"] * 10.0,
            min(10.0, h["composite_score"] * 1.8),
        ]
        c = PALETTE[i % len(PALETTE)]
        r_vals = vals + [vals[0]]
        t_cats  = cats + [cats[0]]
        try:
            rgb = tuple(int(c.lstrip("#")[j:j+2], 16) for j in (0, 2, 4))
            fill_c = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.10)"
        except Exception:
            fill_c = "rgba(0,255,136,0.10)"

        fig.add_trace(go.Scatterpolar(
            r=r_vals, theta=t_cats,
            name=f"N°{h['number']} {h['name'][:14]}",
            line=dict(color=c, width=2),
            fill="toself",
            fillcolor=fill_c,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], gridcolor=GRID_CLR,
                            tickfont=dict(color=TEXT_CLR, size=9)),
            angularaxis=dict(gridcolor=GRID_CLR, tickfont=dict(color=TEXT_CLR)),
            bgcolor=DARK_BG,
        ),
        title=dict(text="Profil Multi-Critères — Top 5", font=dict(size=15, color=TEXT_CLR)),
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_CLR),
        height=470,
        legend=dict(font=dict(size=10)),
    )
    return fig


def fig_mc_distribution(results: list, mc: dict) -> go.Figure:
    fig = go.Figure()
    for i, r in enumerate(results[:6]):
        mu   = r["model_prob"] / 100
        sigma = r["mc_std"] / 100
        if sigma < 1e-6:
            continue
        x = np.linspace(max(0, mu - 4 * sigma), min(1, mu + 4 * sigma), 200)
        y = norm.pdf(x, mu, sigma)
        c = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(
            x=x * 100, y=y,
            mode="lines",
            name=f"N°{r['number']} {r['name'][:12]}",
            line=dict(color=c, width=2),
            fill="toself",
            fillcolor=c.replace(")", ",0.10)").replace("rgba", "rgba") if "rgba" in c
                      else f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.10)",
        ))

    fig.update_layout(
        title=dict(text="Distribution Monte Carlo des probabilités (top 6)", font=dict(size=15, color=TEXT_CLR)),
        xaxis=dict(title="Probabilité (%)", gridcolor=GRID_CLR, color=TEXT_CLR),
        yaxis=dict(title="Densité", gridcolor=GRID_CLR, color=TEXT_CLR),
        height=400,
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_CLR),
    )
    return fig

# =============================================================================
# SECTION 14 — INTERFACE STREAMLIT
# =============================================================================

def apply_css() -> None:
    st.markdown("""
<style>
/* Global background */
.stApp {
    background: linear-gradient(135deg, #07071a 0%, #0d1b2a 40%, #12192b 100%);
}
/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a, #07071a);
    border-right: 1px solid rgba(0,255,136,0.12);
}
/* Metric cards */
[data-testid="metric-container"] {
    background: rgba(13,27,42,0.85);
    border: 1px solid rgba(0,255,136,0.18);
    border-radius: 10px;
    padding: 10px 14px;
}
/* Expander */
details {
    background: rgba(13,27,42,0.60) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
}
/* Tabs */
[data-baseweb="tab"] {
    background: rgba(13,27,42,0.50);
    border-radius: 8px 8px 0 0;
}
/* Bouton principal */
.stButton > button {
    background: linear-gradient(135deg, #00c896, #00b4d8);
    color: #000;
    font-weight: 700;
    font-size: 1.05em;
    border: none;
    border-radius: 10px;
    padding: 14px 0;
    width: 100%;
    letter-spacing: 0.04em;
    transition: box-shadow 0.25s;
}
.stButton > button:hover {
    box-shadow: 0 0 24px rgba(0,200,150,0.55);
}
/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,255,136,0.12);
    border-radius: 8px;
}
/* Headers */
h1, h2, h3, h4 { color: #e8e8e8 !important; }
/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #07071a; }
::-webkit-scrollbar-thumb { background: rgba(0,255,136,0.35); border-radius: 6px; }
/* Horse card custom */
.horse-card {
    background: linear-gradient(135deg, rgba(13,27,42,0.92), rgba(18,25,43,0.92));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 6px 0;
}
.horse-card-value {
    border: 1.5px solid #00ff88;
    box-shadow: 0 0 18px rgba(0,255,136,0.22);
}
</style>
""", unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(f"""
<div style="text-align:center; padding: 22px 0 8px 0;">
    <h1 style="font-size:2.6em; background: linear-gradient(90deg,#00ff88,#00b4d8,#a29bfe);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:4px;">
        🏇 {APP_NAME}
    </h1>
    <p style="color:#6b7fa3; font-size:1.05em; margin-top:0; letter-spacing:0.05em;">
        Moteur de prédiction quantitatif — Probabilistique · Bayésien · Monte Carlo
    </p>
    <p style="color:#3a4a5a; font-size:0.78em;">v{APP_VERSION}</p>
</div>
""", unsafe_allow_html=True)


def sidebar_config() -> tuple:
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        mc_iter = st.slider("Itérations Monte Carlo", 500, 5000, DEFAULT_MC_ITERATIONS, 250,
                            help="Plus d'itérations = plus précis mais plus lent")
        mw = st.slider("Poids bayésien du marché", 0.00, 0.60, DEFAULT_MARKET_WEIGHT, 0.05,
                       help="Part accordée aux cotes bookmaker (0 = modèle pur)")
        vt = st.slider("Seuil Value Bet (ratio)", 1.05, 1.60, DEFAULT_VALUE_THRESHOLD, 0.05,
                       help="Ratio modèle/marché minimum pour signaler un value bet")
        st.markdown("---")
        st.markdown("### 📖 Légende")
        st.markdown("🟢 Value bet détecté  \n🔴 Cheval surpayé  \n⚪ Neutre  \n⭐ Base  \n💎 Outsider value  \n📈 Tendance haussière")
        st.markdown("---")
        st.markdown("### 🎵 Format musique")
        st.markdown("```\n1a = 1er en attelé\n2p = 2e en plat\n3h = 3e en haies\n0m = non classé monté\nD  = disqualifié\nA  = abandonné\nT  = tombé\n```")
    return mc_iter, mw, vt


def tab_saisie(race_types_list: list) -> tuple:
    st.markdown("## 🏁 Informations de la course")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        race_type = st.selectbox("Type de course", race_types_list)
    with c2:
        distance = st.number_input("Distance (m)", 800, 7200, 1600, 100)
    with c3:
        discipline = st.text_input("Prix / Discipline", "")
    with c4:
        level = st.text_input("Niveau / Catégorie", "", placeholder="Groupe I, Réclamer, Maiden…")

    st.markdown("---")
    st.markdown("## 🐎 Saisie des partants")

    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown("Renseignez les informations complètes de chaque partant.")
    with col_b:
        n_horses = st.number_input("Nombre de partants", 2, 20, 10)

    horses_input = []
    for i in range(int(n_horses)):
        with st.expander(f"🐎 Partant {i+1}", expanded=(i < 2)):
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                hn   = st.number_input("N°", 1, 30, i+1, key=f"n_{i}")
                name = st.text_input("Nom", f"Cheval {i+1}", key=f"nm_{i}")
            with c2:
                sex  = st.selectbox("Sexe", ["H","F","G","M","E"], key=f"sx_{i}",
                                    help="H=Hongre F=Femelle G=Gelding M=Mâle E=Entier")
                age  = st.number_input("Âge", 2, 20, 4, key=f"ag_{i}")
            with c3:
                odds = st.number_input("Cote", 0.0, 999.0, 5.0, 0.5, key=f"od_{i}",
                                       help="0 = non communiquée")
                earn = st.number_input("Gains totaux (€)", 0, 9999999, 0, 500, key=f"er_{i}")
            with c4:
                drv  = st.number_input("% Driver/Jockey", 0.0, 100.0, 12.0, 0.5, key=f"dv_{i}")
                trn  = st.number_input("% Entraîneur", 0.0, 100.0, 12.0, 0.5, key=f"tr_{i}")
            with c5:
                mus  = st.text_input("Musique", "", key=f"mu_{i}",
                                     placeholder="Ex: 1a2a3p0m",
                                     help="Chiffre+lettre, gauche=récent. Laisser vide si inédit.")
                draw = st.number_input("N° corde (plat)", 0, 30, 0, key=f"dr_{i}",
                                       help="0 = non applicable (trot/obstacle)")
            horses_input.append({
                "number": hn, "name": name, "sex": sex, "age": age,
                "odds": odds, "earnings": earn,
                "driver_win_pct": drv, "trainer_win_pct": trn,
                "music": mus, "draw": draw,
            })

    # Récapitulatif
    if horses_input:
        st.markdown("---")
        st.markdown("### 📋 Récapitulatif")
        dfp = pd.DataFrame(horses_input)[
            ["number","name","sex","age","odds","earnings","music","driver_win_pct","trainer_win_pct","draw"]
        ]
        dfp.columns = ["N°","Nom","Sexe","Âge","Cote","Gains €","Musique","% Driver","% Entraîneur","Corde"]
        st.dataframe(dfp, use_container_width=True, hide_index=True)

    race_info = {
        "race_type": race_type, "distance": int(distance),
        "discipline": discipline, "race_level": level,
    }
    return race_info, horses_input, int(n_horses)


def tab_results(pred: dict, race: dict) -> None:
    results = pred["results"]

    # ── KPIs ──────────────────────────────────────────────────
    st.markdown("## 📊 Indicateurs globaux")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    ci  = pred["confidence_idx"]
    vi  = pred["volatility_idx"]
    vb  = sum(1 for r in results if r["is_value_bet"])
    top = results[0] if results else {}

    with k1:
        delta_c = "Élevé 🟢" if ci > 65 else ("Moyen 🟡" if ci > 45 else "Faible 🔴")
        st.metric("🎯 Confiance", f"{ci}/100", delta_c)
    with k2:
        delta_v = "Stable" if vi < 38 else ("Modéré" if vi < 62 else "Volatile")
        st.metric("📈 Volatilité", f"{vi}/100", delta_v)
    with k3:
        st.metric("🐎 Partants", len(results))
    with k4:
        st.metric("💎 Value bets", vb)
    with k5:
        st.metric("🏆 Favori modèle", f"N°{top.get('number','-')} {top.get('name','')[:12]}", f"{top.get('model_prob',0):.1f}%")
    with k6:
        or_ = pred.get("overround_pct")
        st.metric("📉 Overround", f"{or_}%" if or_ is not None else "N/A",
                  help="Marge estimée du bookmaker")

    st.markdown("---")

    # ── Tableau principal ──────────────────────────────────────
    st.markdown("## 🏆 Classement probabiliste complet")
    display_rows = []
    for r in results:
        if r["is_value_bet"]:
            flag = "🟢"
        elif r["model_prob"] < r["market_prob"] * 0.76:
            flag = "🔴"
        else:
            flag = "⚪"
        trend_icon = "📈" if r["trend"] > 0.1 else ("📉" if r["trend"] < -0.1 else "➡️")

        display_rows.append({
            "Rg":        r["rank"],
            "N°":        r["number"],
            "Nom":       r["name"],
            "Sexe":      r["sex"],
            "Âge":       r["age"],
            "Cote":      r["odds"] if r["odds"] > 0 else "-",
            "P.Modèle%": f"{r['model_prob']:.2f}",
            "P.Marché%": f"{r['market_prob']:.2f}",
            "P.Place%":  f"{r['place_prob']:.1f}",
            "Ratio":     f"{r['value_ratio']:.2f}x",
            "Score":     f"{r['composite_score']:.4f}",
            "Musique":   f"{r['music_score']:.2f}",
            "Forme":     f"{r['recent_form']:.2f}",
            "Rég.":      f"{r['regularity']:.2f}",
            "Tend.":     trend_icon,
            "MC±%":      f"±{r['mc_std']:.2f}",
            "Value":     flag,
        })

    df_disp = pd.DataFrame(display_rows)

    def color_rows(row):
        if row["Value"] == "🟢":
            return ["background-color:rgba(0,255,136,0.10)"] * len(row)
        elif row["Value"] == "🔴":
            return ["background-color:rgba(255,77,109,0.09)"] * len(row)
        return [""] * len(row)

    st.dataframe(df_disp.style.apply(color_rows, axis=1),
                 use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Bases & Outsiders ──────────────────────────────────────
    col_base, col_out = st.columns(2)

    with col_base:
        st.markdown("### ⭐ Bases Ultra Solides")
        for b in pred["bases"]:
            vc = "horse-card horse-card-value" if b["is_value_bet"] else "horse-card"
            vt = " 🟢 VALUE" if b["is_value_bet"] else ""
            st.markdown(f"""
<div class="{vc}">
  <h4 style="color:#00ff88;margin:0 0 6px 0;">⭐ N°{b['number']} — {b['name']}{vt}</h4>
  <div style="display:flex;gap:20px;flex-wrap:wrap;color:#aab4c8;font-size:0.9em;">
    <span>Modèle <b style="color:white">{b['model_prob']:.2f}%</b></span>
    <span>Marché {b['market_prob']:.2f}%</span>
    <span>Score {b['composite_score']:.4f}</span>
    <span>Forme {b['recent_form']:.2f}/10</span>
    <span>MC ±{b['mc_std']:.2f}%</span>
  </div>
</div>""", unsafe_allow_html=True)

    with col_out:
        st.markdown("### 💎 Outsiders à Value")
        if pred["outsiders"]:
            for o in pred["outsiders"]:
                col = "#00ff88" if o["value_ratio"] > 1.30 else "#ff9f43"
                st.markdown(f"""
<div class="horse-card">
  <h4 style="color:{col};margin:0 0 6px 0;">💎 N°{o['number']} — {o['name']}</h4>
  <div style="display:flex;gap:20px;flex-wrap:wrap;color:#aab4c8;font-size:0.9em;">
    <span>Modèle <b style="color:white">{o['model_prob']:.2f}%</b></span>
    <span>Marché {o['market_prob']:.2f}%</span>
    <span>Ratio <b style="color:{col}">{o['value_ratio']:.2f}x</b></span>
    <span>Cote {o['odds']}</span>
  </div>
</div>""", unsafe_allow_html=True)
        else:
            st.info("Aucun outsider à value significative détecté avec les paramètres actuels.")

    st.markdown("---")

    # ── Combinaisons ──────────────────────────────────────────
    st.markdown("## 🎯 Sélections de jeu")
    col_trio, col_q = st.columns(2)

    def combo_confidence(combo_nums, results_list, size):
        pool = [r for r in results_list if r["number"] in combo_nums]
        pool.sort(key=lambda x: x["model_prob"], reverse=True)
        probs = [h["model_prob"] for h in pool[:size]]
        return sum(probs) / max(len(probs), 1)

    with col_trio:
        st.markdown("### 🎲 10 Combinaisons Trio")
        trio_rows = []
        for i, combo in enumerate(pred["trio_combos"]):
            sc = combo_confidence(combo, results, 3)
            trio_rows.append({
                "#":          i + 1,
                "Trio":       f"{combo[0]} – {combo[1]} – {combo[2]}",
                "Score moy.": f"{sc:.1f}%",
                "★":          "⭐⭐⭐" if i < 3 else ("⭐⭐" if i < 6 else "⭐"),
            })
        st.dataframe(pd.DataFrame(trio_rows), use_container_width=True, hide_index=True)

    with col_q:
        st.markdown("### 🎲 10 Combinaisons Quinté")
        quinte_rows = []
        for i, combo in enumerate(pred["quinte_combos"]):
            sc = combo_confidence(combo, results, 5)
            quinte_rows.append({
                "#":          i + 1,
                "Quinté":     " – ".join(str(c) for c in combo),
                "Score moy.": f"{sc:.1f}%",
                "★":          "⭐⭐⭐" if i < 3 else ("⭐⭐" if i < 6 else "⭐"),
            })
        st.dataframe(pd.DataFrame(quinte_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Graphiques ────────────────────────────────────────────
    st.markdown("## 📈 Visualisations analytiques")
    vt1, vt2, vt3, vt4 = st.tabs(
        ["📊 Probabilités", "🔍 Modèle vs Marché", "🕸 Profil multi-critères", "🎲 Distribution MC"]
    )
    with vt1:
        st.plotly_chart(fig_probabilities(results), use_container_width=True)
    with vt2:
        if pred["has_odds"]:
            st.plotly_chart(fig_model_vs_market(results), use_container_width=True)
        else:
            st.info("Renseignez les cotes pour activer ce graphique.")
    with vt3:
        st.plotly_chart(fig_radar(results), use_container_width=True)
    with vt4:
        st.plotly_chart(fig_mc_distribution(results, pred["mc"]), use_container_width=True)

    st.markdown("---")

    # ── Analyse textuelle ─────────────────────────────────────
    st.markdown("## 🧠 Analyse automatique — Style analyste professionnel")
    st.markdown(generate_analysis(pred, race))

    st.markdown("---")

    # ── Détail des scores ─────────────────────────────────────
    st.markdown("## 🔬 Détail des scores par composante")
    det = []
    for r in results:
        det.append({
            "N°": r["number"], "Nom": r["name"],
            "Musique":   r["music_score"],
            "Forme":     r["recent_form"],
            "Régularité":r["regularity"],
            "Tendance":  r["trend"],
            "Vic.%":     r["win_ratio"],
            "Pod.%":     r["podium_ratio"],
            "Humain":    round(r["human_factor"] * 100, 2),
            "Gains fct": r["earnings_factor"],
            "Draw fct":  r["draw_factor"],
            "Âge/dist":  round(r.get("earnings_factor", 1.0), 3),
            "Composite": r["composite_score"],
            "P.Modèle%": f"{r['model_prob']:.2f}",
            "Z-score":   r["prob_z"],
            "MC±%":      f"±{r['mc_std']:.2f}",
        })
    st.dataframe(pd.DataFrame(det), use_container_width=True, hide_index=True)

    # Footer
    st.markdown("""
<div style="text-align:center;padding:28px 0 10px;color:#334;border-top:1px solid rgba(255,255,255,0.06);margin-top:30px;">
    <p style="color:#3a4a5a;font-size:0.85em;">
        QuantTurf Pro — Softmax · Logit (Platt) · Ajustement Bayésien Log-odds · Monte Carlo
    </p>
    <p style="color:#2a3a4a;font-size:0.75em;">
        ⚠️ Outil d'aide à la décision uniquement. Le jeu peut être dangereux. Jouez de manière responsable.
    </p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SECTION 15 — POINT D'ENTRÉE PRINCIPAL
# =============================================================================

def main() -> None:
    st.set_page_config(
        page_title=f"🏇 {APP_NAME}",
        page_icon="🏇",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_css()
    render_header()

    mc_iter, market_w, value_t = sidebar_config()

    tab1, tab2 = st.tabs(["📥  Saisie des données", "📊  Analyse & Prédictions"])

    with tab1:
        race_info, horses_input, _ = tab_saisie(RACE_TYPES)
        st.markdown("---")
        analyze_btn = st.button("🚀  ANALYSER LA COURSE", use_container_width=True)

        if analyze_btn:
            if len(horses_input) < 2:
                st.error("Veuillez saisir au moins 2 partants.")
                return

            progress = st.progress(0)
            status   = st.empty()

            steps = [
                (12, "🔄  Parsing de la musique & extraction des features…"),
                (25, "📊  Normalisation Z-score & Min-Max…"),
                (40, "🧠  Calcul des scores composites pondérés…"),
                (55, "🎲  Lancement de la simulation Monte Carlo…"),
                (70, "🔮  Calibration Logit & ajustement Bayésien…"),
                (85, "💎  Détection des value bets & génération des combinaisons…"),
                (95, "📈  Calcul des indices de confiance & volatilité…"),
            ]
            for pct, msg in steps:
                status.markdown(f"<small style='color:#6b7fa3'>{msg}</small>", unsafe_allow_html=True)
                progress.progress(pct)
                time.sleep(0.18)

            pred = run_engine(
                race_info, horses_input,
                mc_iter=mc_iter,
                market_weight=market_w,
                value_threshold=value_t,
            )

            progress.progress(100)
            status.empty()
            progress.empty()

            st.session_state["prediction"]   = pred
            st.session_state["race_info"]     = race_info
            st.session_state["horses_input"]  = horses_input

            st.success("✅ Analyse terminée ! Consultez l'onglet **Analyse & Prédictions**.")

    with tab2:
        if "prediction" not in st.session_state:
            st.markdown("""
<div style="text-align:center;padding:60px 20px;color:#3a5a7a;">
    <h2 style="font-size:3em;margin-bottom:12px;">🏇</h2>
    <p style="font-size:1.1em;">Saisissez les données dans l'onglet <strong>Saisie des données</strong><br>
    puis cliquez sur <strong>🚀 ANALYSER LA COURSE</strong>.</p>
</div>""", unsafe_allow_html=True)
        else:
            tab_results(st.session_state["prediction"], st.session_state["race_info"])


if __name__ == "__main__":
    main()
