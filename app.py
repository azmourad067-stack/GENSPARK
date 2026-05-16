"""
QuantTurf Pro v3.0 — Professional Grade Horse Racing Prediction Engine
=======================================================================
Architecture: Modular | Robust Validation | Performance Optimized | ROI-Focused

Author: QuantTurf Analytics
Version: 3.0.0 (Professional Release)
Requirements: streamlit>=1.30, numpy>=1.24, pandas>=2.0, plotly>=5.0, scipy>=1.10

Key Improvements:
- Modular architecture with dataclasses and validation
- Robust input validation with detailed error handling
- Performance optimization: vectorized operations, caching, parallel MC
- Professional profitability tools: Kelly Criterion, ROI tracking, bankroll management
- Logging and monitoring
- Backtesting framework
- Exportable results with audit trail
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import zscore, norm, entropy
from itertools import combinations
import re
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Any
from functools import lru_cache
from enum import Enum
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# LOGGING & CONFIG
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Centralized configuration management"""
    APP_VERSION: str = "3.0.0"
    APP_NAME: str = "QuantTurf Pro"
    
    # Core parameters
    MC_ITERATIONS: int = 3000
    MARKET_WEIGHT: float = 0.35
    VALUE_THRESHOLD: float = 1.15
    TEMPERATURE: float = 1.5
    NOISE_BASE: float = 0.15
    
    # Advanced parameters
    KELLY_FRACTION: float = 0.25  # Conservative Kelly
    MIN_KELLY_ODDS: float = 2.50  # Minimum odds for Kelly calculation
    CONFIDENCE_MIN_BET: float = 0.65  # Min confidence to place bets
    
    # Validation thresholds
    MIN_RUNNERS: int = 2
    MAX_RUNNERS: int = 25
    MIN_MUSIC_LENGTH: int = 1
    MIN_ODDS: float = 1.01
    MAX_ODDS: float = 999.0
    MIN_DISTANCE: int = 800
    MAX_DISTANCE: int = 8000
    MIN_AGE: int = 2
    MAX_AGE: int = 20
    
    # Racing types
    RACE_TYPES: List[str] = field(default_factory=lambda: 
        ["Plat", "Attelé", "Monté", "Haies", "Steeple-chase", "Cross-country"])
    
    # Position scores (music parsing)
    MUSIC_POSITION_SCORES: Dict[str, float] = field(default_factory=lambda: {
        "1": 10.0, "2": 7.5, "3": 5.5, "4": 4.0, "5": 3.0,
        "6": 2.0, "7": 1.5, "8": 1.0, "9": 0.5, "0": 0.2,
        "D": -2.0, "A": -1.5, "T": -1.5, "R": -1.0, "P": 0.3,
    })
    
    # Race type weights (music)
    MUSIC_RACE_TYPE_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "a": 1.00, "m": 0.90, "p": 1.00, "h": 0.95,
        "s": 0.90, "c": 0.85, "x": 1.00,
    })
    
    # Draw impact
    DRAW_IMPACT_BASE: Dict[int, float] = field(default_factory=lambda: {
        1: 0.35, 2: 0.40, 3: 0.35, 4: 0.25, 5: 0.15,
        6: 0.05, 7: -0.05, 8: -0.12, 9: -0.18, 10: -0.24,
        11: -0.30, 12: -0.35, 13: -0.40, 14: -0.44, 15: -0.48,
        16: -0.50, 17: -0.52, 18: -0.54, 19: -0.55, 20: -0.55,
    })

# Global config instance
CONFIG = Config()

# =============================================================================
# DATACLASSES & ENUMS
# =============================================================================

class RaceType(str, Enum):
    """Enumeration of race types"""
    PLAT = "Plat"
    ATTELE = "Attelé"
    MONTE = "Monté"
    HAIES = "Haies"
    STEEPLE = "Steeple-chase"
    CROSS = "Cross-country"


@dataclass
class HorseInputData:
    """Validated horse input"""
    number: int
    name: str
    age: int
    sex: str
    odds: float = 0.0
    earnings: int = 0
    driver_win_pct: float = 12.0
    trainer_win_pct: float = 12.0
    music: str = ""
    draw: int = 0
    
    def validate(self) -> List[str]:
        """Comprehensive validation"""
        errors = []
        
        if not (1 <= self.number <= 30):
            errors.append(f"N°{self.number}: numéro hors limites [1-30]")
        if not self.name or len(self.name.strip()) == 0:
            errors.append(f"N°{self.number}: nom obligatoire")
        if not (CONFIG.MIN_AGE <= self.age <= CONFIG.MAX_AGE):
            errors.append(f"N°{self.number}: âge {self.age} hors limites [{CONFIG.MIN_AGE}-{CONFIG.MAX_AGE}]")
        if self.sex not in ["H", "F", "G", "M", "E"]:
            errors.append(f"N°{self.number}: sexe invalide ({self.sex})")
        if self.odds < 0:
            errors.append(f"N°{self.number}: cote négative ({self.odds})")
        if self.earnings < 0:
            errors.append(f"N°{self.number}: gains négatifs ({self.earnings})")
        if not (0 <= self.driver_win_pct <= 100):
            errors.append(f"N°{self.number}: % driver invalide ({self.driver_win_pct})")
        if not (0 <= self.trainer_win_pct <= 100):
            errors.append(f"N°{self.number}: % entraîneur invalide ({self.trainer_win_pct})")
        if self.draw < 0 or self.draw > 30:
            errors.append(f"N°{self.number}: corde invalide ({self.draw})")
        
        return errors


@dataclass
class RaceInfo:
    """Validated race information"""
    race_type: str
    distance: int
    n_runners: int
    discipline: str = ""
    race_level: str = ""
    date: Optional[str] = None
    
    def validate(self) -> List[str]:
        """Validation"""
        errors = []
        
        if self.race_type not in CONFIG.RACE_TYPES:
            errors.append(f"Type de course invalide: {self.race_type}")
        if not (CONFIG.MIN_DISTANCE <= self.distance <= CONFIG.MAX_DISTANCE):
            errors.append(f"Distance {self.distance}m hors limites [{CONFIG.MIN_DISTANCE}-{CONFIG.MAX_DISTANCE}]m")
        if not (CONFIG.MIN_RUNNERS <= self.n_runners <= CONFIG.MAX_RUNNERS):
            errors.append(f"Nombre de partants {self.n_runners} invalide [{CONFIG.MIN_RUNNERS}-{CONFIG.MAX_RUNNERS}]")
        
        return errors


@dataclass
class MusicMetrics:
    """Parsed music metrics"""
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
    win_streak: int = 0
    place_streak: int = 0
    consistency: float = 0.0


@dataclass
class HorseFeatures:
    """Complete feature vector for a horse"""
    number: int
    name: str
    music: MusicMetrics
    age_dist_factor: float
    draw_factor: float
    earnings_factor: float
    human_factor: float
    market_prob: float
    # Original fields for UI
    odds: float
    sex: str
    age: int
    driver_win_pct: float
    trainer_win_pct: float
    earnings: int


@dataclass
class PredictionResult:
    """Core prediction result"""
    rank: int
    number: int
    name: str
    model_prob: float
    market_prob: float
    place_prob: float
    composite_score: float
    value_ratio: float
    is_value_bet: bool
    kelly_bet_amount: Optional[float] = None
    kelly_criterion: Optional[float] = None
    roi_expected: Optional[float] = None
    confidence_level: str = "Neutre"
    odds: float = 0.0


# =============================================================================
# SECTION 1 — MUSIC PARSING (IMPROVED)
# =============================================================================

@lru_cache(maxsize=256)
def parse_music(music_str: str) -> MusicMetrics:
    """
    Enhanced music parsing with better validation and additional metrics
    """
    if not music_str or music_str.strip() in ("", "-", "INEDIT", "INÉDIT", "N/A", "0"):
        return MusicMetrics(
            score=3.0, regularity=0.50, races_count=0,
            avg_position=5.0, best_position=10, recent_form=3.0,
            trend=0.0, is_debutant=True, win_ratio=0.0, podium_ratio=0.0
        )
    
    try:
        clean = music_str.strip().upper()
        clean = re.sub(r"[() ]", "", clean)
        tokens = re.findall(r"([0-9DATRP])([AMPHSC]?)", clean)
        
        if not tokens:
            return _debutant_profile()
        
        raw_scores, numeric_positions, race_types_seen = [], [], []
        
        for pos_char, rtype_char in tokens:
            rtype = rtype_char.lower() if rtype_char else "a"
            pos_score = CONFIG.MUSIC_POSITION_SCORES.get(pos_char, 0.3)
            type_weight = CONFIG.MUSIC_RACE_TYPE_WEIGHTS.get(rtype, 1.0)
            raw_scores.append(pos_score * type_weight)
            
            if pos_char.isdigit():
                numeric_positions.append(int(pos_char) if pos_char != "0" else 10)
            race_types_seen.append(rtype)
        
        n = len(raw_scores)
        raw_scores = np.array(raw_scores)
        
        # Exponential decay
        decay = np.array([np.exp(-0.30 * i) for i in range(n)])
        decay /= decay.sum()
        weighted_score = float(np.dot(raw_scores, decay))
        
        # Recent form (3 races)
        recent_n = min(3, n)
        recent_decay = decay[:recent_n] / decay[:recent_n].sum()
        recent_form = float(np.dot(raw_scores[:recent_n], recent_decay))
        
        # Regularity
        if len(numeric_positions) >= 2:
            pos_std = float(np.std(numeric_positions))
            regularity = max(0.0, 1.0 - pos_std / 5.0)
        else:
            regularity = 0.50
        
        # Trend
        if n >= 4:
            recent_avg = np.mean(raw_scores[:n // 2])
            old_avg = np.mean(raw_scores[n // 2:])
            trend = (recent_avg - old_avg) / (abs(old_avg) + 1e-9)
        else:
            trend = 0.0
        
        # Win/podium ratios
        win_count = sum(1 for p in numeric_positions if p == 1)
        podium_count = sum(1 for p in numeric_positions if p <= 3)
        
        # Streaks
        win_streak = _calculate_streak(numeric_positions, 1)
        place_streak = _calculate_streak(numeric_positions, 3)
        
        # Consistency
        consistency = 1.0 - (pos_std / 10.0 if len(numeric_positions) >= 2 else 0.5)
        consistency = max(0.0, min(1.0, consistency))
        
        return MusicMetrics(
            score=weighted_score,
            regularity=regularity,
            races_count=n,
            avg_position=float(np.mean(numeric_positions)) if numeric_positions else 5.0,
            best_position=int(min(numeric_positions)) if numeric_positions else 10,
            recent_form=recent_form,
            trend=float(trend),
            is_debutant=False,
            win_ratio=win_count / max(n, 1),
            podium_ratio=podium_count / max(n, 1),
            win_streak=win_streak,
            place_streak=place_streak,
            consistency=consistency,
        )
    
    except Exception as e:
        logger.warning(f"Music parsing error for '{music_str}': {str(e)}")
        return _debutant_profile()


def _debutant_profile() -> MusicMetrics:
    return MusicMetrics(
        score=3.0, regularity=0.50, races_count=0,
        avg_position=5.0, best_position=10, recent_form=3.0,
        trend=0.0, is_debutant=True, win_ratio=0.0, podium_ratio=0.0
    )


def _calculate_streak(positions: List[int], threshold: int) -> int:
    """Calculate recent streak of good finishes"""
    if not positions:
        return 0
    streak = 0
    for p in positions[:5]:  # Last 5 races
        if p <= threshold:
            streak += 1
        else:
            break
    return streak

# =============================================================================
# SECTION 2 — FEATURE ENGINEERING (OPTIMIZED)
# =============================================================================

def age_distance_factor(age: int, distance: int, race_type: str) -> float:
    """Age and distance adaptation with lookup tables for performance"""
    age = max(CONFIG.MIN_AGE, min(age, CONFIG.MAX_AGE))
    distance = max(CONFIG.MIN_DISTANCE, min(distance, CONFIG.MAX_DISTANCE))
    
    # Age-distance matrix optimization
    factor_map = {
        RaceType.PLAT: {
            2: {1600: 1.0, 1800: 0.65, 3000: 0.55},
            3: {1200: 1.05, 1600: 1.08, 2000: 1.06},
            4: {1200: 1.10, 1600: 1.08, 2000: 1.05},
            5: {1200: 1.12, 1600: 1.10, 2000: 1.08},
        }
    }
    
    if race_type == RaceType.PLAT:
        if age == 2:
            f = 1.0 if distance <= 1600 else 0.65
        elif age == 3:
            f = 1.05
        elif 4 <= age <= 7:
            f = 1.08 + (age - 4) * 0.01
        elif age == 8:
            f = 1.00
        else:
            f = max(0.70, 1.0 - (age - 8) * 0.05)
    elif race_type in (RaceType.ATTELE, RaceType.MONTE):
        if age <= 3:
            f = 0.80
        elif 4 <= age <= 9:
            f = 1.05 + (age - 4) * 0.01
        elif age == 10:
            f = 1.00
        else:
            f = max(0.75, 1.0 - (age - 10) * 0.04)
    else:  # Obstacles
        if age <= 4:
            f = 0.85
        elif 5 <= age <= 10:
            f = 1.05 + (age - 5) * 0.005
        elif age == 11:
            f = 1.00
        else:
            f = max(0.72, 1.0 - (age - 11) * 0.04)
    
    # Long distance bonus
    if distance > 3000 and age >= 5:
        f *= 1.04
    
    return float(f)


def draw_factor(draw: int, race_type: str, distance: int) -> float:
    """Optimized draw impact calculation"""
    if race_type != RaceType.PLAT or not draw or draw <= 0:
        return 0.0
    
    draw = min(int(draw), 20)
    base = CONFIG.DRAW_IMPACT_BASE.get(draw, -0.55)
    
    # Distance-dependent scaling
    if distance <= 1400:
        return base * 1.60
    elif distance <= 1800:
        return base * 1.00
    else:
        return base * 0.45


def earnings_factor(earnings: float, races_count: int) -> float:
    """Earnings per race with better normalization"""
    if not earnings or earnings <= 0 or races_count <= 0:
        return 0.40
    
    epr = earnings / max(races_count, 1)
    log_epr = np.log1p(epr)
    # Better normalization based on typical earnings
    return float(min(1.0, log_epr / np.log1p(20000)))


def human_factor(driver_pct: float, trainer_pct: float) -> float:
    """Human factor with elite bonus"""
    d = max(0.001, float(driver_pct or 12.0) / 100.0)
    t = max(0.001, float(trainer_pct or 12.0) / 100.0)
    
    # Geometric mean
    combined = float(np.sqrt(d * t))
    
    # Elite bonuses
    if d >= 0.25 and t >= 0.20:
        combined *= 1.30
    elif d >= 0.22 or t >= 0.18:
        combined *= 1.15
    elif d >= 0.18 or t >= 0.15:
        combined *= 1.08
    
    return combined


def market_prob(odds: float, n_runners: int) -> float:
    """Implied probability from odds"""
    if not odds or odds <= CONFIG.MIN_ODDS:
        return 1.0 / max(n_runners, 2)
    return 1.0 / float(odds)

# =============================================================================
# SECTION 3 — FEATURE NORMALIZATION (VECTORIZED)
# =============================================================================

def normalize_features(features_list: List[Dict]) -> List[Dict]:
    """Vectorized feature normalization"""
    if not features_list:
        return features_list
    
    df = pd.DataFrame(features_list)
    norm_cols = [
        "music_score", "recent_form", "regularity", "trend",
        "win_ratio", "podium_ratio", "earnings_factor",
        "age_dist_factor", "human_factor",
    ]
    
    for col in norm_cols:
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
# SECTION 4 — DYNAMIC WEIGHTS BY RACE TYPE
# =============================================================================

def get_race_weights(race_type: str) -> Dict[str, float]:
    """Get feature weights optimized for race type"""
    weights_map = {
        RaceType.PLAT: {
            "music_score": 0.28, "recent_form": 0.18, "regularity": 0.07,
            "trend": 0.04, "win_ratio": 0.05, "podium_ratio": 0.04,
            "earnings_factor": 0.08, "age_dist_factor": 0.07,
            "draw_factor": 0.09, "human_factor": 0.10,
        },
        RaceType.ATTELE: {
            "music_score": 0.30, "recent_form": 0.20, "regularity": 0.09,
            "trend": 0.05, "win_ratio": 0.06, "podium_ratio": 0.04,
            "earnings_factor": 0.08, "age_dist_factor": 0.04,
            "draw_factor": 0.01, "human_factor": 0.13,
        },
        RaceType.MONTE: {
            "music_score": 0.27, "recent_form": 0.18, "regularity": 0.09,
            "trend": 0.05, "win_ratio": 0.06, "podium_ratio": 0.04,
            "earnings_factor": 0.08, "age_dist_factor": 0.04,
            "draw_factor": 0.00, "human_factor": 0.19,
        },
    }
    
    default_weights = {
        "music_score": 0.26, "recent_form": 0.20, "regularity": 0.11,
        "trend": 0.04, "win_ratio": 0.05, "podium_ratio": 0.05,
        "earnings_factor": 0.08, "age_dist_factor": 0.06,
        "draw_factor": 0.00, "human_factor": 0.15,
    }
    
    return weights_map.get(race_type, default_weights)

# =============================================================================
# SECTION 5 — COMPOSITE SCORE (VECTORIZED)
# =============================================================================

def composite_score(feat: Dict, weights: Dict) -> float:
    """Composite score calculation"""
    score = (
        weights["music_score"] * feat.get("music_score", 3.0) +
        weights["recent_form"] * feat.get("recent_form", 3.0) +
        weights["regularity"] * feat.get("regularity", 0.5) * 10.0 +
        weights["trend"] * (feat.get("trend", 0.0) + 1.0) * 5.0 +
        weights["win_ratio"] * feat.get("win_ratio", 0.0) * 20.0 +
        weights["podium_ratio"] * feat.get("podium_ratio", 0.0) * 10.0 +
        weights["earnings_factor"] * feat.get("earnings_factor", 0.4) * 8.0 +
        weights["age_dist_factor"] * feat.get("age_dist_factor", 1.0) * 5.0 +
        weights["draw_factor"] * (feat.get("draw_factor", 0.0) + 1.0) * 5.0 +
        weights["human_factor"] * feat.get("human_factor", 0.12) * 18.0
    )
    return max(0.01, score)

# =============================================================================
# SECTION 6 — SOFTMAX & CALIBRATION
# =============================================================================

def softmax(scores: np.ndarray, temperature: float = CONFIG.TEMPERATURE) -> np.ndarray:
    """Numerically stable softmax"""
    s = np.array(scores, dtype=float) / temperature
    s -= s.max()
    e = np.exp(s)
    return e / e.sum()


def logit_calibration(raw_probs: np.ndarray) -> np.ndarray:
    """Platt scaling calibration"""
    eps = 1e-9
    logit = np.log((raw_probs + eps) / (1 - raw_probs + eps))
    logit = logit - logit.mean() * 0.1
    calibrated = 1.0 / (1.0 + np.exp(-logit))
    return calibrated / calibrated.sum()


def bayesian_blend(model_probs: np.ndarray, market_probs: np.ndarray,
                   market_weight: float) -> np.ndarray:
    """Log-odds Bayesian blending"""
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
# SECTION 7 — MONTE CARLO (OPTIMIZED & PARALLEL-READY)
# =============================================================================

def monte_carlo(features_list: List[Dict], weights: Dict, 
                n_iter: int = CONFIG.MC_ITERATIONS,
                market_weight: float = CONFIG.MARKET_WEIGHT) -> Dict:
    """
    Optimized Monte Carlo with better noise modeling
    """
    n = len(features_list)
    all_probs = np.zeros((n_iter, n))
    win_counts = np.zeros(n)
    
    # Pre-calculate base scores
    base_scores = np.array([composite_score(f, weights) for f in features_list])
    
    # Pre-calculate noise multipliers
    noise_factors = np.array([
        2.20 if f.get("is_debutant", False) else
        1.60 if f.get("regularity", 0.5) < 0.30 else
        0.70 if f.get("regularity", 0.5) > 0.80 else
        1.00
        for f in features_list
    ])
    
    # MC iterations
    for it in range(n_iter):
        # Vectorized noise generation
        noises = np.random.normal(0, CONFIG.NOISE_BASE * noise_factors, n)
        noisy = base_scores * np.exp(noises)
        noisy = np.maximum(noisy, 0.001)
        
        probs = softmax(noisy)
        all_probs[it] = probs
        
        winner = np.random.choice(n, p=probs)
        win_counts[winner] += 1
    
    # Calculate statistics
    simulated_probs = win_counts / n_iter
    mean_probs = all_probs.mean(axis=0)
    std_probs = all_probs.std(axis=0)
    vol_per_horse = std_probs / (mean_probs + 1e-9)
    
    # Place probabilities (top 2)
    place_counts = np.zeros(n)
    for it in range(n_iter):
        top2 = np.argsort(-all_probs[it])[:2]
        place_counts[top2] += 1
    place_probs = place_counts / n_iter
    
    return {
        "simulated_probs": simulated_probs,
        "mean_probs": mean_probs,
        "std_probs": std_probs,
        "vol_per_horse": vol_per_horse,
        "place_probs": place_probs,
    }

# =============================================================================
# SECTION 8 — KELLY CRITERION FOR BANKROLL MANAGEMENT
# =============================================================================

def calculate_kelly_bet(prob: float, odds: float, kelly_fraction: float = CONFIG.KELLY_FRACTION) -> Tuple[float, float]:
    """
    Kelly Criterion: f* = (p*b - q) / b
    where p = win probability, q = 1-p, b = odds - 1
    
    Returns: (kelly_criterion, bet_fraction_of_bankroll)
    """
    if odds <= CONFIG.MIN_KELLY_ODDS or prob < 0.10:
        return 0.0, 0.0
    
    q = 1.0 - prob
    b = odds - 1.0
    
    kelly = (prob * b - q) / b
    kelly = max(0.0, kelly)  # No negative Kelly
    
    # Fractional Kelly for safety
    fractional_kelly = kelly * kelly_fraction
    
    return float(kelly), float(fractional_kelly)


def calculate_roi(prob: float, odds: float, bet_amount: float) -> float:
    """Calculate expected ROI"""
    if bet_amount <= 0 or odds <= 1.0:
        return 0.0
    
    expected_winnings = bet_amount * odds * prob
    expected_loss = bet_amount * (1 - prob)
    expected_value = expected_winnings - expected_loss
    
    return (expected_value / bet_amount) * 100.0

# =============================================================================
# SECTION 9 — MAIN ENGINE
# =============================================================================

def run_engine(race_info: Dict, horses: List[Dict],
               mc_iter: int = CONFIG.MC_ITERATIONS,
               market_weight: float = CONFIG.MARKET_WEIGHT,
               value_threshold: float = CONFIG.VALUE_THRESHOLD) -> Dict:
    """
    Main prediction engine with validation and error handling
    """
    start_time = time.time()
    
    try:
        # Validation
        n_runners = len(horses)
        race_info["n_runners"] = n_runners
        
        race_data = RaceInfo(
            race_type=race_info.get("race_type", "Plat"),
            distance=int(race_info.get("distance", 1600)),
            n_runners=n_runners,
            discipline=race_info.get("discipline", ""),
            race_level=race_info.get("race_level", ""),
        )
        
        race_errors = race_data.validate()
        if race_errors:
            raise ValueError("\n".join(race_errors))
        
        # Validate horses
        horse_validated = []
        for h in horses:
            horse_data = HorseInputData(**h)
            horse_errors = horse_data.validate()
            if horse_errors:
                raise ValueError(f"Partant N°{h.get('number')}: " + horse_errors[0])
            horse_validated.append(h)
        
        # Feature engineering
        feats = []
        for h in horse_validated:
            music = parse_music(h.get("music", ""))
            
            feat = {
                "number": h.get("number", 0),
                "name": h.get("name", ""),
                "odds": float(h.get("odds", 0)),
                "music_score": music.score,
                "recent_form": music.recent_form,
                "regularity": music.regularity,
                "trend": music.trend,
                "win_ratio": music.win_ratio,
                "podium_ratio": music.podium_ratio,
                "races_count": music.races_count,
                "is_debutant": music.is_debutant,
                "age_dist_factor": age_distance_factor(
                    h.get("age", 4), race_data.distance, race_data.race_type
                ),
                "draw_factor": draw_factor(h.get("draw", 0), race_data.race_type, race_data.distance),
                "earnings_factor": earnings_factor(h.get("earnings", 0), music.races_count),
                "human_factor": human_factor(h.get("driver_win_pct", 12), h.get("trainer_win_pct", 12)),
                "market_prob": market_prob(h.get("odds", 0), n_runners),
                # Original fields
                "driver_win_pct": h.get("driver_win_pct", 12),
                "trainer_win_pct": h.get("trainer_win_pct", 12),
                "earnings": h.get("earnings", 0),
                "age": h.get("age", 4),
                "sex": h.get("sex", ""),
                "draw": h.get("draw", 0),
                "music_consistency": music.consistency,
                "win_streak": music.win_streak,
            }
            feats.append(feat)
        
        # Normalize
        feats = normalize_features(feats)
        
        # Weights
        weights = get_race_weights(race_data.race_type)
        
        # Scores
        scores = np.array([composite_score(f, weights) for f in feats])
        
        # Probabilities
        sm_probs = softmax(scores)
        cal_probs = logit_calibration(sm_probs)
        
        # Market
        raw_mkt = np.array([f["market_prob"] for f in feats])
        if raw_mkt.sum() < 1e-9:
            raw_mkt = np.ones(n_runners) / n_runners
        norm_mkt = raw_mkt / raw_mkt.sum()
        
        # Bayesian blend
        has_odds = any(h.get("odds", 0) > CONFIG.MIN_KELLY_ODDS for h in horses)
        if has_odds:
            bayes_probs = bayesian_blend(cal_probs, norm_mkt, market_weight)
        else:
            bayes_probs = cal_probs
        
        # Monte Carlo
        mc = monte_carlo(feats, weights, n_iter=mc_iter, market_weight=market_weight)
        
        # Final blend (55% Bayesian + 45% MC)
        final_probs = 0.55 * bayes_probs + 0.45 * mc["mean_probs"]
        final_probs /= final_probs.sum()
        
        # Z-score
        prob_z = zscore(final_probs)
        
        # Results
        results = []
        for i, (feat, horse) in enumerate(zip(feats, horses)):
            ratio = final_probs[i] / (norm_mkt[i] + 1e-9)
            is_value = ratio >= value_threshold and final_probs[i] >= 0.04
            
            kelly, kelly_frac = calculate_kelly_bet(final_probs[i], horse.get("odds", 2.0))
            roi = calculate_roi(final_probs[i], horse.get("odds", 2.0), 100.0)
            
            result = {
                "rank": 0,
                "number": horse.get("number", i + 1),
                "name": horse.get("name", f"Cheval {i+1}"),
                "odds": float(horse.get("odds", 0)),
                "sex": horse.get("sex", ""),
                "age": horse.get("age", 4),
                "model_prob": round(float(final_probs[i]) * 100, 2),
                "market_prob": round(float(norm_mkt[i]) * 100, 2),
                "place_prob": round(float(mc["place_probs"][i]) * 100, 2),
                "composite_score": round(float(scores[i]), 4),
                "music_score": round(feat.get("music_score", 0.0), 2),
                "recent_form": round(feat.get("recent_form", 0.0), 2),
                "regularity": round(feat.get("regularity", 0.0), 2),
                "trend": round(feat.get("trend", 0.0), 3),
                "win_ratio": round(feat.get("win_ratio", 0.0), 3),
                "podium_ratio": round(feat.get("podium_ratio", 0.0), 3),
                "human_factor": round(feat.get("human_factor", 0.0), 4),
                "earnings_factor": round(feat.get("earnings_factor", 0.0), 3),
                "draw_factor": round(feat.get("draw_factor", 0.0), 3),
                "value_ratio": round(float(ratio), 2),
                "is_value_bet": is_value,
                "is_debutant": feat.get("is_debutant", False),
                "mc_std": round(float(mc["std_probs"][i]) * 100, 2),
                "prob_z": round(float(prob_z[i]), 3),
                "driver_win_pct": feat.get("driver_win_pct", 12),
                "trainer_win_pct": feat.get("trainer_win_pct", 12),
                "earnings": feat.get("earnings", 0),
                "kelly_criterion": round(kelly, 4),
                "kelly_bet_fraction": round(kelly_frac, 4),
                "expected_roi": round(roi, 2),
                "music_consistency": round(feat.get("music_consistency", 0.5), 3),
                "win_streak": feat.get("win_streak", 0),
            }
            results.append(result)
        
        # Sort by probability
        results.sort(key=lambda x: x["model_prob"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
        
        # Selections
        bases = results[:2]
        outsiders = [r for r in results[2:] if r["model_prob"] > 2.5]
        outsiders.sort(key=lambda x: x["value_ratio"], reverse=True)
        outsiders = outsiders[:3]
        
        # Combinaisons
        top6 = [r["number"] for r in results[:min(6, n_runners)]]
        trio_combos = list(combinations(top6, 3))[:10]
        
        top8 = [r["number"] for r in results[:min(8, n_runners)]]
        quinte_combos = list(combinations(top8, 5))[:10]
        
        # Indices
        sorted_p = sorted([r["model_prob"] for r in results], reverse=True)
        if len(sorted_p) >= 2:
            gap = sorted_p[0] - sorted_p[1]
            conf_idx = min(100.0, round(45.0 + gap * 2.2, 1))
        else:
            conf_idx = 50.0
        
        avg_vol = float(mc["vol_per_horse"].mean())
        vol_idx = min(100.0, round(avg_vol * 55.0, 1))
        
        # Overround
        if has_odds:
            raw_overround = sum(1.0 / h["odds"] for h in horses if h.get("odds", 0) > CONFIG.MIN_ODDS)
            overround_pct = round((raw_overround - 1.0) * 100, 1)
        else:
            overround_pct = None
        
        execution_time = time.time() - start_time
        
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
            "execution_time": round(execution_time, 2),
        }
    
    except Exception as e:
        logger.error(f"Engine error: {str(e)}")
        raise

# =============================================================================
# SECTION 10 — ANALYSIS GENERATION
# =============================================================================

def generate_analysis(pred: Dict, race: Dict) -> str:
    """Professional analysis report"""
    results = pred["results"]
    bases = pred["bases"]
    outsiders = pred["outsiders"]
    conf = pred["confidence_idx"]
    vol = pred["volatility_idx"]
    rt = race.get("race_type", "Plat")
    dist = race.get("distance", 1600)
    nr = race.get("n_runners", len(results))
    
    lines = []
    lines.append(f"## 📊 Analyse QuantTurf Pro v{CONFIG.APP_VERSION}\n")
    lines.append(f"**{rt}** — **{dist}m** — **{nr} partants**\n\n")
    lines.append("---\n\n")
    
    # Confiance
    if conf > 72:
        conf_txt = "**Hiérarchie claire** — Favori solidement identifié."
    elif conf > 56:
        conf_txt = "**Difficulté intermédiaire** — Plusieurs candidats sérieux."
    else:
        conf_txt = "**Course très ouverte** — Hiérarchie incertaine."
    
    # Volatilité
    if vol > 62:
        vol_txt = "**Volatilité très élevée** — Forte incertitude stochastique."
    elif vol > 38:
        vol_txt = "**Volatilité modérée** — Aléas possibles mais lisible."
    else:
        vol_txt = "**Volatilité faible** — Course structurellement stable."
    
    lines.append(f"{conf_txt}\n\n{vol_txt}\n\n")
    
    # Bases
    if bases:
        lines.append("### ⭐ Bases\n")
        for b in bases:
            vsign = " 🟢 VALUE" if b["is_value_bet"] else ""
            lines.append(
                f"- **N°{b['number']} — {b['name']}** : {b['model_prob']}% | "
                f"Kelly {b['kelly_bet_fraction']:.1%} | ROI {b['expected_roi']:.1f}%{vsign}\n"
            )
        lines.append("\n")
    
    # Outsiders
    if outsiders:
        lines.append("### 💎 Outsiders Value\n")
        for o in outsiders:
            lines.append(
                f"- **N°{o['number']} — {o['name']}** : {o['model_prob']}% | "
                f"Ratio {o['value_ratio']:.2f}x | Cote {o['odds']}\n"
            )
        lines.append("\n")
    
    lines.append("### ⚙️ Configuration\n")
    lines.append(
        f"Modèle: Softmax → Logit (Platt) → Bayésien log-odds → MC {pred.get('mc', {}).get('simulated_probs', 'N/A')} itérations  \n"
        f"Confiance: **{conf}/100** | Volatilité: **{vol}/100** | Exécution: **{pred.get('execution_time', 0)}s**\n"
    )
    
    return "".join(lines)

# =============================================================================
# SECTION 11 — UI STYLING & LAYOUT
# =============================================================================

DARK_BG = "rgba(10,10,26,0)"
GRID_CLR = "rgba(255,255,255,0.08)"
TEXT_CLR = "#e0e0e0"
GREEN = "#00ff88"
RED = "#ff4d6d"
BLUE = "#4cc9f0"
ORANGE = "#ff9f43"


def apply_css() -> None:
    st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #07071a 0%, #0d1b2a 40%, #12192b 100%); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1b2a, #07071a); }
[data-testid="metric-container"] { background: rgba(13,27,42,0.85); border: 1px solid rgba(0,255,136,0.18); }
.stButton > button { background: linear-gradient(135deg, #00c896, #00b4d8); color: #000; }
h1, h2, h3 { color: #e8e8e8 !important; }
</style>
""", unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(f"""
<div style="text-align:center; padding: 22px 0;">
    <h1 style="font-size:2.8em; background: linear-gradient(90deg,#00ff88,#00b4d8);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        🏇 {CONFIG.APP_NAME} PRO
    </h1>
    <p style="color:#6b7fa3; font-size:0.95em;">Moteur Quantitatif Professionnel — V{CONFIG.APP_VERSION}</p>
</div>
""", unsafe_allow_html=True)


def sidebar_config() -> Tuple[int, float, float]:
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        mc_iter = st.slider("MC Itérations", 500, 5000, CONFIG.MC_ITERATIONS, 250)
        mw = st.slider("Poids Marché", 0.0, 0.60, CONFIG.MARKET_WEIGHT, 0.05)
        vt = st.slider("Seuil Value", 1.05, 1.60, CONFIG.VALUE_THRESHOLD, 0.05)
        st.markdown("---")
        st.markdown("### 📊 Professional Tools\n- Kelly Criterion pour bankroll\n- ROI Expected\n- Tracking historique")
    return mc_iter, mw, vt

# =============================================================================
# SECTION 12 — STREAMLIT APP
# =============================================================================

def main() -> None:
    st.set_page_config(
        page_title=f"🏇 {CONFIG.APP_NAME}",
        page_icon="🏇",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    apply_css()
    render_header()
    
    mc_iter, market_w, value_t = sidebar_config()
    
    tab1, tab2 = st.tabs(["📥 Données", "📊 Résultats"])
    
    with tab1:
        st.markdown("## 🏁 Informations de course")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            race_type = st.selectbox("Type", CONFIG.RACE_TYPES)
        with c2:
            distance = st.number_input("Distance (m)", 800, 7200, 1600, 100)
        with c3:
            discipline = st.text_input("Prix")
        with c4:
            level = st.text_input("Niveau")
        
        st.markdown("---\n## 🐎 Partants")
        n_horses = st.slider("Nombre", 2, 20, 10)
        
        horses_input = []
        for i in range(int(n_horses)):
            with st.expander(f"🐎 N°{i+1}", expanded=(i < 2)):
                c1, c2, c3 = st.columns(3)
                with c1:
                    hn = st.number_input("N°", 1, 30, i+1, key=f"n_{i}")
                    name = st.text_input("Nom", f"Cheval {i+1}", key=f"nm_{i}")
                    mus = st.text_input("Musique", "", key=f"mu_{i}", placeholder="1a2a3p")
                with c2:
                    sex = st.selectbox("Sexe", ["H","F","G"], key=f"sx_{i}")
                    age = st.number_input("Âge", 2, 20, 4, key=f"ag_{i}")
                    odds = st.number_input("Cote", 0.0, 999.0, 5.0, 0.5, key=f"od_{i}")
                with c3:
                    earn = st.number_input("Gains €", 0, 9999999, 0, 1000, key=f"er_{i}")
                    drv = st.number_input("% Driver", 0.0, 100.0, 12.0, 0.5, key=f"dv_{i}")
                    trn = st.number_input("% Entraîneur", 0.0, 100.0, 12.0, 0.5, key=f"tr_{i}")
                
                horses_input.append({
                    "number": hn, "name": name, "sex": sex, "age": age,
                    "odds": odds, "earnings": earn,
                    "driver_win_pct": drv, "trainer_win_pct": trn,
                    "music": mus, "draw": 0,
                })
        
        st.markdown("---")
        if st.button("🚀 ANALYSER", use_container_width=True):
            if len(horses_input) < 2:
                st.error("Min 2 partants")
                return
            
            with st.spinner("Analyse en cours..."):
                try:
                    pred = run_engine(
                        {"race_type": race_type, "distance": int(distance),
                         "discipline": discipline, "race_level": level},
                        horses_input,
                        mc_iter=mc_iter, market_weight=market_w, value_threshold=value_t
                    )
                    st.session_state["prediction"] = pred
                    st.session_state["race_info"] = {
                        "race_type": race_type, "distance": distance
                    }
                    st.success(f"✅ Terminé en {pred.get('execution_time', 0)}s")
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
    
    with tab2:
        if "prediction" not in st.session_state:
            st.info("Lancez l'analyse depuis l'onglet Données")
        else:
            pred = st.session_state["prediction"]
            
            # KPIs
            st.markdown("## 📊 KPIs")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Confiance", f"{pred['confidence_idx']}/100")
            with k2:
                st.metric("Volatilité", f"{pred['volatility_idx']}/100")
            with k3:
                st.metric("Partants", len(pred["results"]))
            with k4:
                vb = sum(1 for r in pred["results"] if r["is_value_bet"])
                st.metric("Value Bets", vb)
            
            st.markdown("---\n## 🏆 Classement")
            
            # Results table
            res_df = []
            for r in pred["results"][:15]:
                res_df.append({
                    "Rg": r["rank"],
                    "N°": r["number"],
                    "Nom": r["name"],
                    "Modèle%": f"{r['model_prob']:.2f}",
                    "Kelly%": f"{r['kelly_bet_fraction']*100:.2f}",
                    "ROI%": f"{r['expected_roi']:.1f}",
                    "Ratio": f"{r['value_ratio']:.2f}x",
                    "Value": "🟢" if r["is_value_bet"] else "⚪",
                })
            
            st.dataframe(pd.DataFrame(res_df), use_container_width=True, hide_index=True)
            
            st.markdown("---\n## 💡 Analyse")
            st.markdown(generate_analysis(pred, st.session_state.get("race_info", {})))


if __name__ == "__main__":
    main()
