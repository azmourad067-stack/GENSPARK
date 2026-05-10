"""
QuantTurf Pro v3 — Configuration & Professional Tools
======================================================
External configuration + backtesting + performance tracking
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

# =============================================================================
# SECTION 1 — CONFIGURATION SYSTEM
# =============================================================================

@dataclass
class BettingConfig:
    """Betting parameters"""
    kelly_fraction: float = 0.25          # Conservative Kelly (20-50% of full Kelly)
    min_kelly_odds: float = 2.50          # Minimum odds for Kelly calculation
    max_bet_pct: float = 0.02             # Max bet as % of bankroll (2%)
    max_daily_exposure: float = 0.10      # Max daily exposure (10%)
    min_confidence: float = 0.60          # Minimum confidence to place bets
    min_value_ratio: float = 1.15         # Minimum value bet threshold
    min_model_prob: float = 0.04          # Minimum model probability to consider


@dataclass
class RaceConfig:
    """Race-specific parameters"""
    min_runners: int = 2
    max_runners: int = 25
    max_overround_pct: float = 15.0       # Skip races with >15% overround
    min_market_price: float = 1.01
    max_market_price: float = 999.0


@dataclass
class MCConfig:
    """Monte Carlo parameters"""
    iterations_standard: int = 3000
    iterations_fast: int = 1000           # For quick analysis
    iterations_deep: int = 5000           # For uncertain races
    noise_base: float = 0.15
    noise_debutant_multiplier: float = 2.20
    noise_low_regularity_multiplier: float = 1.60
    noise_high_regularity_multiplier: float = 0.70


@dataclass
class ModelConfig:
    """Model parameters"""
    bayesian_weight: float = 0.35
    bayesian_blend_model: float = 0.55    # 55% Bayesian + 45% MC
    bayesian_blend_mc: float = 0.45
    temperature: float = 1.5              # Softmax temperature
    calibration_scale: float = 0.1        # Logit calibration scaling


@dataclass
class ConfigSet:
    """Master configuration"""
    betting: BettingConfig = None
    race: RaceConfig = None
    mc: MCConfig = None
    model: ModelConfig = None
    
    def __post_init__(self):
        if self.betting is None:
            self.betting = BettingConfig()
        if self.race is None:
            self.race = RaceConfig()
        if self.mc is None:
            self.mc = MCConfig()
        if self.model is None:
            self.model = ModelConfig()
    
    def to_dict(self) -> Dict:
        """Export as dict"""
        return {
            "betting": asdict(self.betting),
            "race": asdict(self.race),
            "mc": asdict(self.mc),
            "model": asdict(self.model),
        }
    
    def to_json(self, filepath: str) -> None:
        """Save to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: str) -> "ConfigSet":
        """Load from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(
            betting=BettingConfig(**data.get("betting", {})),
            race=RaceConfig(**data.get("race", {})),
            mc=MCConfig(**data.get("mc", {})),
            model=ModelConfig(**data.get("model", {})),
        )


# =============================================================================
# SECTION 2 — BACKTESTING FRAMEWORK
# =============================================================================

@dataclass
class BetRecord:
    """Single bet record"""
    race_id: str
    date: str
    horse_number: int
    horse_name: str
    odds: float
    model_prob: float
    market_prob: float
    bet_type: str              # "win", "place", "exacta", etc.
    bet_amount: float
    kelly_fraction: float
    result: Optional[str] = None  # "win", "place", "lose"
    payout: float = 0.0
    profit: float = 0.0
    roi: float = 0.0


class Backtester:
    """Professional backtesting framework"""
    
    def __init__(self):
        self.bets: List[BetRecord] = []
        self.races: Dict = {}
    
    def add_bet(self, bet: BetRecord) -> None:
        """Record a bet"""
        self.bets.append(bet)
    
    def settle_bet(self, race_id: str, horse_number: int, 
                   result: str, payout: float = 0.0) -> None:
        """Settle a bet after race"""
        for bet in self.bets:
            if bet.race_id == race_id and bet.horse_number == horse_number:
                bet.result = result
                bet.payout = payout
                if result == "win":
                    bet.profit = payout - bet.bet_amount
                elif result == "place":
                    bet.profit = (payout - bet.bet_amount) * 0.5
                else:
                    bet.profit = -bet.bet_amount
                bet.roi = (bet.profit / bet.bet_amount) * 100
    
    def get_statistics(self) -> Dict:
        """Calculate performance statistics"""
        df = pd.DataFrame([asdict(b) for b in self.bets])
        
        if df.empty:
            return {}
        
        df['result'].fillna('lose', inplace=True)
        
        total_bets = len(df)
        wins = len(df[df['result'] == 'win'])
        places = len(df[df['result'] == 'place'])
        losses = len(df[df['result'] == 'lose'])
        
        total_staked = df['bet_amount'].sum()
        total_profit = df['profit'].sum()
        total_roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
        
        # By confidence level
        high_conf = df[df['kelly_fraction'] > 0.02]
        if len(high_conf) > 0:
            high_conf_roi = (high_conf['profit'].sum() / high_conf['bet_amount'].sum() * 100)
        else:
            high_conf_roi = 0
        
        return {
            "total_bets": total_bets,
            "wins": wins,
            "win_rate": wins / total_bets if total_bets > 0 else 0,
            "places": places,
            "losses": losses,
            "total_staked": total_staked,
            "total_profit": total_profit,
            "roi": total_roi,
            "avg_odds": df['odds'].mean(),
            "avg_model_prob": df['model_prob'].mean(),
            "avg_bet_amount": df['bet_amount'].mean(),
            "high_confidence_roi": high_conf_roi,
            "kelly_efficiency": self._calculate_kelly_efficiency(df),
        }
    
    def _calculate_kelly_efficiency(self, df: pd.DataFrame) -> float:
        """Calculate how well Kelly sizing performed"""
        # Compare actual ROI vs expected from model_prob
        expected_roi = (df['model_prob'] * df['odds'] - 1).mean() * 100
        actual_roi = (df['profit'].sum() / df['bet_amount'].sum() * 100) if df['bet_amount'].sum() > 0 else 0
        
        return actual_roi - expected_roi if expected_roi != 0 else 0
    
    def export_csv(self, filepath: str) -> None:
        """Export bet history to CSV"""
        df = pd.DataFrame([asdict(b) for b in self.bets])
        df.to_csv(filepath, index=False)
    
    def plot_cumulative_roi(self):
        """Plot cumulative ROI curve"""
        df = pd.DataFrame([asdict(b) for b in self.bets])
        df['cumulative_profit'] = df['profit'].cumsum()
        df['cumulative_roi'] = (df['cumulative_profit'] / df['bet_amount'].cumsum() * 100)
        return df  # Return for plotting


# =============================================================================
# SECTION 3 — PERFORMANCE MONITOR
# =============================================================================

class PerformanceMonitor:
    """Track real-time performance metrics"""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.predictions: List[Dict] = []
    
    def add_prediction(self, pred: Dict) -> None:
        """Add prediction result"""
        self.predictions.append({
            "timestamp": datetime.now().isoformat(),
            "confidence_idx": pred.get("confidence_idx"),
            "volatility_idx": pred.get("volatility_idx"),
            "value_bets": sum(1 for r in pred.get("results", []) if r["is_value_bet"]),
            "top_3_model_probs": sorted(
                [r["model_prob"] for r in pred.get("results", [])],
                reverse=True
            )[:3],
        })
    
    def get_metrics(self) -> Dict:
        """Calculate current performance metrics"""
        if not self.predictions:
            return {}
        
        recent = self.predictions[-self.window_size:]
        
        return {
            "avg_confidence": np.mean([p["confidence_idx"] for p in recent]),
            "avg_volatility": np.mean([p["volatility_idx"] for p in recent]),
            "avg_value_bets": np.mean([p["value_bets"] for p in recent]),
            "prediction_count": len(self.predictions),
            "confidence_trend": self._calculate_trend([p["confidence_idx"] for p in recent]),
            "volatility_trend": self._calculate_trend([p["volatility_idx"] for p in recent]),
        }
    
    @staticmethod
    def _calculate_trend(values: List[float]) -> float:
        """Calculate trend (slope) of values"""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        y = np.array(values)
        z = np.polyfit(x, y, 1)
        return float(z[0])


# =============================================================================
# SECTION 4 — RISK MANAGEMENT
# =============================================================================

class RiskManager:
    """Professional risk management"""
    
    def __init__(self, initial_bankroll: float):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.daily_exposure = 0.0
        self.daily_bets: List[BetRecord] = []
    
    def calculate_kelly_sizing(self, prediction: Dict, horse_result: Dict) -> float:
        """Calculate Kelly-optimized bet size"""
        prob = horse_result["model_prob"] / 100
        odds = horse_result["odds"]
        kelly_frac = horse_result.get("kelly_bet_fraction", 0.0)
        
        max_bet = self.current_bankroll * 0.02  # 2% max per horse
        kelly_bet = self.current_bankroll * kelly_frac
        
        return min(max_bet, kelly_bet)
    
    def check_daily_limit(self, bet_amount: float) -> bool:
        """Check if daily exposure limit exceeded"""
        if self.daily_exposure + bet_amount > self.current_bankroll * 0.10:
            return False
        self.daily_exposure += bet_amount
        return True
    
    def record_result(self, bet: BetRecord) -> None:
        """Record bet result and update bankroll"""
        self.current_bankroll += bet.profit
        self.daily_bets.append(bet)
    
    def reset_daily(self) -> None:
        """Reset daily tracking"""
        self.daily_exposure = 0.0
        self.daily_bets = []
    
    def get_status(self) -> Dict:
        """Get current status"""
        daily_profit = sum(b.profit for b in self.daily_bets)
        return {
            "bankroll": round(self.current_bankroll, 2),
            "roi_from_initial": round(
                (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll * 100, 2
            ),
            "daily_exposure": round(self.daily_exposure, 2),
            "daily_profit": round(daily_profit, 2),
            "daily_bet_count": len(self.daily_bets),
        }


# =============================================================================
# SECTION 5 — WEIGHT OPTIMIZATION
# =============================================================================

class WeightOptimizer:
    """Optimize feature weights based on historical data"""
    
    def __init__(self):
        self.historical_races: List[Dict] = []
    
    def add_race(self, race_id: str, predictions: List[Dict], 
                 actual_winner: int, actual_placers: List[int]) -> None:
        """Record race for optimization"""
        self.historical_races.append({
            "race_id": race_id,
            "predictions": predictions,
            "winner": actual_winner,
            "placers": actual_placers,
            "timestamp": datetime.now().isoformat(),
        })
    
    def calculate_top3_accuracy(self) -> float:
        """Calculate hit rate on top 3"""
        if not self.historical_races:
            return 0.0
        
        hits = 0
        for race in self.historical_races:
            top3 = [r["number"] for r in race["predictions"][:3]]
            if race["winner"] in top3:
                hits += 1
        
        return hits / len(self.historical_races)
    
    def suggest_weight_adjustment(self, feature: str, 
                                  current_weight: float) -> float:
        """Suggest weight adjustment for underperforming feature"""
        # This is a placeholder - real implementation would use correlation analysis
        accuracy = self.calculate_top3_accuracy()
        
        if accuracy < 0.28:  # Below 28% baseline
            # Reduce reliance on underperforming feature
            return current_weight * 0.9
        elif accuracy > 0.35:  # Above 35% (good)
            # Increase reliance on good performers
            return current_weight * 1.05
        
        return current_weight


# =============================================================================
# SECTION 6 — CONFIGURATION TEMPLATES
# =============================================================================

# Conservative configuration (for new users)
CONSERVATIVE_CONFIG = ConfigSet(
    betting=BettingConfig(
        kelly_fraction=0.15,      # 15% of Kelly only
        max_bet_pct=0.01,         # 1% max per bet
        min_confidence=0.70,      # Only high-confidence bets
    ),
    mc=MCConfig(iterations_standard=2000),
)

# Balanced configuration (typical use)
BALANCED_CONFIG = ConfigSet(
    betting=BettingConfig(
        kelly_fraction=0.25,
        max_bet_pct=0.02,
        min_confidence=0.60,
    ),
    mc=MCConfig(iterations_standard=3000),
)

# Aggressive configuration (experienced users)
AGGRESSIVE_CONFIG = ConfigSet(
    betting=BettingConfig(
        kelly_fraction=0.40,
        max_bet_pct=0.03,
        min_confidence=0.45,
    ),
    mc=MCConfig(iterations_standard=4000),
)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example 1: Load/save configuration
    config = BALANCED_CONFIG
    config.to_json("config_balanced.json")
    config_loaded = ConfigSet.from_json("config_balanced.json")
    
    # Example 2: Backtesting
    backtester = Backtester()
    bet1 = BetRecord(
        race_id="PMU_2025_01_15_1",
        date="2025-01-15",
        horse_number=5,
        horse_name="Winner Horse",
        odds=3.50,
        model_prob=0.35,
        market_prob=0.25,
        bet_type="win",
        bet_amount=100.0,
        kelly_fraction=0.025,
    )
    backtester.add_bet(bet1)
    backtester.settle_bet("PMU_2025_01_15_1", 5, "win", 350.0)
    
    stats = backtester.get_statistics()
    print("Backtest Stats:", stats)
    
    # Example 3: Risk management
    risk_mgr = RiskManager(initial_bankroll=10000.0)
    status = risk_mgr.get_status()
    print("Risk Status:", status)
