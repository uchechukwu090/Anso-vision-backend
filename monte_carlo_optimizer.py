"""
Monte Carlo Optimizer - Dynamic TP/SL Based on Probabilistic Trend Projection
✅ FIXED: Now syncs with HMM state, volatility, and momentum
"""
import numpy as np
from typing import Dict


class MonteCarloOptimizer:
    """
    Monte Carlo simulation for dynamic TP/SL.
    ✅ NEW: Adapts horizon, quantiles, and model (GBM vs mean-reversion) based on HMM state
    """

    def __init__(self, n_simulations: int = 25_000, confidence_level: float = 0.90, seed: int | None = None):
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        self.rng = np.random.default_rng(seed)

        self.sl_percentile = (1 - confidence_level) * 100
        self.tp_percentile = (1 - (1 - confidence_level) / 2) * 100
        
        # ✅ NEW: State-based configuration
        self.state_config = {
            'TRENDING_UP': {'horizon_mult': 2.0, 'tp_pct': 75, 'sl_pct': 20, 'vol_mult': 1.2},
            'TRENDING_DOWN': {'horizon_mult': 2.0, 'tp_pct': 25, 'sl_pct': 80, 'vol_mult': 1.2},
            'RANGING': {'horizon_mult': 0.6, 'tp_pct': 60, 'sl_pct': 40, 'vol_mult': 0.8},
            'VOLATILE': {'horizon_mult': 0.4, 'tp_pct': 55, 'sl_pct': 45, 'vol_mult': 1.5}
        }

        print(f"🎲 Monte Carlo Optimizer Initialized (Sims: {self.n_simulations}, CL: {self.confidence_level:.0%})")

    def calculate_volatility(self, price_data: np.ndarray, window: int = 75) -> float:
        if len(price_data) < 2:
            return 0.0
        if len(price_data) > window:
            price_data = price_data[-window:]
        returns = np.diff(np.log(price_data))
        return float(np.std(returns) * np.sqrt(252))

    def calculate_drift(self, price_data: np.ndarray, window: int = 75) -> float:
        if len(price_data) < 2:
            return 0.0
        if len(price_data) > window:
            price_data = price_data[-window:]
        returns = np.diff(np.log(price_data))
        return float(np.mean(returns) * 252)
    
    def calculate_momentum(self, price_data: np.ndarray, window: int = 14) -> float:
        """Calculate momentum as percentage change over window"""
        if len(price_data) < window:
            return 0.0
        recent = price_data[-window:]
        pct_change = ((recent[-1] - recent[0]) / recent[0]) * 100
        return float(pct_change)

    def simulate_paths(self, current_price: float, mu_annual: float, sigma_annual: float, time_horizon: int) -> np.ndarray:
        mu_period = mu_annual / 252.0
        sigma_period = sigma_annual / np.sqrt(252.0)
        dt = 1.0
        
        paths = np.zeros((self.n_simulations, time_horizon + 1))
        paths[:, 0] = current_price
        
        z = self.rng.standard_normal((self.n_simulations, time_horizon))
        drift_term = (mu_period - 0.5 * sigma_period**2) * dt
        vol_term = sigma_period * np.sqrt(dt)

        for t in range(1, time_horizon + 1):
            paths[:, t] = paths[:, t - 1] * np.exp(drift_term + vol_term * z[:, t - 1])

        return paths
    
    def simulate_ou_paths(self, current_price: float, mean_price: float, mean_reversion_speed: float, sigma: float, time_horizon: int) -> np.ndarray:
        """Mean-reverting paths for ranging markets"""
        paths = np.zeros((self.n_simulations, time_horizon + 1))
        paths[:, 0] = current_price
        
        z = self.rng.standard_normal((self.n_simulations, time_horizon))
        
        for t in range(1, time_horizon + 1):
            dt = 1.0
            drift = mean_reversion_speed * (mean_price - paths[:, t-1])
            vol_term = sigma * np.sqrt(dt)
            paths[:, t] = paths[:, t-1] + drift * dt + vol_term * z[:, t-1]
        
        return paths

    def _detect_state(self, prices: np.ndarray, volatility: float, momentum: float, trend_state: str = None) -> str:
        """
        ✅ NEW: Detect market state from volatility and momentum if HMM state not provided
        """
        if trend_state and trend_state in self.state_config:
            return trend_state
        
        # Fallback: detect from volatility and momentum
        vol_percentile = volatility / (np.std([self.calculate_volatility(prices[max(0, i-50):i]) 
                                                for i in range(50, len(prices), 10)] or [volatility]) or volatility)
        
        if abs(momentum) > 2.0:  # Strong trend
            if momentum > 0:
                return 'TRENDING_UP'
            else:
                return 'TRENDING_DOWN'
        elif volatility > 0.015:  # High volatility
            return 'VOLATILE'
        else:
            return 'RANGING'

    def calculate_tp_sl(self, prices: np.ndarray, current_price: float, signal_type: str, 
                       time_horizon: int = 50, volatility: float = None, trend_state: str = None, 
                       momentum: float = None) -> Dict:
        """
        ✅ FIXED: Now adapts horizon and model based on HMM state, volatility, and momentum
        """
        if volatility is None:
            volatility = self.calculate_volatility(prices)
        
        if momentum is None:
            momentum = self.calculate_momentum(prices)
        
        # ✅ NEW: Detect market state and adapt
        state = self._detect_state(prices, volatility, momentum, trend_state)
        config = self.state_config.get(state, self.state_config['RANGING'])
        
        # ✅ NEW: Adjust horizon based on state
        adaptive_horizon = max(10, int(time_horizon * config['horizon_mult']))
        
        print(f"\n   📊 MC State Detection:")
        print(f"      State: {state}")
        print(f"      Volatility: {volatility:.6f}")
        print(f"      Momentum: {momentum:.2f}%")
        print(f"      Adaptive Horizon: {adaptive_horizon} (vs default {time_horizon})")
        print(f"      TP/SL Percentiles: {config['tp_pct']}/{config['sl_pct']}")
        
        mu = self.calculate_drift(prices)
        
        if volatility == 0.0:
            return {
                'tp': current_price * 1.002,
                'sl': current_price * 0.998,
                'median_target': current_price,
                'confidence': self.confidence_level,
                'volatility': 0.0,
                'state': state
            }
        
        # ✅ NEW: Use mean-reversion for ranging, GBM for trending
        if state == 'RANGING':
            mean_price = np.mean(prices[-50:])
            paths = self.simulate_ou_paths(current_price, mean_price, 0.1, volatility, adaptive_horizon)
        else:
            paths = self.simulate_paths(current_price, mu, volatility, adaptive_horizon)
        
        max_prices = np.max(paths, axis=1)
        min_prices = np.min(paths, axis=1)
        median_target = float(np.median(paths[:, -1]))
        
        # ✅ NEW: Use adaptive percentiles from state
        if signal_type == 'BUY':
            tp = float(np.percentile(max_prices, config['tp_pct']))
            sl = float(np.percentile(min_prices, config['sl_pct']))
        elif signal_type == 'SELL':
            tp = float(np.percentile(min_prices, 100 - config['tp_pct']))
            sl = float(np.percentile(max_prices, 100 - config['sl_pct']))
        else:
            tp, sl = current_price, current_price
        
        # Enforce minimum 1:1 R:R
        if signal_type == 'BUY':
            reward = tp - current_price
            risk = current_price - sl
            if risk <= 0 or (reward / risk) < 1.0:
                sl = current_price - max(reward, 1e-9)
        elif signal_type == 'SELL':
            reward = current_price - tp
            risk = sl - current_price
            if risk <= 0 or (reward / risk) < 1.0:
                sl = current_price + max(reward, 1e-9)
        
        return {
            'tp': tp,
            'sl': sl,
            'median_target': median_target,
            'confidence': self.confidence_level,
            'volatility': float(volatility),
            'state': state,
            'momentum': float(momentum),
            'adaptive_horizon': adaptive_horizon
        }

    def calculate_risk_metrics(self, price_data: np.ndarray, current_price: float,
                               tp: float, sl: float, signal_type: str = 'BUY') -> Dict[str, float]:
        if signal_type == 'BUY':
            reward = tp - current_price
            risk = current_price - sl
        else:
            reward = current_price - tp
            risk = sl - current_price
        
        rr = float(reward / risk) if risk > 0 else 0.0
        prob_tp = 0.5 if risk > 0 and reward > 0 else 0.0
        
        return {
            "risk_reward_ratio": rr,
            "potential_profit_pct": float((reward / current_price) * 100) if current_price > 0 else 0.0,
            "potential_loss_pct": float((risk / current_price) * 100) if current_price > 0 else 0.0,
            "prob_tp_hit": float(prob_tp),
            "prob_sl_hit": float(1.0 - prob_tp),
            "expected_value": float((reward * prob_tp) - (risk * (1.0 - prob_tp))),
            "expected_value_pct": float(((reward * prob_tp) - (risk * (1.0 - prob_tp))) / current_price * 100) if current_price > 0 else 0.0
        }
