"""
Monte Carlo Optimizer - Dynamic TP/SL Based on Probabilistic Trend Projection
Core risk/exit logic replacing fixed ATR multipliers.
"""
import numpy as np
from typing import Dict


class MonteCarloOptimizer:
    """
    Monte Carlo simulation for dynamic Take Profit (TP) and Stop Loss (SL) prediction.

    TP/SL are set from the distribution of extreme prices (max/min) reached
    over a defined horizon using GBM paths.
    """

    def __init__(self, n_simulations: int = 25_000, confidence_level: float = 0.90, seed: int | None = None):
        """
        Args:
            n_simulations: Number of Monte Carlo paths to simulate (higher = more stable quantiles)
            confidence_level: Confidence level for quantile calculations (e.g., 0.90 for 90%)
            seed: Optional RNG seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        self.rng = np.random.default_rng(seed)

        # SL uses outer tail (e.g., 5%), TP uses inner tail (e.g., 90% of max for BUY)
        self.sl_percentile = (1 - confidence_level) * 100        # e.g., 10% if confidence=0.90
        self.tp_percentile = (1 - (1 - confidence_level) / 2) * 100  # e.g., 95% (unused in current scheme)

        print(f"🎲 Monte Carlo Optimizer Initialized (Sims: {self.n_simulations}, CL: {self.confidence_level:.0%})")

    # --------- estimation ----------
    def calculate_volatility(self, price_data: np.ndarray, window: int = 75) -> float:
        """Annualized volatility from log returns."""
        if len(price_data) < 2:
            return 0.0
        if len(price_data) > window:
            price_data = price_data[-window:]
        returns = np.diff(np.log(price_data))
        return float(np.std(returns) * np.sqrt(252))

    def calculate_drift(self, price_data: np.ndarray, window: int = 75) -> float:
        """Annualized drift from log returns."""
        if len(price_data) < 2:
            return 0.0
        if len(price_data) > window:
            price_data = price_data[-window:]
        returns = np.diff(np.log(price_data))
        return float(np.mean(returns) * 252)

    # --------- simulation ----------
    def simulate_paths(self, current_price: float, mu_annual: float, sigma_annual: float, time_horizon: int) -> np.ndarray:
        """
        Generate GBM paths using per-bar drift/vol implied by annualized inputs.
        Assumes 252 trading periods per year.
        """
        mu_period = mu_annual / 252.0
        sigma_period = sigma_annual / np.sqrt(252.0)

        dt = 1.0
        paths = np.zeros((self.n_simulations, time_horizon + 1))
        paths[:, 0] = current_price

        # Draw all shocks at once for vectorized performance
        z = self.rng.standard_normal((self.n_simulations, time_horizon))
        drift_term = (mu_period - 0.5 * sigma_period**2) * dt
        vol_term = sigma_period * np.sqrt(dt)

        for t in range(1, time_horizon + 1):
            paths[:, t] = paths[:, t - 1] * np.exp(drift_term + vol_term * z[:, t - 1])

        return paths

    # --------- TP/SL ----------
    def calculate_tp_sl(self, prices: np.ndarray, current_price: float, signal_type: str, time_horizon: int = 50) -> Dict:
        """
        Calculate dynamic TP/SL using quantiles of path extremes.
        - For BUY: TP = high quantile of max path price, SL = low quantile of min path price.
        - For SELL: TP = low quantile of min path price, SL = high quantile of max path price.
        """
        sigma = self.calculate_volatility(prices)
        mu = self.calculate_drift(prices)

        if sigma == 0.0:
            return {
                'tp': current_price * 1.002,
                'sl': current_price * 0.998,
                'median_target': current_price,
                'confidence': self.confidence_level,
                'volatility': 0.0
            }

        paths = self.simulate_paths(current_price, mu, sigma, time_horizon)

        # Path extremes
        max_prices = np.max(paths, axis=1)
        min_prices = np.min(paths, axis=1)
        median_target = float(np.median(paths[:, -1]))

        if signal_type == 'BUY':
            tp = float(np.percentile(max_prices, self.confidence_level * 100))             # e.g., 90% of max
            sl = float(np.percentile(min_prices, self.sl_percentile))                     # e.g., 10% of min
        elif signal_type == 'SELL':
            tp = float(np.percentile(min_prices, 100 - (self.confidence_level * 100)))    # e.g., 10% of min
            sl = float(np.percentile(max_prices, 100 - self.sl_percentile))               # e.g., 90% of max
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
            'volatility': float(sigma)
        }

    # --------- risk ----------
    def calculate_risk_metrics(self, price_data: np.ndarray, current_price: float,
                               tp: float, sl: float, signal_type: str = 'BUY') -> Dict[str, float]:
        """
        Calculate basic risk/reward metrics and a simple TP-before-SL probability proxy.
        """
        if signal_type == 'BUY':
            reward = tp - current_price
            risk = current_price - sl
            distance_to_tp = reward
            distance_to_sl = risk
        else:
            reward = current_price - tp
            risk = sl - current_price
            distance_to_tp = reward
            distance_to_sl = risk

        rr = float(reward / risk) if risk > 0 else 0.0
        profit_pct = float((reward / current_price) * 100.0) if current_price > 0 else 0.0
        loss_pct = float((risk / current_price) * 100.0) if current_price > 0 else 0.0

        total_distance = distance_to_tp + distance_to_sl
        prob_tp = float(distance_to_tp / total_distance) if total_distance > 0 else 0.5
        expected_value = float(prob_tp * reward - (1 - prob_tp) * risk)
        expected_value_pct = float((expected_value / current_price) * 100.0) if current_price > 0 else 0.0

        return {
            'risk_reward_ratio': rr,
            'potential_profit_pct': profit_pct,
            'potential_loss_pct': loss_pct,
            'prob_tp': prob_tp,
            'prob_sl': float(1.0 - prob_tp),
            'expected_value': expected_value,
            'expected_value_pct': expected_value_pct
        }


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    prices = np.cumsum(np.random.normal(0.001, 0.02, 200)) + 100
    current_price = float(prices[-1])

    optimizer = MonteCarloOptimizer(n_simulations=10_000, confidence_level=0.95, seed=42)
    result = optimizer.calculate_tp_sl(prices, current_price, signal_type='BUY', time_horizon=20)

    print("Monte Carlo Analysis Results:")
    print(f"Current Price: {current_price:.2f}")
    print(f"Take Profit: {result['tp']:.2f}")
    print(f"Stop Loss: {result['sl']:.2f}")
    print(f"Median Target: {result['median_target']:.2f}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Volatility (annualized): {result['volatility']:.2%}")

    risk = optimizer.calculate_risk_metrics(prices, current_price, result['tp'], result['sl'], 'BUY')
    print("\nRisk Metrics:")
    print(f"R:R: {risk['risk_reward_ratio']:.2f}:1")
    print(f"Potential Profit: {risk['potential_profit_pct']:.2f}%")
    print(f"Potential Loss: {risk['potential_loss_pct']:.2f}%")
    print(f"Probability of TP: {risk['prob_tp']:.2%}")
    print(f"Expected Value: {risk['expected_value_pct']:.2f}%")
