import numpy as np
from scipy import stats
from typing import Dict, Tuple

class MonteCarloOptimizer:
    """
    Monte Carlo simulation for dynamic Take Profit (TP) and Stop Loss (SL) prediction.
    
    Advantages over HMM for TP/SL:
    1. Captures tail risk and fat tails in price distributions
    2. Generates realistic price paths considering volatility
    3. Provides probabilistic confidence intervals
    4. Adapts to changing market conditions
    5. No need for historical trade logs - uses price data directly
    """
    
    def __init__(self, n_simulations: int = 10000, confidence_level: float = 0.88):
        """
        Args:
            n_simulations: Number of Monte Carlo paths to simulate
            confidence_level: Confidence level for quantile calculations (0.88 = 88% for more conservative targets)
        """
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        self.percentile_low = (1 - confidence_level) / 2 * 100
        self.percentile_high = (1 + confidence_level) / 2 * 100
    
    def calculate_volatility(self, price_data: np.ndarray, window: int = 75) -> float:
        """
        Calculate annualized volatility from price returns.
        
        Args:
            price_data: 1D array of prices
            window: Rolling window for volatility calculation (default: 75 for more stable estimates)
            
        Returns:
            Annualized volatility (assuming 252 trading days/year)
        """
        if len(price_data) < window:
            window = max(2, len(price_data) - 1)
        
        returns = np.diff(np.log(price_data))
        if len(returns) < 2:
            return 0.02  # Default fallback
        
        # Calculate rolling volatility
        volatility = np.std(returns[-window:]) * np.sqrt(252)
        
        # Ensure volatility is within reasonable bounds
        volatility = np.clip(volatility, 0.01, 2.0)
        
        return volatility
    
    def calculate_drift(self, price_data: np.ndarray, window: int = 100) -> float:
        """
        Calculate drift (mean return) from recent price data.
        
        Args:
            price_data: 1D array of prices
            window: Number of recent periods to consider (default: 100 for more stable estimates)
            
        Returns:
            Annualized drift (mean return)
        """
        if len(price_data) < window:
            window = max(2, len(price_data) - 1)
        
        recent_prices = price_data[-window:]
        returns = np.diff(np.log(recent_prices))
        
        if len(returns) == 0:
            return 0.0
        
        drift = np.mean(returns) * 252  # Annualize
        
        # Clip drift to reasonable bounds to avoid extreme predictions
        drift = np.clip(drift, -1.0, 1.0)
        
        return drift
    
    def simulate_price_paths(self, current_price: float, drift: float, volatility: float, 
                           time_steps: int = 20, dt: float = 1/252) -> np.ndarray:
        """
        Simulate price paths using Geometric Brownian Motion (GBM).
        
        GBM: dS = μ*S*dt + σ*S*dW
        
        Args:
            current_price: Current market price
            drift: Expected return (annualized)
            volatility: Volatility (annualized)
            time_steps: Number of time steps to simulate
            dt: Time step size (default: 1 day as fraction of year)
            
        Returns:
            Array of shape (n_simulations, time_steps) with simulated price paths
        """
        paths = np.zeros((self.n_simulations, time_steps + 1))
        paths[:, 0] = current_price
        
        for t in range(1, time_steps + 1):
            # Random normal increments
            dW = np.random.normal(0, np.sqrt(dt), self.n_simulations)
            
            # GBM update
            paths[:, t] = paths[:, t-1] * np.exp(
                (drift - 0.5 * volatility**2) * dt + volatility * dW
            )
        
        return paths
    
    def calculate_tp_sl(self, price_data: np.ndarray, current_price: float, 
                       signal_type: str = 'BUY', time_horizon: int = 20) -> Dict[str, float]:
        """
        Calculate optimal TP and SL using Monte Carlo simulation.
        
        Args:
            price_data: Historical price data (1D array)
            current_price: Current market price
            signal_type: 'BUY' or 'SELL'
            time_horizon: Number of periods to simulate (default: 20 days)
            
        Returns:
            Dictionary with 'tp' (take profit) and 'sl' (stop loss) levels
        """
        if len(price_data) < 10:
            # Fallback for insufficient data
            if signal_type == 'BUY':
                return {
                    'tp': current_price * 1.02,
                    'sl': current_price * 0.98,
                    'confidence': 0.5,
                    'simulation_count': 0
                }
            else:
                return {
                    'tp': current_price * 0.98,
                    'sl': current_price * 1.02,
                    'confidence': 0.5,
                    'simulation_count': 0
                }
        
        # Calculate volatility and drift from recent data
        volatility = self.calculate_volatility(price_data)
        drift = self.calculate_drift(price_data)
        
        # Simulate price paths
        paths = self.simulate_price_paths(
            current_price=current_price,
            drift=drift,
            volatility=volatility,
            time_steps=time_horizon
        )
        
        # Get final prices across all simulations
        final_prices = paths[:, -1]
        
        # Calculate confidence intervals
        lower_bound = np.percentile(final_prices, self.percentile_low)
        upper_bound = np.percentile(final_prices, self.percentile_high)
        median_price = np.median(final_prices)
        
        if signal_type == 'BUY':
            # For long positions: TP is optimistic scenario, SL is pessimistic
            tp = upper_bound * 0.95  # Slightly below best case to allow profit taking
            sl = lower_bound * 1.01   # Slightly above worst case for stop loss
            
            # Ensure SL < current < TP
            sl = min(sl, current_price * 0.98)
            tp = max(tp, current_price * 1.02)
            
        else:  # SELL
            # For short positions: TP is pessimistic scenario, SL is optimistic
            tp = lower_bound * 1.05  # Slightly above best case to allow profit taking
            sl = upper_bound * 0.99  # Slightly below worst case for stop loss
            
            # Ensure TP < current < SL
            tp = min(tp, current_price * 0.98)
            sl = max(sl, current_price * 1.02)
        
        # Calculate confidence metric (0-1): how much the distribution favors the trade direction
        if signal_type == 'BUY':
            confidence = np.mean(final_prices > current_price) * self.confidence_level
        else:
            confidence = np.mean(final_prices < current_price) * self.confidence_level
        
        return {
            'tp': float(tp),
            'sl': float(sl),
            'median_target': float(median_price),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'confidence': float(confidence),
            'volatility': float(volatility),
            'drift': float(drift),
            'simulation_count': self.n_simulations,
        }
    
    def calculate_risk_metrics(self, price_data: np.ndarray, current_price: float,
                              tp: float, sl: float, signal_type: str = 'BUY') -> Dict[str, float]:
        """
        Calculate risk/reward metrics for the suggested TP/SL.
        
        Args:
            price_data: Historical price data
            current_price: Current market price
            tp: Take profit level
            sl: Stop loss level
            signal_type: 'BUY' or 'SELL'
            
        Returns:
            Dictionary with risk metrics
        """
        if signal_type == 'BUY':
            reward = tp - current_price
            risk = current_price - sl
        else:
            reward = current_price - tp
            risk = sl - current_price
        
        risk_reward_ratio = reward / risk if risk > 0 else 0
        potential_profit_pct = (reward / current_price) * 100 if current_price > 0 else 0
        potential_loss_pct = (risk / current_price) * 100 if current_price > 0 else 0
        
        # Calculate probability of reaching TP before SL using random walk theory
        if signal_type == 'BUY':
            distance_to_tp = tp - current_price
            distance_to_sl = current_price - sl
        else:
            distance_to_tp = current_price - tp
            distance_to_sl = sl - current_price
        
        total_distance = distance_to_tp + distance_to_sl
        if total_distance > 0:
            prob_tp = distance_to_tp / total_distance
        else:
            prob_tp = 0.5
        
        # Expected value of the trade
        expected_value = (prob_tp * reward) - ((1 - prob_tp) * risk)
        
        return {
            'risk_reward_ratio': float(risk_reward_ratio),
            'potential_profit_pct': float(potential_profit_pct),
            'potential_loss_pct': float(potential_loss_pct),
            'prob_tp': float(prob_tp),
            'prob_sl': float(1 - prob_tp),
            'expected_value': float(expected_value),
            'expected_value_pct': float((expected_value / current_price) * 100) if current_price > 0 else 0,
        }


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    
    # Generate sample price data (trending upward with volatility)
    prices = np.cumsum(np.random.normal(0.001, 0.02, 200)) + 100
    current_price = prices[-1]
    
    # Initialize optimizer
    optimizer = MonteCarloOptimizer(n_simulations=10000, confidence_level=0.95)
    
    # Calculate TP/SL for a BUY signal
    result = optimizer.calculate_tp_sl(prices, current_price, signal_type='BUY', time_horizon=20)
    
    print("Monte Carlo Analysis Results:")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Take Profit: ${result['tp']:.2f}")
    print(f"Stop Loss: ${result['sl']:.2f}")
    print(f"Median Target: ${result['median_target']:.2f}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Volatility (annualized): {result['volatility']:.2%}")
    
    # Calculate risk metrics
    risk_metrics = optimizer.calculate_risk_metrics(prices, current_price, result['tp'], result['sl'], 'BUY')
    
    print("\nRisk Metrics:")
    print(f"Risk/Reward Ratio: {risk_metrics['risk_reward_ratio']:.2f}:1")
    print(f"Potential Profit: {risk_metrics['potential_profit_pct']:.2f}%")
    print(f"Potential Loss: {risk_metrics['potential_loss_pct']:.2f}%")
    print(f"Probability of TP: {risk_metrics['prob_tp']:.2%}")
    print(f"Expected Value: {risk_metrics['expected_value_pct']:.2f}%")
