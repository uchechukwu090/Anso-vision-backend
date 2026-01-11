"""
Pure Monte Carlo Trading Engine
Implements Monte Carlo simulation for trading scenarios
"""
import numpy as np
from typing import Dict, List, Optional


class MonteCarloTradingEngine:
    """
    Monte Carlo simulation engine for trading analysis.
    Simulates price paths and evaluates trading scenarios.
    """
    
    def __init__(self, n_simulations: int = 10000, seed: Optional[int] = None):
        """
        Initialize the Monte Carlo engine.
        
        Args:
            n_simulations: Number of Monte Carlo simulations to run
            seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)
        
    def simulate_price_paths(self, 
                            current_price: float,
                            drift: float,
                            volatility: float,
                            time_horizon: int) -> np.ndarray:
        """
        Simulate future price paths using Geometric Brownian Motion.
        
        Args:
            current_price: Current asset price
            drift: Expected return (annualized)
            volatility: Volatility (annualized)
            time_horizon: Number of time steps to simulate
            
        Returns:
            Array of simulated price paths (n_simulations x time_horizon)
        """
        dt = 1.0 / 252.0  # Daily time step
        paths = np.zeros((self.n_simulations, time_horizon + 1))
        paths[:, 0] = current_price
        
        for t in range(1, time_horizon + 1):
            z = self.rng.standard_normal(self.n_simulations)
            paths[:, t] = paths[:, t-1] * np.exp(
                (drift - 0.5 * volatility**2) * dt + 
                volatility * np.sqrt(dt) * z
            )
        
        return paths
    
    def evaluate_trade_scenario(self,
                                current_price: float,
                                entry_price: float,
                                tp_price: float,
                                sl_price: float,
                                drift: float,
                                volatility: float,
                                max_holding_period: int = 50) -> Dict:
        """
        Evaluate a trade scenario using Monte Carlo simulation.
        
        Args:
            current_price: Current market price
            entry_price: Planned entry price
            tp_price: Take profit price
            sl_price: Stop loss price
            drift: Expected drift
            volatility: Price volatility
            max_holding_period: Maximum candles to hold
            
        Returns:
            Dict with win probability, expected return, etc.
        """
        paths = self.simulate_price_paths(
            current_price, drift, volatility, max_holding_period
        )
        
        # Determine trade direction
        is_long = tp_price > entry_price
        
        wins = 0
        losses = 0
        returns = []
        
        for path in paths:
            hit_tp = False
            hit_sl = False
            
            for price in path:
                if is_long:
                    if price >= tp_price:
                        hit_tp = True
                        break
                    elif price <= sl_price:
                        hit_sl = True
                        break
                else:  # Short
                    if price <= tp_price:
                        hit_tp = True
                        break
                    elif price >= sl_price:
                        hit_sl = True
                        break
            
            if hit_tp:
                wins += 1
                ret = abs(tp_price - entry_price) / entry_price
                returns.append(ret)
            elif hit_sl:
                losses += 1
                ret = -abs(sl_price - entry_price) / entry_price
                returns.append(ret)
            else:
                # Neither hit - use final price
                final_price = path[-1]
                if is_long:
                    ret = (final_price - entry_price) / entry_price
                else:
                    ret = (entry_price - final_price) / entry_price
                returns.append(ret)
        
        win_prob = wins / self.n_simulations
        avg_return = np.mean(returns) if returns else 0.0
        
        return {
            'win_probability': float(win_prob),
            'loss_probability': float(losses / self.n_simulations),
            'expected_return': float(avg_return),
            'expected_return_pct': float(avg_return * 100),
            'wins': wins,
            'losses': losses,
            'total_simulations': self.n_simulations
        }
    
    def calculate_optimal_position_size(self,
                                        account_balance: float,
                                        win_prob: float,
                                        avg_win: float,
                                        avg_loss: float,
                                        risk_percent: float = 0.02) -> Dict:
        """
        Calculate optimal position size using Kelly Criterion and risk management.
        
        Args:
            account_balance: Total account balance
            win_prob: Probability of winning
            avg_win: Average win amount
            avg_loss: Average loss amount
            risk_percent: Maximum risk per trade (default 2%)
            
        Returns:
            Dict with position size recommendations
        """
        # Kelly Criterion: f = (p * b - q) / b
        # where p = win prob, q = loss prob, b = win/loss ratio
        if avg_loss == 0 or win_prob == 0:
            return {
                'kelly_fraction': 0.0,
                'position_size': 0.0,
                'risk_amount': 0.0
            }
        
        b = abs(avg_win / avg_loss)
        q = 1 - win_prob
        kelly_fraction = (win_prob * b - q) / b
        
        # Cap Kelly at 25% (full Kelly is too aggressive)
        kelly_fraction = max(0, min(kelly_fraction, 0.25))
        
        # Also consider fixed risk percent
        max_risk_amount = account_balance * risk_percent
        
        # Use the more conservative of the two
        kelly_position = account_balance * kelly_fraction
        risk_position = max_risk_amount / abs(avg_loss) if avg_loss != 0 else 0
        
        recommended_position = min(kelly_position, risk_position)
        
        return {
            'kelly_fraction': float(kelly_fraction),
            'kelly_position_size': float(kelly_position),
            'risk_based_position': float(risk_position),
            'recommended_position': float(recommended_position),
            'risk_amount': float(max_risk_amount)
        }
