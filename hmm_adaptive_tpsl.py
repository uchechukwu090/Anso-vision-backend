"""
HMM-ADAPTIVE TP/SL CALCULATOR
Calculates optimal TP/SL based on what the HMM is seeing in the market
Uses volatility, trend strength, and market state to adapt dynamically
"""
import numpy as np
from typing import Dict


class HMMAdaptiveTPSL:
    """
    Calculates TP/SL that adapts to HMM market state
    
    State-based logic:
    - TRENDING_UP (State 2): Wider TP, tighter SL (trend following)
    - RANGING (State 1): Balanced TP/SL (mean reversion)
    - TRENDING_DOWN (State 0): Wider SL, tighter TP (counter-trend)
    """
    
    def __init__(self):
        # State configurations (multipliers for volatility-based sizing)
        self.state_config = {
            0: {  # TRENDING_DOWN / BEARISH
                'name': 'BEARISH',
                'buy_tp_mult': 1.5,   # Conservative TP for BUY (counter-trend)
                'buy_sl_mult': 2.0,   # Wider SL for BUY (needs room)
                'sell_tp_mult': 2.5,  # Aggressive TP for SELL (with trend)
                'sell_sl_mult': 1.0   # Tight SL for SELL (trend should hold)
            },
            1: {  # RANGING / NEUTRAL
                'name': 'RANGING',
                'buy_tp_mult': 1.8,   # Balanced
                'buy_sl_mult': 1.2,   # Balanced
                'sell_tp_mult': 1.8,  # Balanced
                'sell_sl_mult': 1.2   # Balanced
            },
            2: {  # TRENDING_UP / BULLISH
                'name': 'BULLISH',
                'buy_tp_mult': 2.5,   # Aggressive TP for BUY (with trend)
                'buy_sl_mult': 1.0,   # Tight SL for BUY (trend should hold)
                'sell_tp_mult': 1.5,  # Conservative TP for SELL (counter-trend)
                'sell_sl_mult': 2.0   # Wider SL for SELL (needs room)
            }
        }
    
    def calculate_volatility(self, prices: np.ndarray, window: int = 14) -> float:
        """
        Calculate true volatility (ATR-style)
        
        Math: ATR = Average of |High - Low| over N periods
        Simplified: std(price_changes) * sqrt(window)
        """
        if len(prices) < window:
            window = len(prices)
        
        price_changes = np.diff(prices[-window:])
        volatility = np.std(price_changes) * np.sqrt(window)
        
        return float(volatility)
    
    def calculate_trend_strength(self, prices: np.ndarray, window: int = 20) -> float:
        """
        Calculate trend strength using R-squared
        
        Returns: 0.0 to 1.0 (0 = no trend, 1 = perfect trend)
        """
        if len(prices) < window:
            window = len(prices)
        
        recent = prices[-window:]
        x = np.arange(len(recent))
        y = recent
        
        # Linear regression correlation
        correlation = np.corrcoef(x, y)[0, 1]
        r_squared = correlation ** 2
        
        return float(r_squared)
    
    def calculate_tp_sl(self, 
                       prices: np.ndarray,
                       current_price: float,
                       signal_type: str,
                       hmm_state: int,
                       state_confidence: float) -> Dict:
        """
        Calculate HMM-adaptive TP and SL
        
        Args:
            prices: Price history
            current_price: Current market price
            signal_type: 'BUY' or 'SELL'
            hmm_state: 0 (BEARISH), 1 (RANGING), 2 (BULLISH)
            state_confidence: HMM state confidence (0-1)
        
        Returns:
            {'tp': float, 'sl': float, 'method': str}
        """
        
        # Get state configuration
        config = self.state_config.get(hmm_state, self.state_config[1])
        
        # Calculate base volatility
        volatility = self.calculate_volatility(prices, window=14)
        
        # Calculate trend strength
        trend_strength = self.calculate_trend_strength(prices, window=20)
        
        # Adjust volatility based on trend strength
        # Stronger trend = can use wider stops/targets
        adaptive_vol = volatility * (1 + trend_strength * 0.5)
        
        # Get multipliers based on signal type and HMM state
        if signal_type == 'BUY':
            tp_mult = config['buy_tp_mult']
            sl_mult = config['buy_sl_mult']
        else:  # SELL
            tp_mult = config['sell_tp_mult']
            sl_mult = config['sell_sl_mult']
        
        # Adjust multipliers based on state confidence
        # Lower confidence = tighter stops/targets
        confidence_adjustment = 0.7 + (state_confidence * 0.3)  # Range: 0.7 to 1.0
        tp_mult *= confidence_adjustment
        sl_mult *= confidence_adjustment
        
        # Calculate TP and SL
        if signal_type == 'BUY':
            tp = current_price + (adaptive_vol * tp_mult)
            sl = current_price - (adaptive_vol * sl_mult)
        else:  # SELL
            tp = current_price - (adaptive_vol * tp_mult)
            sl = current_price + (adaptive_vol * sl_mult)
        
        # Ensure minimum R:R of 1:1
        if signal_type == 'BUY':
            reward = tp - current_price
            risk = current_price - sl
            if risk > 0 and (reward / risk) < 1.0:
                tp = current_price + risk  # Force 1:1
        else:
            reward = current_price - tp
            risk = sl - current_price
            if risk > 0 and (reward / risk) < 1.0:
                tp = current_price - risk  # Force 1:1
        
        # Calculate final R:R
        if signal_type == 'BUY':
            final_rr = (tp - current_price) / (current_price - sl) if (current_price - sl) > 0 else 0
        else:
            final_rr = (current_price - tp) / (sl - current_price) if (sl - current_price) > 0 else 0
        
        return {
            'tp': float(tp),
            'sl': float(sl),
            'volatility': float(adaptive_vol),
            'trend_strength': float(trend_strength),
            'hmm_state_name': config['name'],
            'tp_multiplier': float(tp_mult),
            'sl_multiplier': float(sl_mult),
            'risk_reward': float(final_rr),
            'method': 'HMM_ADAPTIVE'
        }


if __name__ == '__main__':
    """Test the calculator"""
    # Generate test data
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(200) * 0.01) + 100
    current = prices[-1]
    
    calculator = HMMAdaptiveTPSL()
    
    print("\n" + "="*70)
    print("HMM-ADAPTIVE TP/SL TEST")
    print("="*70)
    print(f"Current Price: {current:.4f}")
    
    for state in [0, 1, 2]:
        for signal in ['BUY', 'SELL']:
            result = calculator.calculate_tp_sl(
                prices=prices,
                current_price=current,
                signal_type=signal,
                hmm_state=state,
                state_confidence=0.85
            )
            
            print(f"\n{result['hmm_state_name']} | {signal}")
            print(f"  TP: {result['tp']:.4f} (mult: {result['tp_multiplier']:.2f})")
            print(f"  SL: {result['sl']:.4f} (mult: {result['sl_multiplier']:.2f})")
            print(f"  R:R: {result['risk_reward']:.2f}:1")
            print(f"  Volatility: {result['volatility']:.6f}")
            print(f"  Trend Strength: {result['trend_strength']:.2%}")
