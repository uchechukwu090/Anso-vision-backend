"""
ATR-Based TP/SL Calculator - Much Better Than Monte Carlo for Forex/Crypto
Uses Average True Range for realistic stop losses and take profits
"""
import numpy as np
from typing import Dict

class ATRCalculator:
    """
    Professional-grade TP/SL using ATR (Average True Range)
    Used by institutional traders for proper risk management
    """
    
    def __init__(self, atr_period: int = 14, tp_multiplier: float = 2.0, sl_multiplier: float = 1.0):
        """
        Args:
            atr_period: Period for ATR calculation (default 14)
            tp_multiplier: Take profit as multiple of ATR (default 2.0 for 2:1 R:R)
            sl_multiplier: Stop loss as multiple of ATR (default 1.0)
        """
        self.atr_period = atr_period
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
    
    def calculate_atr(self, prices: np.ndarray, period: int = None) -> float:
        """
        Calculate Average True Range
        ATR = Average of True Range over N periods
        True Range = Max(H-L, |H-PC|, |L-PC|)
        
        For close-only data: Use close as proxy
        """
        if period is None:
            period = self.atr_period
        
        if len(prices) < period + 1:
            period = max(2, len(prices) - 1)
        
        # For close-only data, calculate range between consecutive closes
        true_ranges = []
        for i in range(1, len(prices)):
            # Simplified TR for close-only: absolute change
            tr = abs(prices[i] - prices[i-1])
            true_ranges.append(tr)
        
        if len(true_ranges) < period:
            atr = np.mean(true_ranges) if true_ranges else prices[-1] * 0.001
        else:
            atr = np.mean(true_ranges[-period:])
        
        # Ensure ATR is reasonable (0.1% to 5% of price)
        min_atr = prices[-1] * 0.001  # 0.1%
        max_atr = prices[-1] * 0.05   # 5%
        atr = np.clip(atr, min_atr, max_atr)
        
        return float(atr)
    
    def calculate_tp_sl(self, prices: np.ndarray, current_price: float, 
                       signal_type: str = 'BUY') -> Dict[str, float]:
        """
        Calculate TP and SL using ATR
        
        Args:
            prices: Historical price data
            current_price: Current market price
            signal_type: 'BUY' or 'SELL'
            
        Returns:
            Dict with tp, sl, atr, risk_reward_ratio
        """
        atr = self.calculate_atr(prices)
        
        if signal_type == 'BUY':
            entry = current_price * 1.0005  # Small spread adjustment
            tp = entry + (self.tp_multiplier * atr)
            sl = entry - (self.sl_multiplier * atr)
        else:  # SELL
            entry = current_price * 0.9995  # Small spread adjustment
            tp = entry - (self.tp_multiplier * atr)
            sl = entry + (self.sl_multiplier * atr)
        
        # Calculate metrics
        if signal_type == 'BUY':
            profit_distance = tp - entry
            loss_distance = entry - sl
        else:
            profit_distance = entry - tp
            loss_distance = sl - entry
        
        rr_ratio = profit_distance / loss_distance if loss_distance > 0 else 0
        profit_pct = (profit_distance / current_price) * 100
        loss_pct = (loss_distance / current_price) * 100
        
        return {
            'tp': float(tp),
            'sl': float(sl),
            'entry': float(entry),
            'atr': float(atr),
            'atr_pct': float((atr / current_price) * 100),
            'risk_reward_ratio': float(rr_ratio),
            'profit_potential_pct': float(profit_pct),
            'loss_potential_pct': float(loss_pct),
            'tp_distance_pips': float(profit_distance),
            'sl_distance_pips': float(loss_distance)
        }
    
    def calculate_position_size(self, account_balance: float, risk_per_trade_pct: float,
                               entry: float, sl: float) -> float:
        """
        Calculate position size based on risk management
        Risk no more than X% of account per trade
        
        Args:
            account_balance: Total account balance
            risk_per_trade_pct: % of account to risk (e.g., 1.0 for 1%)
            entry: Entry price
            sl: Stop loss price
            
        Returns:
            Position size in lots/units
        """
        risk_amount = account_balance * (risk_per_trade_pct / 100)
        price_risk = abs(entry - sl)
        
        if price_risk == 0:
            return 0
        
        position_size = risk_amount / price_risk
        return float(position_size)


def calculate_dynamic_atr_multipliers(prices: np.ndarray, volatility_regime: str = 'normal') -> tuple:
    """
    Calculate adaptive ATR multipliers based on market conditions
    
    Args:
        prices: Historical prices
        volatility_regime: 'low', 'normal', 'high'
        
    Returns:
        (tp_multiplier, sl_multiplier)
    """
    # Calculate recent volatility
    returns = np.diff(np.log(prices[-50:]))
    vol = np.std(returns) * np.sqrt(252)  # Annualized
    
    # Low volatility: Tighter stops, smaller targets
    if vol < 0.1 or volatility_regime == 'low':
        return (1.5, 0.8)  # 1.5:0.8 ≈ 1.88:1 R:R
    
    # High volatility: Wider stops, bigger targets
    elif vol > 0.3 or volatility_regime == 'high':
        return (3.0, 1.5)  # 3.0:1.5 = 2:1 R:R
    
    # Normal volatility
    else:
        return (2.0, 1.0)  # 2:1 R:R


# Test
if __name__ == '__main__':
    np.random.seed(42)
    
    # Generate test EUR/USD prices around 1.1500
    prices = np.cumsum(np.random.normal(0, 0.0005, 200)) + 1.1500
    current_price = prices[-1]
    
    print("="*60)
    print("ATR-BASED TP/SL CALCULATOR TEST")
    print("="*60)
    print(f"\nCurrent Price: {current_price:.5f}")
    
    # Test BUY signal
    atr_calc = ATRCalculator(atr_period=14, tp_multiplier=2.0, sl_multiplier=1.0)
    buy_result = atr_calc.calculate_tp_sl(prices, current_price, 'BUY')
    
    print("\n--- BUY SIGNAL ---")
    print(f"Entry:  {buy_result['entry']:.5f}")
    print(f"TP:     {buy_result['tp']:.5f} (+{buy_result['profit_potential_pct']:.3f}%)")
    print(f"SL:     {buy_result['sl']:.5f} (-{buy_result['loss_potential_pct']:.3f}%)")
    print(f"ATR:    {buy_result['atr']:.5f} ({buy_result['atr_pct']:.3f}%)")
    print(f"R:R:    {buy_result['risk_reward_ratio']:.2f}:1")
    
    # Test SELL signal
    sell_result = atr_calc.calculate_tp_sl(prices, current_price, 'SELL')
    
    print("\n--- SELL SIGNAL ---")
    print(f"Entry:  {sell_result['entry']:.5f}")
    print(f"TP:     {sell_result['tp']:.5f} (+{sell_result['profit_potential_pct']:.3f}%)")
    print(f"SL:     {sell_result['sl']:.5f} (-{sell_result['loss_potential_pct']:.3f}%)")
    print(f"ATR:    {sell_result['atr']:.5f} ({sell_result['atr_pct']:.3f}%)")
    print(f"R:R:    {sell_result['risk_reward_ratio']:.2f}:1")
    
    # Position sizing example
    account = 10000  # $10,000 account
    risk_pct = 1.0   # Risk 1% per trade
    position = atr_calc.calculate_position_size(account, risk_pct, sell_result['entry'], sell_result['sl'])
    print(f"\nPosition Size (1% risk): {position:.2f} units")
    print(f"Risk Amount: ${account * (risk_pct/100):.2f}")
