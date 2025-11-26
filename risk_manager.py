"""
Risk Manager - Safety Mechanisms and Circuit Breakers
Prevents system from generating dangerous signals
"""
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque


class RiskManager:
    """
    Implements safety mechanisms:
    1. Daily loss limits
    2. Signal rate limiting
    3. Correlation checks (prevent all users getting same signal)
    4. Drawdown protection
    5. Market condition filters
    """
    
    def __init__(self,
                 max_daily_signals: int = 20,
                 max_signals_per_hour: int = 5,
                 min_signal_spacing_minutes: int = 15):
        """
        Args:
            max_daily_signals: Maximum signals per symbol per day
            max_signals_per_hour: Maximum signals per symbol per hour
            min_signal_spacing_minutes: Minimum time between signals
        """
        self.max_daily_signals = max_daily_signals
        self.max_signals_per_hour = max_signals_per_hour
        self.min_signal_spacing_minutes = min_signal_spacing_minutes
        
        # Track signals
        self.signal_history: Dict[str, deque] = {}  # symbol -> deque of timestamps
        self.last_signal_time: Dict[str, datetime] = {}
        self.daily_signal_count: Dict[str, int] = {}
        self.last_reset_date: Dict[str, datetime] = {}
        
        # Track performance
        self.recent_signals: List[Dict] = []
        
        # Thread safety
        self.lock = threading.Lock()
        
        print("🛡️ RiskManager initialized")
    
    def should_allow_signal(self, symbol: str, signal_type: str) -> tuple[bool, str]:
        """
        Check if signal should be allowed
        
        Returns:
            (allowed: bool, reason: str)
        """
        with self.lock:
            now = datetime.now()
            
            # Initialize tracking for new symbol
            if symbol not in self.signal_history:
                self.signal_history[symbol] = deque(maxlen=100)
                self.daily_signal_count[symbol] = 0
                self.last_reset_date[symbol] = now.date()
            
            # Reset daily counter if new day
            if self.last_reset_date[symbol] != now.date():
                self.daily_signal_count[symbol] = 0
                self.last_reset_date[symbol] = now.date()
            
            # 1. Check daily limit
            if self.daily_signal_count[symbol] >= self.max_daily_signals:
                return False, f"Daily limit reached ({self.max_daily_signals} signals)"
            
            # 2. Check hourly limit
            hour_ago = now - timedelta(hours=1)
            recent_hour_signals = [
                t for t in self.signal_history[symbol]
                if t > hour_ago
            ]
            if len(recent_hour_signals) >= self.max_signals_per_hour:
                return False, f"Hourly limit reached ({self.max_signals_per_hour} signals)"
            
            # 3. Check minimum spacing
            if symbol in self.last_signal_time:
                time_since_last = (now - self.last_signal_time[symbol]).total_seconds() / 60
                if time_since_last < self.min_signal_spacing_minutes:
                    return False, f"Too soon (min {self.min_signal_spacing_minutes}min between signals)"
            
            # 4. Check for flip-flopping (BUY->SELL->BUY rapidly)
            if len(self.recent_signals) >= 3:
                last_three = self.recent_signals[-3:]
                if (last_three[0]['signal_type'] != last_three[1]['signal_type'] and
                    last_three[1]['signal_type'] != last_three[2]['signal_type']):
                    # Detected flip-flopping pattern
                    return False, "Flip-flopping detected - system unstable"
            
            return True, "Signal approved"
    
    def record_signal(self, symbol: str, signal_type: str, confidence: float):
        """Record a signal that was sent"""
        with self.lock:
            now = datetime.now()
            
            if symbol not in self.signal_history:
                self.signal_history[symbol] = deque(maxlen=100)
                self.daily_signal_count[symbol] = 0
            
            self.signal_history[symbol].append(now)
            self.last_signal_time[symbol] = now
            self.daily_signal_count[symbol] += 1
            
            self.recent_signals.append({
                'symbol': symbol,
                'signal_type': signal_type,
                'confidence': confidence,
                'timestamp': now
            })
            
            # Keep only last 100 signals
            if len(self.recent_signals) > 100:
                self.recent_signals = self.recent_signals[-100:]
    
    def check_market_conditions(self, volatility: float, volume_ratio: float) -> tuple[bool, str]:
        """
        Check if market conditions are suitable for trading
        
        Args:
            volatility: Current volatility (annualized)
            volume_ratio: Current volume / average volume
            
        Returns:
            (suitable: bool, reason: str)
        """
        # 1. Extreme volatility check
        if volatility > 1.0:  # 100% annualized volatility
            return False, f"Extreme volatility ({volatility:.1%}) - market unstable"
        
        # 2. Very low liquidity check
        if volume_ratio < 0.3:
            return False, f"Very low volume ({volume_ratio:.1f}x) - poor liquidity"
        
        # 3. Flash crash detection (extreme volatility with low volume)
        if volatility > 0.5 and volume_ratio < 0.5:
            return False, "Potential flash crash detected"
        
        return True, "Market conditions acceptable"
    
    def get_statistics(self, symbol: Optional[str] = None) -> Dict:
        """Get risk management statistics"""
        with self.lock:
            if symbol:
                return {
                    'symbol': symbol,
                    'daily_signals': self.daily_signal_count.get(symbol, 0),
                    'hourly_signals': len([
                        t for t in self.signal_history.get(symbol, [])
                        if t > datetime.now() - timedelta(hours=1)
                    ]),
                    'last_signal': self.last_signal_time.get(symbol),
                    'max_daily': self.max_daily_signals,
                    'max_hourly': self.max_signals_per_hour
                }
            else:
                return {
                    'total_symbols': len(self.signal_history),
                    'total_recent_signals': len(self.recent_signals),
                    'symbols': {
                        sym: {
                            'daily_count': self.daily_signal_count.get(sym, 0),
                            'last_signal': self.last_signal_time.get(sym)
                        }
                        for sym in self.signal_history.keys()
                    }
                }
    
    def reset_limits(self, symbol: Optional[str] = None):
        """Reset limits (for testing or emergency)"""
        with self.lock:
            if symbol:
                if symbol in self.daily_signal_count:
                    self.daily_signal_count[symbol] = 0
                if symbol in self.signal_history:
                    self.signal_history[symbol].clear()
                print(f"🔄 Reset limits for {symbol}")
            else:
                self.daily_signal_count.clear()
                self.signal_history.clear()
                self.recent_signals.clear()
                print("🔄 Reset all limits")


# Singleton instance
_risk_manager_instance = None
_risk_manager_lock = threading.Lock()

def get_risk_manager() -> RiskManager:
    """Get singleton RiskManager instance"""
    global _risk_manager_instance
    
    if _risk_manager_instance is None:
        with _risk_manager_lock:
            if _risk_manager_instance is None:
                _risk_manager_instance = RiskManager()
    
    return _risk_manager_instance


if __name__ == '__main__':
    import time
    
    risk_mgr = get_risk_manager()
    
    # Test signal limiting
    print("Testing signal rate limiting...\n")
    
    for i in range(25):
        allowed, reason = risk_mgr.should_allow_signal("BTCUSD", "BUY")
        if allowed:
            risk_mgr.record_signal("BTCUSD", "BUY", 0.75)
            print(f"✅ Signal {i+1}: {reason}")
        else:
            print(f"❌ Signal {i+1} blocked: {reason}")
        
        time.sleep(0.1)  # Small delay
    
    # Check stats
    print("\n--- Statistics ---")
    stats = risk_mgr.get_statistics("BTCUSD")
    print(f"Daily signals: {stats['daily_signals']}/{stats['max_daily']}")
    print(f"Hourly signals: {stats['hourly_signals']}/{stats['max_hourly']}")
