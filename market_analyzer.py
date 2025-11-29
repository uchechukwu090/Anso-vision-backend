import numpy as np

class MarketAnalyzer:
    """
    Analyzes market structure, patterns, and anomalies.
    FIXED: Returns proper key levels for signal_generator
    """

    def analyze_market_structure(self, prices, volumes):
        if len(prices) < 50 or len(volumes) < 20:
            return {
                'error': 'Insufficient data',
                'required_prices': 50,
                'required_volumes': 20,
                'provided_prices': len(prices),
                'provided_volumes': len(volumes)
            }

        # Get support/resistance
        sr_levels = self.find_support_resistance(prices)
        
        return {
            'trend': self.detect_trend(prices),
            'trend_strength': self.calculate_trend_strength(prices),
            'volume_analysis': self.analyze_volume_profile(volumes),
            'support_resistance': sr_levels,
            'price_levels': {
                # ✅ FIX: Return the keys that signal_generator expects
                'nearest_support': sr_levels.get('support', 0.0),
                'nearest_resistance': sr_levels.get('resistance', 0.0),
                'high_50': float(np.max(prices[-50:])),
                'low_50': float(np.min(prices[-50:])),
                'avg_50': float(np.mean(prices[-50:])),
                'current': float(prices[-1]),
                'prev_close': float(prices[-2])
            },
            'volatility': self.calculate_volatility(prices),
            'momentum': self.calculate_momentum(prices),
        }

    def detect_trend(self, prices):
        """Enhanced trend detection using multiple timeframes"""
        if len(prices) < 20:
            return "Insufficient data"
        
        # Use multiple lookback periods
        recent_10 = prices[-10:]
        recent_20 = prices[-20:]
        
        # Short-term trend (10 candles)
        short_slope = (recent_10[-1] - recent_10[0]) / len(recent_10)
        
        # Medium-term trend (20 candles)
        med_slope = (recent_20[-1] - recent_20[0]) / len(recent_20)
        
        # Calculate percentage moves
        short_pct = (recent_10[-1] - recent_10[0]) / recent_10[0] * 100
        med_pct = (recent_20[-1] - recent_20[0]) / recent_20[0] * 100
        
        # Lower threshold for trend detection (0.5% instead of 1%)
        if short_slope > 0 and med_slope > 0 and med_pct > 0.5:
            return 'UPTREND'
        elif short_slope < 0 and med_slope < 0 and med_pct < -0.5:
            return 'DOWNTREND'
        else:
            return 'SIDEWAYS'

    def calculate_trend_strength(self, prices):
        """Enhanced trend strength calculation"""
        if len(prices) < 20:
            return 0.0
        
        # Use linear regression to measure trend strength
        x = np.arange(len(prices[-20:]))
        y = prices[-20:]
        
        # Calculate correlation coefficient (R-squared)
        correlation = np.corrcoef(x, y)[0, 1]
        
        # Also measure price deviation
        avg = np.mean(y)
        current = prices[-1]
        deviation = abs(current - avg) / avg if avg > 0 else 0
        
        # Combine both measures
        strength = (abs(correlation) + deviation) / 2
        
        return float(min(strength, 1.0))

    def analyze_volume_profile(self, volumes):
        if len(volumes) < 20:
            return {"error": "Insufficient volume data"}
        recent_vol = volumes[-20:]
        avg_vol = np.mean(recent_vol)
        current_vol = volumes[-1]
        ratio = current_vol / avg_vol if avg_vol > 0 else 1

        return {
            'current_volume': int(current_vol),
            'average_volume': int(avg_vol),
            'volume_ratio': ratio,
            'volume_level': 'HIGH' if ratio > 1.5 else 'LOW' if ratio < 0.7 else 'NORMAL',
            'trend': 'Increasing' if current_vol > avg_vol else 'Decreasing'
        }

    def find_support_resistance(self, prices, lookback=50):
        """Find support and resistance with proper pivot detection"""
        if len(prices) < lookback:
            lookback = len(prices)
        
        recent = prices[-lookback:]
        
        # Find local highs and lows (pivot points)
        pivots_high = []
        pivots_low = []
        
        for i in range(2, len(recent) - 2):
            # Local high
            if recent[i] > recent[i-1] and recent[i] > recent[i-2] and \
               recent[i] > recent[i+1] and recent[i] > recent[i+2]:
                pivots_high.append(recent[i])
            
            # Local low
            if recent[i] < recent[i-1] and recent[i] < recent[i-2] and \
               recent[i] < recent[i+1] and recent[i] < recent[i+2]:
                pivots_low.append(recent[i])
        
        # Use pivots if found, otherwise use simple high/low
        if len(pivots_high) > 0:
            resistance = float(np.max(pivots_high))
        else:
            resistance = float(np.max(recent))
        
        if len(pivots_low) > 0:
            support = float(np.min(pivots_low))
        else:
            support = float(np.min(recent))
        
        current_price = float(prices[-1])
        
        return {
            'resistance': resistance,
            'support': support,
            'range': float(resistance - support),
            'current_vs_resistance': float(current_price - resistance),
            'current_vs_support': float(current_price - support),
            'pivot_highs_found': len(pivots_high),
            'pivot_lows_found': len(pivots_low)
        }

    def get_key_levels(self, prices):
        """Get key price levels - DEPRECATED, use price_levels from analyze_market_structure"""
        if len(prices) < 50:
            return {"error": "Insufficient price data"}
        recent_50 = prices[-50:]
        return {
            'high_50': float(np.max(recent_50)),
            'low_50': float(np.min(recent_50)),
            'avg_50': float(np.mean(recent_50)),
            'current': float(prices[-1]),
            'prev_close': float(prices[-2])
        }

    def calculate_volatility(self, prices):
        if len(prices) < 2:
            return 0.0
        returns = np.diff(np.log(prices + 1e-10))  # Add small value to avoid log(0)
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        return float(volatility)

    def calculate_momentum(self, prices):
        """Enhanced momentum with RSI-style calculation"""
        if len(prices) < 20:
            return 0.0
        
        # Calculate price changes
        changes = np.diff(prices[-20:])
        
        # Separate gains and losses
        gains = changes[changes > 0].sum() if len(changes[changes > 0]) > 0 else 0
        losses = abs(changes[changes < 0].sum()) if len(changes[changes < 0]) > 0 else 0
        
        # RSI-style momentum
        if losses == 0:
            momentum = 100.0
        else:
            rs = gains / losses
            momentum = 100 - (100 / (1 + rs))
        
        # Normalize to -100 to +100 range
        momentum = (momentum - 50) * 2
        
        return float(momentum)

    def detect_pullback(self, prices, volumes):
        if len(volumes) < 20:
            return {"error": "Insufficient volume data"}
        recent_vol = volumes[-1]
        avg_vol = np.mean(volumes[-20:])
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1

        if vol_ratio > 1.3:
            return {'is_pullback': True,'volume_confirmed': True,'confidence': 0.85}
        else:
            return {'is_pullback': True,'volume_confirmed': False,'confidence': 0.4}

    def detect_fakeout(self, prices, volumes):
        if len(volumes) < 20:
            return {"error": "Insufficient volume data"}
        recent_vol = volumes[-1]
        avg_vol = np.mean(volumes[-20:])
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1

        if vol_ratio < 0.8:
            return {'is_fakeout_risk': True,'confidence': 0.7,'reason': 'Low volume breakout'}
        else:
            return {'is_fakeout_risk': False,'confidence': 0.2,'reason': 'Volume confirmed'}
