import numpy as np

class MarketAnalyzer:
    """
    Analyzes market structure, patterns, and anomalies.
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

        return {
            'trend': self.detect_trend(prices),
            'trend_strength': self.calculate_trend_strength(prices),
            'volume_analysis': self.analyze_volume_profile(volumes),
            'support_resistance': self.find_support_resistance(prices),
            'price_levels': self.get_key_levels(prices),
            'volatility': self.calculate_volatility(prices),
            'momentum': self.calculate_momentum(prices),
        }

    def detect_trend(self, prices):
        if len(prices) < 20:
            return "Insufficient data"
        recent_20 = prices[-20:]
        avg = np.mean(recent_20)
        current = prices[-1]
        strength = abs(current - avg) / avg

        if current > avg and strength > 0.01:
            return 'UPTREND'
        elif current < avg and strength > 0.01:
            return 'DOWNTREND'
        else:
            return 'SIDEWAYS'

    def calculate_trend_strength(self, prices):
        if len(prices) < 20:
            return 0.0
        recent_20 = prices[-20:]
        avg = np.mean(recent_20)
        current = prices[-1]
        return min(abs(current - avg) / avg, 1)

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
        if len(prices) < lookback:
            return {"error": "Insufficient price data"}
        recent = prices[-lookback:]
        high = np.max(recent)
        low = np.min(recent)

        return {
            'resistance': float(high),
            'support': float(low),
            'range': float(high - low),
            'current_vs_resistance': float(prices[-1] - high),
            'current_vs_support': float(prices[-1] - low)
        }

    def get_key_levels(self, prices):
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
        returns = np.diff(np.log(prices))
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        return float(volatility)

    def calculate_momentum(self, prices):
        if len(prices) < 20:
            return 0.0
        recent = prices[-20:]
        avg = np.mean(recent)
        current = prices[-1]
        momentum = (current - avg) / avg * 100
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