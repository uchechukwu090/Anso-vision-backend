"""
LIQUIDITY ZONE DETECTOR - Mathematical Implementation
Detects stop-hunt zones using price clustering and swing point analysis
"""
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class LiquidityZoneDetector:
    """
    Detects liquidity zones where stops accumulate (stop-hunt magnets)
    Uses: Price clustering, swing highs/lows, volume analysis
    """
    
    def __init__(self, cluster_tolerance: float = 0.0005, min_touches: int = 3):
        self.cluster_tolerance = cluster_tolerance
        self.min_touches = min_touches
    
    def detect_liquidity_zones(self, prices: np.ndarray, volumes: np.ndarray) -> Dict:
        """
        Detect all liquidity zones in the market
        
        Returns:
            {
                'buy_side_liquidity': [levels below price],
                'sell_side_liquidity': [levels above price],
                'nearest_buy_zone': float,
                'nearest_sell_zone': float,
                'distance_to_buy_pct': float,
                'distance_to_sell_pct': float
            }
        """
        current_price = prices[-1]
        lookback = min(100, len(prices))
        recent_prices = prices[-lookback:]
        
        # Find swing points (where stops cluster)
        swing_highs, swing_lows = self._find_swing_points(recent_prices)
        
        # Find price clusters (fractal levels)
        clusters = self._detect_price_clusters(recent_prices)
        
        # Separate into buy/sell side
        buy_side = []
        sell_side = []
        
        for level in swing_lows:
            if level < current_price:
                buy_side.append(float(level))
        
        for level in swing_highs:
            if level > current_price:
                sell_side.append(float(level))
        
        for cluster_price, _ in clusters:
            if cluster_price < current_price:
                buy_side.append(float(cluster_price))
            else:
                sell_side.append(float(cluster_price))
        
        # Get nearest zones
        nearest_buy = max(buy_side) if buy_side else current_price * 0.99
        nearest_sell = min(sell_side) if sell_side else current_price * 1.01
        
        return {
            'buy_side_liquidity': sorted(set(buy_side), reverse=True),
            'sell_side_liquidity': sorted(set(sell_side)),
            'nearest_buy_zone': float(nearest_buy),
            'nearest_sell_zone': float(nearest_sell),
            'distance_to_buy_pct': abs(current_price - nearest_buy) / current_price * 100,
            'distance_to_sell_pct': abs(nearest_sell - current_price) / current_price * 100
        }
    
    def _find_swing_points(self, prices: np.ndarray, window: int = 5) -> Tuple[List, List]:
        """Find swing highs and lows"""
        swing_highs = []
        swing_lows = []
        
        for i in range(window, len(prices) - window):
            if prices[i] == np.max(prices[i-window:i+window+1]):
                swing_highs.append(prices[i])
            if prices[i] == np.min(prices[i-window:i+window+1]):
                swing_lows.append(prices[i])
        
        return swing_highs, swing_lows
    
    def _detect_price_clusters(self, prices: np.ndarray) -> List[Tuple[float, int]]:
        """Find price levels where price repeatedly bounces"""
        price_bins = defaultdict(int)
        tolerance = self.cluster_tolerance
        
        for price in prices:
            bin_key = round(price / (price * tolerance)) * (price * tolerance)
            price_bins[bin_key] += 1
        
        clusters = [
            (price, count) 
            for price, count in price_bins.items() 
            if count >= self.min_touches
        ]
        
        return sorted(clusters, key=lambda x: x[1], reverse=True)
    
    def calculate_stop_hunt_risk(self, current_price: float, liquidity_map: Dict, signal_type: str) -> Dict:
        """Calculate risk of being stop-hunted"""
        if signal_type == 'BUY':
            distance = liquidity_map['distance_to_buy_pct']
            target = liquidity_map['nearest_buy_zone']
        else:
            distance = liquidity_map['distance_to_sell_pct']
            target = liquidity_map['nearest_sell_zone']
        
        if distance < 0.3:
            risk = 'EXTREME'
            risk_score = 0.95
        elif distance < 0.5:
            risk = 'HIGH'
            risk_score = 0.80
        elif distance < 1.0:
            risk = 'MEDIUM'
            risk_score = 0.50
        else:
            risk = 'LOW'
            risk_score = 0.20
        
        return {
            'risk_level': risk,
            'risk_score': risk_score,
            'hunt_target': float(target),
            'distance_pct': float(distance)
        }
    
    def adjust_stop_loss(self, sl: float, liquidity_map: Dict, signal_type: str, buffer: float) -> float:
        """Adjust SL to avoid liquidity zones"""
        if signal_type == 'BUY':
            nearest_liq = liquidity_map['nearest_buy_zone']
            if sl > nearest_liq:  # SL is in danger zone
                adjusted_sl = nearest_liq - buffer
                return float(adjusted_sl)
        else:
            nearest_liq = liquidity_map['nearest_sell_zone']
            if sl < nearest_liq:  # SL is in danger zone
                adjusted_sl = nearest_liq + buffer
                return float(adjusted_sl)
        
        return float(sl)
