"""
ENHANCED DISCOUNT ENTRY LOGIC - Handles Fresh Trends + Momentum Pullbacks
"""

def _detect_trend_stage_and_entry(self, prices: np.ndarray, signal_type: str, 
                                   support: float, resistance: float,
                                   trend_strength: float) -> Dict:
    """
    ✅ NEW: Intelligent entry system with TWO modes:
    
    MODE 1: FRESH TREND (low strength) → Wait for discounted entry (25-30% zone)
    MODE 2: ESTABLISHED TREND (high strength) → Enter on pullbacks (35-50% zone)
    
    Args:
        prices: Price array
        signal_type: 'BUY' or 'SELL'
        support/resistance: Key levels
        trend_strength: 0.0-1.0 (from market_analyzer)
    
    Returns:
        Dict with entry decision + reasoning
    """
    current = prices[-1]
    price_range = resistance - support
    
    if price_range <= 0:
        return {
            'is_valid_entry': False,
            'entry_mode': 'NONE',
            'reason': 'Invalid price range',
            'should_wait': True
        }
    
    # Calculate volatility-adjusted base threshold
    threshold = self._calculate_volatility_adjusted_threshold(prices)
    
    # Detect recent price momentum (last 10 vs 20 candles)
    recent_10_change = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
    recent_20_change = (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 else 0
    
    # Classify trend stage
    # FRESH: Low strength (<0.4) OR just starting to move
    # ESTABLISHED: Strong (>0.6) AND has momentum
    is_fresh_trend = trend_strength < 0.4 or abs(recent_20_change) < 0.015  # <1.5% move
    is_established_trend = trend_strength > 0.6 and abs(recent_20_change) > 0.02  # >2% move
    
    # --- MODE 1: FRESH TREND - Wait for discounted entry ---
    if is_fresh_trend:
        if signal_type == 'BUY':
            distance_from_support = (current - support) / price_range
            is_discounted = distance_from_support < threshold
            
            if is_discounted:
                return {
                    'is_valid_entry': True,
                    'entry_mode': 'FRESH_TREND_DISCOUNT',
                    'discount_pct': float((1 - distance_from_support) * 100),
                    'threshold_used': threshold,
                    'reason': f'Fresh uptrend - entered at discount zone ({distance_from_support:.1%} from support)',
                    'should_wait': False,
                    'entry_quality': 'EXCELLENT'
                }
            else:
                return {
                    'is_valid_entry': False,
                    'entry_mode': 'FRESH_TREND_WAITING',
                    'distance_from_discount': float((support + price_range * threshold) - current),
                    'distance_pct': float(distance_from_support * 100),
                    'reason': f'Fresh trend but price not discounted yet. Currently at {distance_from_support:.1%} of range, need <{threshold:.1%}',
                    'should_wait': True
                }
        
        else:  # SELL
            distance_from_resistance = (resistance - current) / price_range
            is_premium = distance_from_resistance < threshold
            
            if is_premium:
                return {
                    'is_valid_entry': True,
                    'entry_mode': 'FRESH_TREND_PREMIUM',
                    'premium_pct': float((1 - distance_from_resistance) * 100),
                    'threshold_used': threshold,
                    'reason': f'Fresh downtrend - entered at premium zone ({distance_from_resistance:.1%} from resistance)',
                    'should_wait': False,
                    'entry_quality': 'EXCELLENT'
                }
            else:
                return {
                    'is_valid_entry': False,
                    'entry_mode': 'FRESH_TREND_WAITING',
                    'distance_from_discount': float((resistance - price_range * threshold) - current),
                    'distance_pct': float(distance_from_resistance * 100),
                    'reason': f'Fresh downtrend but price not at premium yet. Currently {distance_from_resistance:.1%} from resistance, need <{threshold:.1%}',
                    'should_wait': True
                }
    
    # --- MODE 2: ESTABLISHED TREND - Enter on pullbacks ---
    elif is_established_trend:
        # Wider zone for pullback entries (35-50%)
        pullback_threshold = threshold + 0.15  # Add 15% to base threshold
        
        if signal_type == 'BUY':
            distance_from_support = (current - support) / price_range
            
            # Check if we're in a pullback zone (not too high, not too low)
            is_pullback_zone = 0.20 < distance_from_support < pullback_threshold
            
            # Confirm pullback: recent price pulled back from highs
            recent_high = np.max(prices[-10:])
            pullback_depth = (recent_high - current) / recent_high if recent_high > 0 else 0
            has_pullback = pullback_depth > 0.01  # At least 1% pullback
            
            if is_pullback_zone and has_pullback:
                return {
                    'is_valid_entry': True,
                    'entry_mode': 'MOMENTUM_PULLBACK',
                    'pullback_depth_pct': float(pullback_depth * 100),
                    'position_in_range': float(distance_from_support * 100),
                    'threshold_used': pullback_threshold,
                    'reason': f'Established uptrend pullback ({pullback_depth*100:.1f}% from recent high). Entering at {distance_from_support:.1%} of range',
                    'should_wait': False,
                    'entry_quality': 'GOOD'
                }
            elif distance_from_support > pullback_threshold:
                return {
                    'is_valid_entry': False,
                    'entry_mode': 'MOMENTUM_OVEREXTENDED',
                    'reason': f'Trend moving but price too high ({distance_from_support:.1%} of range). Waiting for pullback to <{pullback_threshold:.1%}',
                    'should_wait': True
                }
            else:
                return {
                    'is_valid_entry': False,
                    'entry_mode': 'MOMENTUM_NO_PULLBACK',
                    'reason': f'Trend moving but no pullback yet (only {pullback_depth*100:.1f}% from high). Wait for deeper retracement',
                    'should_wait': True
                }
        
        else:  # SELL in established downtrend
            distance_from_resistance = (resistance - current) / price_range
            
            is_pullback_zone = 0.20 < distance_from_resistance < pullback_threshold
            
            # Confirm rally/pullback in downtrend
            recent_low = np.min(prices[-10:])
            rally_depth = (current - recent_low) / recent_low if recent_low > 0 else 0
            has_rally = rally_depth > 0.01
            
            if is_pullback_zone and has_rally:
                return {
                    'is_valid_entry': True,
                    'entry_mode': 'MOMENTUM_PULLBACK',
                    'rally_depth_pct': float(rally_depth * 100),
                    'position_in_range': float(distance_from_resistance * 100),
                    'threshold_used': pullback_threshold,
                    'reason': f'Established downtrend rally ({rally_depth*100:.1f}% from recent low). Shorting at {distance_from_resistance:.1%} from resistance',
                    'should_wait': False,
                    'entry_quality': 'GOOD'
                }
            elif distance_from_resistance > pullback_threshold:
                return {
                    'is_valid_entry': False,
                    'entry_mode': 'MOMENTUM_OVEREXTENDED',
                    'reason': f'Downtrend moving but price too low ({distance_from_resistance:.1%} from resistance). Waiting for rally to <{pullback_threshold:.1%}',
                    'should_wait': True
                }
            else:
                return {
                    'is_valid_entry': False,
                    'entry_mode': 'MOMENTUM_NO_PULLBACK',
                    'reason': f'Downtrend moving but no rally yet (only {rally_depth*100:.1f}% from low). Wait for bounce',
                    'should_wait': True
                }
    
    # --- MODE 3: MODERATE TREND - Use standard discount logic ---
    else:
        # Trend strength 0.4-0.6: moderate, use standard zones
        if signal_type == 'BUY':
            distance_from_support = (current - support) / price_range
            is_discounted = distance_from_support < (threshold + 0.05)  # Slightly wider
            
            if is_discounted:
                return {
                    'is_valid_entry': True,
                    'entry_mode': 'MODERATE_TREND',
                    'position_in_range': float(distance_from_support * 100),
                    'threshold_used': threshold + 0.05,
                    'reason': f'Moderate trend - acceptable entry at {distance_from_support:.1%} of range',
                    'should_wait': False,
                    'entry_quality': 'FAIR'
                }
            else:
                return {
                    'is_valid_entry': False,
                    'entry_mode': 'MODERATE_TREND_WAITING',
                    'reason': f'Moderate trend but price at {distance_from_support:.1%} of range. Need <{threshold+0.05:.1%}',
                    'should_wait': True
                }
        
        else:  # SELL
            distance_from_resistance = (resistance - current) / price_range
            is_premium = distance_from_resistance < (threshold + 0.05)
            
            if is_premium:
                return {
                    'is_valid_entry': True,
                    'entry_mode': 'MODERATE_TREND',
                    'position_in_range': float(distance_from_resistance * 100),
                    'threshold_used': threshold + 0.05,
                    'reason': f'Moderate downtrend - acceptable entry at {distance_from_resistance:.1%} from resistance',
                    'should_wait': False,
                    'entry_quality': 'FAIR'
                }
            else:
                return {
                    'is_valid_entry': False,
                    'entry_mode': 'MODERATE_TREND_WAITING',
                    'reason': f'Moderate downtrend but price at {distance_from_resistance:.1%} from resistance. Need <{threshold+0.05:.1%}',
                    'should_wait': True
                }
