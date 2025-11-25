import numpy as np            # Numeric operations on arrays (mean, std), needed for analysis
from enum import Enum         # Enumerations for clean, readable categorical states

# Define HMM states for clarity and type-safety
class HMMState(Enum):
    BEARISH = 0
    NEUTRAL = 1
    BULLISH = 2

# Trend states from actual price behavior
class Trend(Enum):
    DOWNTREND = -1
    SIDEWAYS = 0
    UPTREND = 1

# Relative volume levels compared to recent average
class VolumeLevel(Enum):
    LOW = -1
    NORMAL = 0
    HIGH = 1

class ContextAwareHMM:
    """
    Compares HMM state with real market conditions (trend, volume, price action)
    to produce context-aware signals.
    """

    def analyze_with_context(self, prices, volumes, hmm_state):
        # Guard against insufficient data that would break downstream slices
        if len(prices) < 20 or len(volumes) < 20:
            return {
                'signal': 'WAIT',
                'confidence': 0.0,
                'reasoning': 'Insufficient data for analysis',
                'context': {}
            }

        # Derive market context from price and volume series
        actual_trend = self._detect_actual_trend(prices)
        volume_level = self._analyze_volume(volumes)
        price_action = self._detect_price_action(prices)

        # Build a context block for transparency and downstream use
        context = {
            'hmm_state': self._state_name(hmm_state),
            'actual_trend': actual_trend.name,
            'volume_level': volume_level.name,
            'price_action': price_action,
            'volume_ratio': self._get_volume_ratio(volumes),
            'trend_strength': self._calculate_trend_strength(prices)
        }

        # Decide signal using a rule matrix that cross-checks HMM vs reality
        signal = self._make_contextual_decision(hmm_state, actual_trend, volume_level, prices)
        signal['context'] = context
        return signal

    def _detect_actual_trend(self, prices):
        # Use last 20 closes to compute deviation from average
        recent = prices[-20:]
        average = np.mean(recent)
        current = prices[-1]
        trend_strength = abs(current - average) / average if average != 0 else 0

        # Threshold avoids classifying minor noise as trend
        if current > average and trend_strength > 0.01:
            return Trend.UPTREND
        elif current < average and trend_strength > 0.01:
            return Trend.DOWNTREND
        else:
            return Trend.SIDEWAYS

    def _analyze_volume(self, volumes):
        # Compare current volume to recent average to classify level
        # Using stricter thresholds for more accurate volume signals
        average = np.mean(volumes[-20:])
        current = volumes[-1]
        ratio = current / average if average > 0 else 1

        if ratio > 2.0:  # Stricter threshold: was 1.5
            return VolumeLevel.HIGH
        elif ratio < 0.5:  # Stricter threshold: was 0.7
            return VolumeLevel.LOW
        else:
            return VolumeLevel.NORMAL

    def _get_volume_ratio(self, volumes):
        # Provide ratio for context diagnostics (used by responses)
        average = np.mean(volumes[-20:])
        current = volumes[-1]
        return current / average if average > 0 else 1

    def _detect_price_action(self, prices):
        # Compare recent 5-candle move to the previous 5-candle move
        if len(prices) < 10:
            return "Insufficient data"
        recent = prices[-5:]
        previous = prices[-10:-5]

        recent_change = abs(recent[-1] - recent[0]) / recent[0] if recent[0] != 0 else 0
        previous_change = abs(previous[-1] - previous[0]) / previous[0] if previous[0] != 0 else 0

        # Classify acceleration/deceleration to enrich context
        if recent_change > previous_change * 1.5:
            return "Accelerating move"
        elif recent_change < previous_change * 0.5:
            return "Decelerating move"
        else:
            return "Continuous trend"

    def _calculate_trend_strength(self, prices):
        # Normalized distance from 20-period average (capped at 1 for stability)
        recent = prices[-20:]
        average = np.mean(recent)
        current = prices[-1]
        if average == 0:
            return 0.0
        return float(min(abs(current - average) / average, 1))

    def _make_contextual_decision(self, hmm_state, actual_trend, volume_level, prices):
        # Map int HMM state to enum for readable comparisons
        if hmm_state == 0:
            hmm_enum = HMMState.BEARISH
        elif hmm_state == 2:
            hmm_enum = HMMState.BULLISH
        else:
            hmm_enum = HMMState.NEUTRAL

        # ==================== BULLISH SIGNALS ====================
        if hmm_enum == HMMState.BULLISH and actual_trend == Trend.UPTREND:
            # Strong bullish confluence: HMM + Trend aligned
            if volume_level == VolumeLevel.HIGH:
                return {
                    'signal': 'BUY',
                    'confidence': 0.88,
                    'reasoning': 'Strong bullish confluence: HMM + Trend + High Volume',
                    'type': 'STRONG_BUY'
                }
            elif volume_level == VolumeLevel.NORMAL:
                # 2/3 confluence: HMM + Trend (volume normal, not confirming)
                return {
                    'signal': 'BUY',
                    'confidence': 0.75,
                    'reasoning': 'Bullish confluence: HMM + Uptrend (normal volume)',
                    'type': 'BUY'
                }
            else:  # LOW volume
                # Still signal but with caution
                return {
                    'signal': 'BUY',
                    'confidence': 0.65,
                    'reasoning': 'Bullish HMM + Uptrend but low volume - use tighter stops',
                    'type': 'BUY_WEAK'
                }

        # ==================== BEARISH SIGNALS ====================
        if hmm_enum == HMMState.BEARISH and actual_trend == Trend.DOWNTREND:
            # Strong bearish confluence: HMM + Trend aligned
            if volume_level == VolumeLevel.HIGH:
                return {
                    'signal': 'SELL',
                    'confidence': 0.86,
                    'reasoning': 'Strong bearish confluence: HMM + Trend + High Volume',
                    'type': 'STRONG_SELL'
                }
            elif volume_level == VolumeLevel.NORMAL:
                # 2/3 confluence: HMM + Trend
                return {
                    'signal': 'SELL',
                    'confidence': 0.74,
                    'reasoning': 'Bearish confluence: HMM + Downtrend (normal volume)',
                    'type': 'SELL'
                }
            else:  # LOW volume
                return {
                    'signal': 'SELL',
                    'confidence': 0.63,
                    'reasoning': 'Bearish HMM + Downtrend but low volume - use tighter stops',
                    'type': 'SELL_WEAK'
                }

        # ==================== CONSOLIDATION BREAKOUTS ====================
        if hmm_enum == HMMState.NEUTRAL and actual_trend == Trend.UPTREND:
            # Neutral HMM breaking upward with decent volume
            if volume_level in [VolumeLevel.NORMAL, VolumeLevel.HIGH]:
                confidence = 0.80 if volume_level == VolumeLevel.HIGH else 0.72
                return {
                    'signal': 'BUY',
                    'confidence': confidence,
                    'reasoning': f'Consolidation breakout: Neutral HMM + Uptrend + {volume_level.name} volume',
                    'type': 'CONSOLIDATION_BUY'
                }
            else:
                return {
                    'signal': 'WAIT',
                    'confidence': 0.4,
                    'reasoning': 'Uptrend present but low volume - waiting for confirmation',
                    'type': 'LOW_VOLUME_WAIT'
                }

        if hmm_enum == HMMState.NEUTRAL and actual_trend == Trend.DOWNTREND:
            # Neutral HMM breaking downward with decent volume
            if volume_level in [VolumeLevel.NORMAL, VolumeLevel.HIGH]:
                confidence = 0.78 if volume_level == VolumeLevel.HIGH else 0.70
                return {
                    'signal': 'SELL',
                    'confidence': confidence,
                    'reasoning': f'Consolidation breakout: Neutral HMM + Downtrend + {volume_level.name} volume',
                    'type': 'CONSOLIDATION_SELL'
                }
            else:
                return {
                    'signal': 'WAIT',
                    'confidence': 0.4,
                    'reasoning': 'Downtrend present but low volume - waiting for confirmation',
                    'type': 'LOW_VOLUME_WAIT'
                }

        # ==================== DIVERGENCE SIGNALS (use with caution) ====================
        if hmm_enum == HMMState.BULLISH and actual_trend == Trend.DOWNTREND:
            # Bullish HMM but market going down = potential reversal zone
            return {
                'signal': 'WAIT',
                'confidence': 0.5,
                'reasoning': 'Bullish HMM in downtrend - potential reversal zone, monitor for confirmation',
                'type': 'REVERSAL_WATCH'
            }

        if hmm_enum == HMMState.BEARISH and actual_trend == Trend.UPTREND:
            # Bearish HMM but market going up = potential pullback
            if volume_level == VolumeLevel.HIGH:
                return {
                    'signal': 'WAIT',
                    'confidence': 0.55,
                    'reasoning': 'Bearish divergence in strong uptrend - potential pullback, risky short',
                    'type': 'PULLBACK_RISK'
                }
            else:
                # Low volume bearish signal in uptrend = usually noise
                return {
                    'signal': 'WAIT',
                    'confidence': 0.3,
                    'reasoning': 'Bearish noise in uptrend - likely fakeout, stay bullish',
                    'type': 'FAKEOUT_RISK'
                }

        # ==================== DEFAULT: NEUTRAL/SIDEWAYS ====================
        if hmm_enum == HMMState.NEUTRAL and actual_trend == Trend.SIDEWAYS:
            return {
                'signal': 'WAIT',
                'confidence': 0.2,
                'reasoning': 'Market in consolidation/sideways mode - waiting for directional breakout',
                'type': 'CONSOLIDATION_NEUTRAL'
            }

        # ==================== NO CLEAR CONFLUENCE ====================
        return {
            'signal': 'WAIT',
            'confidence': 0.15,
            'reasoning': f'Insufficient confluence - HMM:{hmm_enum.name}, Trend:{actual_trend.name}, Volume:{volume_level.name}',
            'type': 'INSUFFICIENT_CONFLUENCE'
        }

    def _state_name(self, state):
        # Normalize name for context payload; avoid mismatches like 'CONSOLIDATION' vs 'NEUTRAL'
        if state == 0:
            return 'BEARISH'
        elif state == 2:
            return 'BULLISH'
        else:
            return 'NEUTRAL'