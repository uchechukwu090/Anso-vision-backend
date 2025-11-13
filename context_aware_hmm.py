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
        average = np.mean(volumes[-20:])
        current = volumes[-1]
        ratio = current / average if average > 0 else 1

        if ratio > 1.5:
            return VolumeLevel.HIGH
        elif ratio < 0.7:
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

        # Decision matrix: combine HMM, trend, and volume into actionable signals
        if (hmm_enum == HMMState.BULLISH and
            actual_trend == Trend.UPTREND and
            volume_level == VolumeLevel.HIGH):
            return {'signal': 'BUY', 'confidence': 0.88, 'reasoning': 'Strong bullish confluence', 'type': 'STRONG_BUY'}

        if (hmm_enum == HMMState.BEARISH and
            actual_trend == Trend.UPTREND and
            volume_level == VolumeLevel.HIGH):
            return {'signal': 'SELL', 'confidence': 0.78, 'reasoning': 'Pullback detected', 'type': 'PULLBACK_SHORT'}

        if (hmm_enum == HMMState.BEARISH and
            actual_trend == Trend.UPTREND and
            volume_level == VolumeLevel.LOW):
            return {'signal': 'WAIT', 'confidence': 0.3, 'reasoning': 'Fakeout risk', 'type': 'FAKEOUT_RISK'}

        if (hmm_enum == HMMState.BULLISH and actual_trend == Trend.DOWNTREND):
            return {'signal': 'WAIT', 'confidence': 0.4, 'reasoning': 'Bearish divergence', 'type': 'DIVERGENCE'}

        if (hmm_enum == HMMState.NEUTRAL and
            actual_trend == Trend.UPTREND and
            volume_level == VolumeLevel.HIGH):
            return {'signal': 'BUY', 'confidence': 0.82, 'reasoning': 'Consolidation breakout', 'type': 'BREAKOUT_BUY'}

        if (hmm_enum == HMMState.NEUTRAL and
            actual_trend == Trend.UPTREND and
            volume_level == VolumeLevel.LOW):
            return {'signal': 'WAIT', 'confidence': 0.3, 'reasoning': 'Institutional liquidation', 'type': 'LIQUIDATION_RISK'}

        if (hmm_enum == HMMState.BEARISH and
            actual_trend == Trend.DOWNTREND and
            volume_level == VolumeLevel.HIGH):
            return {'signal': 'SELL', 'confidence': 0.85, 'reasoning': 'Strong bearish confluence', 'type': 'STRONG_SELL'}

        # No strong confluence found
        return {'signal': 'WAIT', 'confidence': 0.2, 'reasoning': 'No clear signal', 'type': 'CONFLICTING'}

    def _state_name(self, state):
        # Normalize name for context payload; avoid mismatches like 'CONSOLIDATION' vs 'NEUTRAL'
        if state == 0:
            return 'BEARISH'
        elif state == 2:
            return 'BULLISH'
        else:
            return 'NEUTRAL'