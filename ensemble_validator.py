"""
Ensemble Signal Validator - Combines Multiple Models for Higher Accuracy
Uses voting mechanism to reduce false signals
"""
import numpy as np
from typing import Dict, List, Tuple
from enum import Enum


class SignalStrength(Enum):
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5


class EnsembleValidator:
    """
    Validates signals using multiple confirmation methods:
    1. HMM + Context (existing system)
    2. Trend confirmation (multiple timeframes)
    3. Volume profile validation
    4. Momentum divergence check
    5. Risk/reward validation
    """
    
    def __init__(self, min_confirmation_score: float = 0.65):
        """
        Args:
            min_confirmation_score: Minimum score (0-1) required to approve signal
        """
        self.min_confirmation_score = min_confirmation_score
    
    def validate_signal(self, 
                       signal: dict,
                       prices: np.ndarray,
                       volumes: np.ndarray) -> Dict:
        """
        Validate a signal using ensemble methods
        
        Returns:
            dict with:
            - approved: bool (whether signal passes validation)
            - confidence: float (0-1)
            - strength: SignalStrength enum
            - warnings: list of warnings
            - confirmations: list of confirmations
        """
        if not signal or signal.get('entry') is None:
            return {
                'approved': False,
                'confidence': 0.0,
                'strength': SignalStrength.VERY_WEAK,
                'warnings': ['No signal provided'],
                'confirmations': []
            }
        
        signal_type = signal.get('signal_type', 'WAIT')
        
        if signal_type not in ['BUY', 'SELL']:
            return {
                'approved': False,
                'confidence': 0.0,
                'strength': SignalStrength.VERY_WEAK,
                'warnings': ['Signal type is WAIT'],
                'confirmations': []
            }
        
        # Run all validation checks
        checks = []
        warnings = []
        confirmations = []
        
        # 1. Trend Confirmation (30% weight)
        trend_score, trend_notes = self._check_trend_alignment(signal_type, prices)
        checks.append(('Trend', trend_score, 0.30))
        if trend_score > 0.7:
            confirmations.append(f"Trend alignment: {trend_notes}")
        elif trend_score < 0.3:
            warnings.append(f"Trend divergence: {trend_notes}")
        
        # 2. Volume Confirmation (20% weight)
        volume_score, volume_notes = self._check_volume_confirmation(signal_type, volumes)
        checks.append(('Volume', volume_score, 0.20))
        if volume_score > 0.7:
            confirmations.append(f"Volume confirms: {volume_notes}")
        elif volume_score < 0.3:
            warnings.append(f"Volume warning: {volume_notes}")
        
        # 3. Momentum Check (15% weight)
        momentum_score, momentum_notes = self._check_momentum(signal_type, prices)
        checks.append(('Momentum', momentum_score, 0.15))
        if momentum_score > 0.7:
            confirmations.append(f"Momentum supports: {momentum_notes}")
        elif momentum_score < 0.3:
            warnings.append(f"Momentum concern: {momentum_notes}")
        
        # 4. Risk/Reward Validation (20% weight)
        rr_score, rr_notes = self._check_risk_reward(signal)
        checks.append(('Risk/Reward', rr_score, 0.20))
        if rr_score > 0.7:
            confirmations.append(f"Good R:R - {rr_notes}")
        elif rr_score < 0.3:
            warnings.append(f"Poor R:R - {rr_notes}")
        
        # 5. Price Action Confirmation (15% weight)
        price_action_score, pa_notes = self._check_price_action(signal_type, prices)
        checks.append(('Price Action', price_action_score, 0.15))
        if price_action_score > 0.7:
            confirmations.append(f"Price action confirms: {pa_notes}")
        elif price_action_score < 0.3:
            warnings.append(f"Price action concern: {pa_notes}")
        
        # Calculate weighted confidence score
        total_confidence = sum(score * weight for _, score, weight in checks)
        
        # Determine signal strength
        if total_confidence >= 0.85:
            strength = SignalStrength.VERY_STRONG
        elif total_confidence >= 0.75:
            strength = SignalStrength.STRONG
        elif total_confidence >= 0.65:
            strength = SignalStrength.MODERATE
        elif total_confidence >= 0.50:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.VERY_WEAK
        
        # Approve only if confidence exceeds threshold
        approved = total_confidence >= self.min_confirmation_score
        
        return {
            'approved': approved,
            'confidence': float(total_confidence),
            'strength': strength.name,
            'warnings': warnings,
            'confirmations': confirmations,
            'checks': {name: score for name, score, _ in checks}
        }
    
    def _check_trend_alignment(self, signal_type: str, prices: np.ndarray) -> Tuple[float, str]:
        """Check if signal aligns with multiple timeframe trends"""
        if len(prices) < 50:
            return 0.5, "Insufficient data"
        
        # Check different timeframes
        short_trend = self._calculate_trend_direction(prices[-20:])   # Last 20 candles
        medium_trend = self._calculate_trend_direction(prices[-50:])  # Last 50 candles
        
        # Count aligned trends
        expected_direction = 1 if signal_type == 'BUY' else -1
        
        alignment_count = 0
        if short_trend == expected_direction:
            alignment_count += 1
        if medium_trend == expected_direction:
            alignment_count += 1
        
        score = alignment_count / 2.0
        
        trend_str = "bullish" if expected_direction == 1 else "bearish"
        notes = f"{alignment_count}/2 timeframes confirm {trend_str}"
        
        return score, notes
    
    def _calculate_trend_direction(self, prices: np.ndarray) -> int:
        """Calculate trend direction: 1=up, -1=down, 0=sideways"""
        if len(prices) < 2:
            return 0
        
        start_price = np.mean(prices[:5])
        end_price = np.mean(prices[-5:])
        
        change_pct = (end_price - start_price) / start_price
        
        if change_pct > 0.01:  # 1% threshold
            return 1
        elif change_pct < -0.01:
            return -1
        else:
            return 0
    
    def _check_volume_confirmation(self, signal_type: str, volumes: np.ndarray) -> Tuple[float, str]:
        """Check if volume supports the signal"""
        if len(volumes) < 20:
            return 0.5, "Insufficient data"
        
        recent_vol = volumes[-5:]
        avg_vol = np.mean(volumes[-20:])
        
        current_vol_ratio = np.mean(recent_vol) / avg_vol if avg_vol > 0 else 1
        
        # High volume is good for breakouts
        if current_vol_ratio > 1.5:
            score = 0.9
            notes = f"High volume ({current_vol_ratio:.1f}x average)"
        elif current_vol_ratio > 1.2:
            score = 0.75
            notes = f"Above average volume ({current_vol_ratio:.1f}x)"
        elif current_vol_ratio > 0.8:
            score = 0.6
            notes = f"Normal volume ({current_vol_ratio:.1f}x)"
        else:
            score = 0.3
            notes = f"Low volume ({current_vol_ratio:.1f}x) - weak confirmation"
        
        return score, notes
    
    def _check_momentum(self, signal_type: str, prices: np.ndarray) -> Tuple[float, str]:
        """Check momentum indicators"""
        if len(prices) < 30:
            return 0.5, "Insufficient data"
        
        # Simple momentum: rate of change
        roc_10 = (prices[-1] - prices[-10]) / prices[-10]
        roc_20 = (prices[-1] - prices[-20]) / prices[-20]
        
        expected_positive = (signal_type == 'BUY')
        
        # Check if momentum aligns with signal
        momentum_aligned = 0
        if (roc_10 > 0) == expected_positive:
            momentum_aligned += 1
        if (roc_20 > 0) == expected_positive:
            momentum_aligned += 1
        
        score = momentum_aligned / 2.0
        notes = f"{momentum_aligned}/2 momentum indicators align"
        
        return score, notes
    
    def _check_risk_reward(self, signal: dict) -> Tuple[float, str]:
        """Validate risk/reward ratio"""
        entry = signal.get('entry')
        tp = signal.get('tp')
        sl = signal.get('sl')
        signal_type = signal.get('signal_type')
        
        if not all([entry, tp, sl]):
            return 0.0, "Missing TP/SL levels"
        
        # Calculate R:R
        if signal_type == 'BUY':
            reward = tp - entry
            risk = entry - sl
        else:
            reward = entry - tp
            risk = sl - entry
        
        if risk <= 0:
            return 0.0, "Invalid risk (negative or zero)"
        
        rr_ratio = reward / risk
        
        # Score based on R:R
        if rr_ratio >= 3.0:
            score = 1.0
            notes = f"Excellent R:R {rr_ratio:.1f}:1"
        elif rr_ratio >= 2.0:
            score = 0.85
            notes = f"Good R:R {rr_ratio:.1f}:1"
        elif rr_ratio >= 1.5:
            score = 0.65
            notes = f"Acceptable R:R {rr_ratio:.1f}:1"
        elif rr_ratio >= 1.0:
            score = 0.4
            notes = f"Marginal R:R {rr_ratio:.1f}:1"
        else:
            score = 0.1
            notes = f"Poor R:R {rr_ratio:.1f}:1"
        
        return score, notes
    
    def _check_price_action(self, signal_type: str, prices: np.ndarray) -> Tuple[float, str]:
        """Check recent price action patterns"""
        if len(prices) < 10:
            return 0.5, "Insufficient data"
        
        recent = prices[-5:]
        
        # Check for consolidation breakout
        volatility = np.std(recent) / np.mean(recent)
        
        # Lower volatility suggests consolidation
        if volatility < 0.005:
            score = 0.8
            notes = "Consolidation breakout pattern"
        elif volatility < 0.01:
            score = 0.6
            notes = "Normal volatility"
        else:
            score = 0.4
            notes = "High volatility - choppy price action"
        
        return score, notes


if __name__ == '__main__':
    # Test the ensemble validator
    validator = EnsembleValidator(min_confirmation_score=0.65)
    
    # Generate test data
    np.random.seed(42)
    test_prices = np.cumsum(np.random.normal(0.002, 0.01, 100)) + 100  # Uptrend
    test_volumes = np.random.normal(1000, 200, 100)
    
    # Test signal
    test_signal = {
        'signal_type': 'BUY',
        'entry': 105.0,
        'tp': 108.0,
        'sl': 104.0
    }
    
    result = validator.validate_signal(test_signal, test_prices, test_volumes)
    
    print("Ensemble Validation Result:")
    print(f"Approved: {result['approved']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Strength: {result['strength']}")
    print(f"\nConfirmations:")
    for conf in result['confirmations']:
        print(f"  ✅ {conf}")
    print(f"\nWarnings:")
    for warn in result['warnings']:
        print(f"  ⚠️ {warn}")
