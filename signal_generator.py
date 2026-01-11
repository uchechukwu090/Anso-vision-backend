"""
ULTRA-CLEAN SIGNAL GENERATOR
✅ Only essentials: HMM + Liquidity + Adaptive TP/SL
✅ Removed: Monte Carlo, ATR, MarketAnalyzer, learning state, wavelet
✅ Single TP/SL method: HMM-Adaptive (mathematically optimal)
"""
import numpy as np
from kalman_filter import apply_kalman_filter
from hmm_model import MarketHMM
from context_aware_hmm import ContextAwareHMM
from liquidity_detector import LiquidityZoneDetector
from hmm_adaptive_tpsl import HMMAdaptiveTPSL
from typing import Dict

# Constants
HMM_COMPONENTS = 3
MIN_CANDLES = 200


class SignalGenerator:
    """
    Clean signal generator with only HMM core + Liquidity awareness
    """
    
    def __init__(self, n_hmm_components=HMM_COMPONENTS, random_state=42):
        # Core models only
        self.hmm_model = MarketHMM(
            n_components=n_hmm_components,
            covariance_type='diag',
            random_state=random_state
        )
        self.context_analyzer = ContextAwareHMM()
        self.liquidity_detector = LiquidityZoneDetector()
        self.tpsl_calculator = HMMAdaptiveTPSL()
        
        print(f"✅ SignalGenerator initialized (ULTRA-CLEAN)")
        print(f"   • HMM components: {n_hmm_components}")
        print(f"   • Liquidity detection: ENABLED")
        print(f"   • TP/SL: HMM-Adaptive (state-aware)")
        print(f"   • Removed: Monte Carlo, ATR, MarketAnalyzer, Learning")
    
    def _prepare_hmm_features(self, prices: np.ndarray) -> np.ndarray:
        """Prepare features for HMM"""
        if len(prices) < 20:
            raise ValueError(f"Need at least 20 candles, got {len(prices)}")
        
        # Use Kalman-filtered prices
        kalman_prices = apply_kalman_filter(prices)
        log_returns = np.diff(np.log(kalman_prices + 1e-10))
        
        # Calculate rolling volatility and momentum
        window = 10
        volatility = np.zeros_like(log_returns)
        momentum = np.zeros_like(log_returns)
        
        for i in range(window, len(log_returns)):
            volatility[i] = np.std(log_returns[i-window:i])
            momentum[i] = log_returns[i-window:i].sum()
        
        # Normalize volatility
        vol_mean = np.mean(volatility[window:])
        vol_std = np.std(volatility[window:])
        volatility_norm = (volatility - vol_mean) / vol_std if vol_std > 0 else volatility
        
        # Stack features
        features = np.column_stack([
            log_returns[window:],
            volatility_norm[window:],
            momentum[window:]
        ])
        
        return features
    
    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """Calculate current volatility"""
        if len(prices) < 14:
            return 0.0
        changes = np.diff(prices[-14:])
        return float(np.std(changes) * np.sqrt(14))
    
    def _detect_whipsaw(self, prices: np.ndarray, volumes: np.ndarray) -> Dict:
        """
        Simple whipsaw detection
        Checks: false breakouts, volume decline, momentum weakness
        """
        if len(prices) < 20:
            return {'risk': 'LOW', 'score': 0.0}
        
        recent_prices = prices[-20:]
        recent_vols = volumes[-20:]
        
        # Check if near extreme with declining volume
        high = np.max(recent_prices)
        low = np.min(recent_prices)
        current = recent_prices[-1]
        price_range = high - low
        
        near_extreme = (
            (high - current) < price_range * 0.15 or
            (current - low) < price_range * 0.15
        )
        
        vol_trend = np.polyfit(np.arange(len(recent_vols)), recent_vols, 1)[0]
        vol_declining = vol_trend < 0
        
        # Check momentum weakness
        moves = np.diff(recent_prices)
        momentum = moves[-5:].sum()
        volatility = np.std(moves[-5:])
        momentum_weak = volatility > abs(momentum) * 0.3
        
        # Calculate risk
        risk_factors = sum([near_extreme and vol_declining, momentum_weak])
        
        if risk_factors >= 2:
            return {'risk': 'HIGH', 'score': 0.8}
        elif risk_factors == 1:
            return {'risk': 'MEDIUM', 'score': 0.5}
        else:
            return {'risk': 'LOW', 'score': 0.2}
    
    def generate_signals(self, prices: np.ndarray, volumes: np.ndarray = None) -> Dict:
        """
        Generate trading signal
        
        Process:
        1. Kalman filter prices
        2. HMM detects market state
        3. Context analyzer validates with trend/volume
        4. Liquidity detector finds stop-hunt zones
        5. Whipsaw check
        6. HMM-adaptive TP/SL calculation
        7. Liquidity-adjusted SL
        """
        try:
            # Validate input
            if len(prices) < MIN_CANDLES:
                return self._wait(f"Need {MIN_CANDLES} candles, got {len(prices)}")
            
            if volumes is None:
                volumes = np.ones_like(prices)
            
            current_price = prices[-1]
            
            print("\n" + "="*70)
            print("🧠 ULTRA-CLEAN SIGNAL GENERATION")
            print("="*70)
            
            # 1. KALMAN FILTERING
            print("\n1️⃣ KALMAN FILTERING")
            kalman_prices = apply_kalman_filter(prices)
            volatility = self._calculate_volatility(kalman_prices)
            print(f"   ✅ Smoothed {len(prices)} candles")
            print(f"   ✅ Volatility: {volatility:.6f}")
            
            # 2. HMM STATE DETECTION
            print("\n2️⃣ HMM STATE DETECTION")
            if not self.hmm_model.is_trained:
                hmm_features = self._prepare_hmm_features(kalman_prices)
                self.hmm_model.train(hmm_features)
                print(f"   ✅ HMM trained")
            
            hmm_features = self._prepare_hmm_features(kalman_prices[-100:])
            state = self.hmm_model.predict_states(hmm_features)[-1]
            state_history = self.hmm_model.predict_states(hmm_features)
            state_confidence = self.hmm_model.get_state_stability(state_history)
            
            state_names = {0: 'BEARISH', 1: 'RANGING', 2: 'BULLISH'}
            print(f"   ✅ State: {state} ({state_names[state]})")
            print(f"   ✅ Confidence: {state_confidence:.1%}")
            
            # 3. CONTEXT VALIDATION
            print("\n3️⃣ CONTEXT VALIDATION")
            context = self.context_analyzer.analyze_with_context(
                prices=kalman_prices,
                volumes=volumes,
                hmm_state=state
            )
            
            signal_type = context['signal']
            base_confidence = context['confidence']
            reasoning = context['reasoning']
            
            print(f"   ✅ Signal: {signal_type}")
            print(f"   ✅ Base Confidence: {base_confidence:.1%}")
            print(f"   ✅ Reasoning: {reasoning[:60]}...")
            
            if signal_type == 'WAIT':
                return self._wait(reasoning)
            
            # 4. LIQUIDITY DETECTION
            print("\n4️⃣ LIQUIDITY DETECTION")
            liquidity_map = self.liquidity_detector.detect_liquidity_zones(kalman_prices, volumes)
            
            print(f"   ✅ Buy-side zones: {len(liquidity_map['buy_side_liquidity'])}")
            print(f"   ✅ Sell-side zones: {len(liquidity_map['sell_side_liquidity'])}")
            print(f"   ✅ Nearest buy: {liquidity_map['nearest_buy_zone']:.5f} ({liquidity_map['distance_to_buy_pct']:.2f}% away)")
            print(f"   ✅ Nearest sell: {liquidity_map['nearest_sell_zone']:.5f} ({liquidity_map['distance_to_sell_pct']:.2f}% away)")
            
            # Calculate stop hunt risk
            stop_hunt = self.liquidity_detector.calculate_stop_hunt_risk(
                current_price, liquidity_map, signal_type
            )
            
            print(f"   ⚠️  Stop Hunt Risk: {stop_hunt['risk_level']} ({stop_hunt['risk_score']:.1%})")
            print(f"   ⚠️  Target: {stop_hunt['hunt_target']:.5f} ({stop_hunt['distance_pct']:.2f}% away)")
            
            # 5. WHIPSAW CHECK
            print("\n5️⃣ WHIPSAW CHECK")
            whipsaw = self._detect_whipsaw(kalman_prices, volumes)
            print(f"   ✅ Whipsaw Risk: {whipsaw['risk']} ({whipsaw['score']:.1%})")
            
            # Adjust confidence based on risks
            confidence = base_confidence
            
            if stop_hunt['risk_level'] == 'EXTREME':
                confidence -= 0.25
                print(f"   ❌ EXTREME stop hunt risk - reducing confidence by 25%")
            elif stop_hunt['risk_level'] == 'HIGH':
                confidence -= 0.15
                print(f"   ⚠️  HIGH stop hunt risk - reducing confidence by 15%")
            
            if whipsaw['risk'] == 'HIGH':
                confidence -= 0.15
                print(f"   ⚠️  HIGH whipsaw risk - reducing confidence by 15%")
            elif whipsaw['risk'] == 'MEDIUM':
                confidence -= 0.08
                print(f"   ⚠️  MEDIUM whipsaw risk - reducing confidence by 8%")
            
            confidence = max(0.3, confidence)
            
            # Block extremely risky signals
            if stop_hunt['risk_level'] == 'EXTREME' and confidence < 0.55:
                return self._wait(
                    f"EXTREME stop hunt risk. Price {stop_hunt['distance_pct']:.2f}% from "
                    f"liquidity at {stop_hunt['hunt_target']:.5f}. Waiting for safer entry."
                )
            
            # 6. HMM-ADAPTIVE TP/SL
            print("\n6️⃣ HMM-ADAPTIVE TP/SL")
            tpsl = self.tpsl_calculator.calculate_tp_sl(
                prices=kalman_prices,
                current_price=current_price,
                signal_type=signal_type,
                hmm_state=state,
                state_confidence=state_confidence
            )
            
            tp = tpsl['tp']
            sl = tpsl['sl']
            
            print(f"   ✅ Method: {tpsl['method']}")
            print(f"   ✅ State: {tpsl['hmm_state_name']}")
            print(f"   ✅ Entry: {current_price:.5f}")
            print(f"   ✅ TP: {tp:.5f} (mult: {tpsl['tp_multiplier']:.2f})")
            print(f"   ✅ SL: {sl:.5f} (mult: {tpsl['sl_multiplier']:.2f})")
            print(f"   ✅ R:R: {tpsl['risk_reward']:.2f}:1")
            
            # 7. LIQUIDITY-ADJUSTED SL
            print("\n7️⃣ LIQUIDITY-ADJUSTED SL")
            buffer = volatility * 0.2
            adjusted_sl = self.liquidity_detector.adjust_stop_loss(
                sl, liquidity_map, signal_type, buffer
            )
            
            if adjusted_sl != sl:
                print(f"   🎯 SL ADJUSTED: {sl:.5f} → {adjusted_sl:.5f}")
                print(f"   🎯 Reason: Avoiding liquidity zone")
                sl = adjusted_sl
            else:
                print(f"   ✅ SL placement OK (not in liquidity zone)")
            
            # Recalculate R:R after adjustment
            if signal_type == 'BUY':
                reward = tp - current_price
                risk = current_price - sl
            else:
                reward = current_price - tp
                risk = sl - current_price
            
            final_rr = reward / risk if risk > 0 else 0
            
            if final_rr < 1.0:
                return self._wait(f"R:R too low after SL adjustment: {final_rr:.2f}:1")
            
            # 8. FINAL OUTPUT
            print("\n✅ SIGNAL APPROVED")
            print(f"   Type: {signal_type}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Final R:R: {final_rr:.2f}:1")
            print("="*70 + "\n")
            
            return {
                'signal': signal_type,
                'signal_type': signal_type,
                'entry': float(current_price),
                'tp': float(tp),
                'sl': float(sl),
                'confidence': float(confidence),
                'reasoning': reasoning,
                'hmm_state': state_names[state],
                'hmm_confidence': float(state_confidence),
                'stop_hunt_risk': stop_hunt['risk_level'],
                'whipsaw_risk': whipsaw['risk'],
                'liquidity_adjusted': adjusted_sl != tpsl['sl'],
                'risk_metrics': {
                    'risk_reward_ratio': float(final_rr),
                    'potential_profit_pct': float((reward / current_price) * 100),
                    'potential_loss_pct': float((risk / current_price) * 100),
                    'volatility': float(volatility)
                }
            }
        
        except Exception as e:
            error_msg = f"Signal generation error: {type(e).__name__}: {str(e)}"
            print(f"\n❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return self._wait(error_msg)
    
    def _wait(self, reason: str) -> Dict:
        """Return WAIT signal"""
        return {
            'signal': 'WAIT',
            'signal_type': 'WAIT',
            'entry': 0.0,
            'tp': 0.0,
            'sl': 0.0,
            'confidence': 0.0,
            'reasoning': reason,
            'hmm_state': 'UNKNOWN',
            'stop_hunt_risk': 'UNKNOWN',
            'whipsaw_risk': 'UNKNOWN',
            'risk_metrics': {
                'risk_reward_ratio': 0.0,
                'potential_profit_pct': 0.0,
                'potential_loss_pct': 0.0
            }
        }
