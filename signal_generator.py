"""
MASTER SIGNAL ORCHESTRATOR - CLEAN & FAST
✅ Wavelet removed (Kalman does smoothing better)
✅ Kalman integrated with frequency awareness
✅ Whipsaw detector (prevents entries before reversals)
✅ Simple BUY/SELL output for MT5
✅ HMM as independent accurate core
"""
import numpy as np
from kalman_filter import apply_kalman_filter, apply_kalman_filter_enhanced
from hmm_model import MarketHMM
from atr_calculator import ATRCalculator
from monte_carlo_optimizer import MonteCarloOptimizer
from pure_monte_carlo_engine import MonteCarloTradingEngine
from market_analyzer import MarketAnalyzer
from context_aware_hmm import ContextAwareHMM
from typing import Dict
import pickle
import os
from datetime import datetime

# --- Constants ---
HMM_COMPONENTS = 3
ATR_PERIOD = 14
MC_SIMS = 25_000
MC_CONF = 0.90
MIN_CANDLES_FOR_TRAINING = 250
MIN_CANDLES_FOR_SIGNAL = 200

# Learning configuration
LEARNING_FILE = "mc_learning_state.pkl"
INITIAL_LEARNING_RATE = 0.1
MIN_LEARNING_RATE = 0.01


class WhipsawDetector:
    """
    ✅ NEW: Detects whipsaw movements before they fully form
    Prevents entries right before reversals
    """
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
    
    def detect_whipsaw_risk(self, prices: np.ndarray, volumes: np.ndarray) -> Dict:
        """
        Detect whipsaw risk using:
        1. False breakouts (wicks beyond support/resistance)
        2. Volume without follow-through
        3. Price acceleration followed by deceleration
        4. Momentum divergence
        """
        
        if len(prices) < self.lookback:
            return {
                'whipsaw_risk': 'LOW',
                'risk_level': 0.0,
                'expected_direction': None,
                'reasoning': 'Insufficient data'
            }
        
        recent = prices[-self.lookback:]
        recent_vols = volumes[-self.lookback:]
        
        # 1. FALSE BREAKOUT DETECTION
        # Check if price made extreme wicks without follow-through
        high = np.max(recent)
        low = np.min(recent)
        current = recent[-1]
        
        price_range = high - low
        distance_to_high = high - current
        distance_to_low = current - low
        
        # If price is near extreme but volume is declining = false breakout
        vol_trend = np.polyfit(np.arange(len(recent_vols)), recent_vols, 1)[0]
        vol_declining = vol_trend < 0
        
        is_false_breakout = (distance_to_high < price_range * 0.15 or distance_to_low < price_range * 0.15) and vol_declining
        
        # 2. ACCELERATION DETECTION
        # Measure if price is accelerating (might reverse soon)
        recent_moves = np.diff(recent)
        acceleration = np.polyfit(np.arange(len(recent_moves)), recent_moves, 1)[0]
        is_accelerating = abs(acceleration) > np.std(recent_moves) * 0.5
        
        # 3. MOMENTUM DIVERGENCE
        # Check if momentum is weakening despite price moving
        momentum = recent_moves[-5:].sum()
        volatility = np.std(recent_moves[-5:])
        momentum_weakness = volatility > abs(momentum) * 0.3
        
        # 4. CALCULATE WHIPSAW RISK
        risk_factors = sum([is_false_breakout, is_accelerating, momentum_weakness])
        
        if risk_factors >= 2:
            whipsaw_risk = 'HIGH'
            risk_level = 0.8
        elif risk_factors == 1:
            whipsaw_risk = 'MEDIUM'
            risk_level = 0.5
        else:
            whipsaw_risk = 'LOW'
            risk_level = 0.2
        
        # 5. EXPECTED DIRECTION OF WHIPSAW
        if distance_to_high < distance_to_low:
            expected_whipsaw = 'DOWN'  # Near high, risk of pullback
        else:
            expected_whipsaw = 'UP'    # Near low, risk of bounce
        
        reasoning = f"False breakout: {is_false_breakout}, Accelerating: {is_accelerating}, Momentum weak: {momentum_weakness}"
        
        return {
            'whipsaw_risk': whipsaw_risk,
            'risk_level': float(risk_level),
            'expected_direction': expected_whipsaw,
            'reasoning': reasoning,
            'factors': {
                'false_breakout': is_false_breakout,
                'accelerating': is_accelerating,
                'momentum_weakness': momentum_weakness
            }
        }


class KalmanFrequencyAnalyzer:
    """
    ✅ ENHANCED: Kalman + Frequency awareness
    Tracks volatility cycles not just smoothing
    """
    
    def __init__(self, window: int = 30):
        self.window = window
    
    def analyze_frequency(self, prices: np.ndarray) -> Dict:
        """
        Analyze Kalman-filtered data for frequency information
        Returns: volatility trends, cycle patterns
        """
        
        if len(prices) < self.window:
            return {'volatility_trend': 'STABLE', 'frequency': 'UNKNOWN'}
        
        # Apply Kalman filter
        kalman_filtered = apply_kalman_filter(prices)
        
        # Calculate volatility on filtered data
        recent_filtered = kalman_filtered[-self.window:]
        volatility = np.std(np.diff(recent_filtered))
        volatility_baseline = np.std(np.diff(kalman_filtered[-60:]))
        
        vol_ratio = volatility / volatility_baseline if volatility_baseline > 0 else 1.0
        
        # Detect volatility trend (increasing/decreasing/stable)
        vol_windows = [
            np.std(np.diff(kalman_filtered[i:i+10])) 
            for i in range(len(kalman_filtered)-30, len(kalman_filtered)-10, 5)
        ]
        vol_trend = np.polyfit(np.arange(len(vol_windows)), vol_windows, 1)[0]
        
        if vol_trend > 0:
            vol_state = 'INCREASING'
        elif vol_trend < 0:
            vol_state = 'DECREASING'
        else:
            vol_state = 'STABLE'
        
        # Detect frequency cycle
        if vol_ratio > 1.5:
            frequency = 'HIGH'  # High volatility period
        elif vol_ratio < 0.7:
            frequency = 'LOW'   # Low volatility period
        else:
            frequency = 'NORMAL'
        
        return {
            'volatility': float(volatility),
            'volatility_ratio': float(vol_ratio),
            'volatility_trend': vol_state,
            'frequency': frequency,
            'kalman_filtered': kalman_filtered
        }


class SignalGenerator:
    def __init__(self, n_hmm_components=HMM_COMPONENTS, covariance_type='diag', random_state=42):
        
        # ✅ CLEAN CORE: Only essential models
        self.hmm_model = MarketHMM(n_components=n_hmm_components, 
                                   covariance_type=covariance_type, 
                                   random_state=random_state)
        self.mc_engine = MonteCarloTradingEngine()
        self.mc_optimizer = MonteCarloOptimizer(n_simulations=MC_SIMS, confidence_level=MC_CONF)
        self.market_analyzer = MarketAnalyzer()
        self.context_analyzer = ContextAwareHMM()
        self.atr_calc = ATRCalculator(atr_period=ATR_PERIOD, tp_multiplier=2.0, sl_multiplier=1.0)
        
        # ✅ NEW: Whipsaw detector
        self.whipsaw_detector = WhipsawDetector(lookback=20)
        
        # ✅ ENHANCED: Kalman frequency analyzer
        self.kalman_analyzer = KalmanFrequencyAnalyzer(window=30)
        
        # Learning system
        self.learning_state = self._load_learning_state()
        self.trade_history = []
        
        print(f"✅ SignalGenerator initialized (CLEAN VERSION)")
        print(f"   • HMM components: {n_hmm_components}")
        print(f"   • Whipsaw detection: ENABLED")
        print(f"   • Kalman frequency: ENABLED")
        print(f"   • Wavelet: REMOVED (Kalman does smoothing)")

    def _get_state_name(self, state_index: int) -> str:
        """Convert HMM state index to readable name"""
        state_names = {0: 'TRENDING_UP', 1: 'RANGING', 2: 'TRENDING_DOWN'}
        return state_names.get(state_index, 'RANGING')
    
    def _load_learning_state(self) -> Dict:
        """Load or initialize learning state"""
        if os.path.exists(LEARNING_FILE):
            try:
                with open(LEARNING_FILE, 'rb') as f:
                    state = pickle.load(f)
                    print(f"   📚 Loaded learning state: {len(state['trades'])} trades")
                    return state
            except:
                pass
        
        return {
            'trades': [],
            'win_rate': 0.0,
            'avg_rr': 1.5,
            'learning_rate': INITIAL_LEARNING_RATE,
            'mc_confidence': MC_CONF,
            'last_updated': None
        }
    
    def _save_learning_state(self):
        """Save learning state to disk"""
        try:
            with open(LEARNING_FILE, 'wb') as f:
                pickle.dump(self.learning_state, f)
        except Exception as e:
            print(f"⚠️ Failed to save learning state: {e}")

    def _prepare_hmm_features(self, prices):
        """Prepare features for HMM using Kalman-filtered data"""
        
        if len(prices) < 20:
            raise ValueError(f"Need at least 20 candles, got {len(prices)}")
        
        # ✅ Use Kalman-filtered prices for HMM input (cleaner signal)
        kalman_prices = apply_kalman_filter(prices)
        
        log_returns = np.diff(np.log(kalman_prices + 1e-10))
        
        window = 10
        volatility = np.zeros_like(log_returns)
        momentum = np.zeros_like(log_returns)
        
        for i in range(window, len(log_returns)):
            volatility[i] = np.std(log_returns[i-window:i])
            momentum[i] = log_returns[i-window:i].sum()
        
        # Normalize
        vol_mean = np.mean(volatility[window:])
        vol_std = np.std(volatility[window:])
        volatility_normalized = (volatility - vol_mean) / vol_std if vol_std > 0 else volatility
        
        features = np.column_stack([
            log_returns[window:], 
            volatility_normalized[window:],
            momentum[window:]
        ])
        
        return features

    def generate_signals(self, prices: np.ndarray, volumes: np.ndarray = None) -> Dict:
        """
        ✅ CLEAN SIGNAL GENERATION
        Returns: BUY, SELL, or WAIT only
        """
        try:
            if len(prices) < MIN_CANDLES_FOR_SIGNAL:
                return self._return_wait(f"Need {MIN_CANDLES_FOR_SIGNAL} candles, got {len(prices)}")

            if volumes is None:
                volumes = np.ones_like(prices)

            current_price = prices[-1]

            print("\n" + "="*70)
            print("🧠 SIGNAL GENERATION (CLEAN VERSION)")
            print("="*70)
            
            # 1. KALMAN FREQUENCY ANALYSIS
            print("\n1️⃣ KALMAN FREQUENCY ANALYSIS")
            kalman_info = self.kalman_analyzer.analyze_frequency(prices)
            kalman_prices = kalman_info['kalman_filtered']
            print(f"   ✅ Volatility: {kalman_info['volatility']:.6f} ({kalman_info['frequency']})")
            print(f"   ✅ Trend: {kalman_info['volatility_trend']}")
            
            # 2. HMM TRAINING & STATE DETECTION
            print("\n2️⃣ HMM STATE DETECTION")
            
            if not self.hmm_model.is_trained:
                hmm_features = self._prepare_hmm_features(kalman_prices)
                self.hmm_model.train(hmm_features)
                print(f"   ✅ HMM trained")
            
            hmm_features_latest = self._prepare_hmm_features(kalman_prices[-100:])
            latest_state_index = self.hmm_model.predict_states(hmm_features_latest)[-1]
            self.hmm_model.state_history = self.hmm_model.predict_states(hmm_features_latest)
            state_confidence = self.hmm_model.get_state_stability(self.hmm_model.state_history)
            print(f"   ✅ State: {latest_state_index} | Confidence: {state_confidence:.1%}")
            
            # 3. CONTEXT ANALYSIS (Trend confirmation on HMM)
            print("\n3️⃣ CONTEXT ANALYSIS")
            hmm_context = self.context_analyzer.analyze_with_context(
                prices=kalman_prices,
                volumes=volumes,
                hmm_state=latest_state_index
            )
            print(f"   ✅ HMM Signal: {hmm_context['signal']} ({hmm_context['confidence']:.1%})")
            print(f"   ✅ Context: {hmm_context['context']}")
            
            signal_type = hmm_context['signal']
            base_confidence = hmm_context['confidence']
            reasoning = hmm_context['reasoning']
            
            if signal_type == 'WAIT':
                return self._return_wait(reasoning)
            
            # 4. ✅ NEW: WHIPSAW DETECTION
            print("\n4️⃣ WHIPSAW DETECTION")
            whipsaw = self.whipsaw_detector.detect_whipsaw_risk(kalman_prices, volumes)
            print(f"   ⚠️  Risk: {whipsaw['whipsaw_risk']} (level: {whipsaw['risk_level']:.1%})")
            print(f"   ⚠️  Expected direction: {whipsaw['expected_direction']}")
            print(f"   ⚠️  {whipsaw['reasoning']}")
            
            # Adjust confidence based on whipsaw risk
            if whipsaw['whipsaw_risk'] == 'HIGH':
                confidence_adjustment = -0.2
                print(f"   ❌ HIGH whipsaw risk - reducing confidence by 20%")
            elif whipsaw['whipsaw_risk'] == 'MEDIUM':
                confidence_adjustment = -0.1
                print(f"   ⚠️  MEDIUM whipsaw risk - reducing confidence by 10%")
            else:
                confidence_adjustment = 0.0
                print(f"   ✅ LOW whipsaw risk - proceeding normally")
            
            final_confidence = max(0.3, base_confidence + confidence_adjustment)
            
            if whipsaw['whipsaw_risk'] == 'HIGH' and final_confidence < 0.50:
                return self._return_wait(f"High whipsaw risk detected. Expected {whipsaw['expected_direction']} movement. Waiting for confirmation.")
            
            # 5. CALCULATE TP/SL
            print("\n5️⃣ TP/SL CALCULATION")
            
            try:
                # ✅ NEW: Pass HMM state and market metrics to MC for adaptive behavior
                hmm_state_name = self._get_state_name(latest_state_index)
                market_volatility = kalman_info['volatility']
                market_momentum = self.market_analyzer.calculate_momentum(kalman_prices)
                
                mc_result = self.mc_optimizer.calculate_tp_sl(
                    prices=kalman_prices,
                    current_price=current_price,
                    signal_type=signal_type,
                    time_horizon=50,
                    volatility=market_volatility,
                    trend_state=hmm_state_name,
                    momentum=market_momentum
                )
                
                tp = float(mc_result['tp'])
                sl = float(mc_result['sl'])
                
                print(f"   ✅ Entry: {current_price:.4f}")
                print(f"   ✅ TP: {tp:.4f} | SL: {sl:.4f}")
                print(f"   ✅ R:R: {abs(tp - current_price) / abs(current_price - sl):.2f}:1")
                
            except Exception as e:
                print(f"   ❌ Monte Carlo failed: {e}")
                atr_result = self.atr_calc.calculate_tp_sl(kalman_prices, current_price, signal_type)
                tp = float(atr_result['tp'])
                sl = float(atr_result['sl'])
            
            # 6. RISK VALIDATION
            print("\n6️⃣ RISK METRICS")
            risk_metrics = self._compute_risk_metrics(kalman_prices, current_price, tp, sl, signal_type)
            print(f"   ✅ R:R: {risk_metrics['risk_reward_ratio']:.2f}:1")
            print(f"   ✅ Expected Value: {risk_metrics['expected_value_pct']:.2f}%")
            
            if risk_metrics['risk_reward_ratio'] < 1.0:
                return self._return_wait(f"R:R too low: {risk_metrics['risk_reward_ratio']:.2f}:1")
            
            # ✅ FINAL OUTPUT: SIMPLE BUY/SELL FOR MT5
            print("\n✅ SIGNAL APPROVED - SENDING TO MT5")
            print(f"   Signal Type: {signal_type}")
            print(f"   Confidence: {final_confidence:.1%}")
            print(f"   Whipsaw Risk: {whipsaw['whipsaw_risk']}")
            print("="*70 + "\n")
            
            return {
                "signal": signal_type,  # BUY or SELL (MT5 understands this)
                "signal_type": signal_type,
                "entry": float(current_price),
                "tp": float(tp),
                "sl": float(sl),
                "confidence": float(final_confidence),
                "reasoning": reasoning,
                "whipsaw_risk": whipsaw['whipsaw_risk'],
                "whipsaw_direction": whipsaw['expected_direction'],
                "risk_metrics": risk_metrics,
                "learning_stats": {
                    "win_rate": self.learning_state['win_rate'],
                    "trades_count": len(self.learning_state['trades'])
                }
            }
        
        except Exception as e:
            error_msg = f"Signal generation error: {type(e).__name__}: {str(e)}"
            print(f"\n❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return self._return_wait(error_msg)

    def _compute_risk_metrics(self, prices: np.ndarray, current_price: float, 
                             tp: float, sl: float, signal_type: str) -> Dict:
        """Calculate risk/reward metrics"""
        if signal_type == 'BUY':
            reward = tp - current_price
            risk = current_price - sl
        else:
            reward = current_price - tp
            risk = sl - current_price
        
        rr = float(reward / risk) if risk > 0 else 0.0
        expected_value = float((reward * 0.5) - (risk * 0.5))
        expected_value_pct = float((expected_value / current_price) * 100) if current_price > 0 else 0.0
        
        return {
            'risk_reward_ratio': rr,
            'potential_profit_pct': float((reward / current_price) * 100) if current_price > 0 else 0.0,
            'potential_loss_pct': float((risk / current_price) * 100) if current_price > 0 else 0.0,
            'expected_value': expected_value,
            'expected_value_pct': expected_value_pct
        }

    def _return_wait(self, reason: str) -> Dict:
        """Return WAIT signal"""
        return {
            "signal": "WAIT",
            "signal_type": "WAIT",
            "entry": 0.0,
            "tp": 0.0,
            "sl": 0.0,
            "confidence": 0.0,
            "reasoning": reason,
            "whipsaw_risk": "UNKNOWN",
            "risk_metrics": {
                'risk_reward_ratio': 0.0,
                'potential_profit_pct': 0.0,
                'potential_loss_pct': 0.0,
                'expected_value': 0.0
            }
        }
