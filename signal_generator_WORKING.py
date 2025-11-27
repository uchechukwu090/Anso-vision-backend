"""
COMPLETE WORKING SIGNAL GENERATOR - TESTED VERSION
This replaces the broken signal_generator.py
"""
import numpy as np
from kalman_filter import apply_kalman_filter
from hmm_model import MarketHMM
from wavelet_analysis import denoise_signal_with_wavelets
from monte_carlo_optimizer import MonteCarloOptimizer
from atr_calculator import ATRCalculator

class SignalGenerator:
    def __init__(self, n_hmm_components=3, wavelet_level=4, covariance_type='diag', random_state=None, n_monte_carlo_sims=10000):
        self.kalman_filter = None
        self.hmm_model = MarketHMM(n_components=n_hmm_components, covariance_type=covariance_type, random_state=random_state)
        self.wavelet_level = wavelet_level
        self.monte_carlo = MonteCarloOptimizer(n_simulations=n_monte_carlo_sims, confidence_level=0.95)
        self.atr_calc = ATRCalculator(atr_period=14, tp_multiplier=2.0, sl_multiplier=1.0)
        print(f"✅ SignalGenerator initialized (HMM components={n_hmm_components})")

    def _prepare_hmm_features(self, smoothed_data):
        """Prepare features for HMM"""
        log_returns = np.diff(np.log(smoothed_data + 1e-10))  # Add small value to avoid log(0)
        
        volatility = np.zeros_like(log_returns)
        window = 10
        for i in range(window - 1, len(log_returns)):
            volatility[i] = np.std(log_returns[i - window + 1:i + 1])
        
        features = np.vstack([log_returns, volatility]).T
        features[~np.isfinite(features)] = 0
        
        return features

    def generate_signals(self, raw_price_data):
        """Generate trading signals with full error handling"""
        try:
            # Validate input
            if not isinstance(raw_price_data, np.ndarray) or raw_price_data.ndim != 1:
                print("❌ Invalid input: must be 1D numpy array")
                return self._return_wait("Invalid input format")
            
            if len(raw_price_data) < 100:
                print(f"❌ Insufficient data: {len(raw_price_data)} candles (need 100+)")
                return self._return_wait(f"Need 100+ candles, got {len(raw_price_data)}")
            
            # Check for valid prices
            if np.any(raw_price_data <= 0):
                print("❌ Invalid prices: contains zero or negative values")
                return self._return_wait("Invalid price data")
            
            print(f"📊 Processing {len(raw_price_data)} candles...")
            
            # Use rolling window
            MAX_ROLLING_WINDOW = 250
            if len(raw_price_data) > MAX_ROLLING_WINDOW:
                raw_price_data = raw_price_data[-MAX_ROLLING_WINDOW:]
            
            # 1. Kalman Filter
            print("🔄 Applying Kalman filter...")
            smoothed_data = apply_kalman_filter(raw_price_data)
            
            # 2. HMM Features
            print("🔄 Preparing HMM features...")
            hmm_features = self._prepare_hmm_features(smoothed_data)
            
            if len(hmm_features) < self.hmm_model.n_components:
                print(f"❌ Not enough features: {len(hmm_features)} < {self.hmm_model.n_components}")
                return self._return_wait("Insufficient features for HMM")
            
            # 3. Train HMM (CRITICAL FIX)
            print("🔄 Training HMM model...")
            try:
                # Always train with fresh data
                self.hmm_model.train(hmm_features)
                print("✅ HMM trained successfully")
            except Exception as e:
                print(f"❌ HMM training failed: {e}")
                return self._return_wait(f"HMM training error: {str(e)}")
            
            # 4. Predict states
            print("🔄 Predicting HMM states...")
            predicted_states = self.hmm_model.predict_states(hmm_features)
            state_probabilities = self.hmm_model.get_state_probabilities(hmm_features)
            
            # 5. Wavelet refinement
            if state_probabilities.shape[1] > 0:
                most_likely_state_prob = state_probabilities[:, predicted_states[-1]]
                refined_prob = denoise_signal_with_wavelets(most_likely_state_prob, level=self.wavelet_level)
            else:
                refined_prob = np.array([])
            
            # 6. Generate signal
            current_market_context = int(predicted_states[-1]) if len(predicted_states) > 0 else None
            
            if current_market_context is None:
                print("❌ No market context available")
                return self._return_wait("No HMM state predicted")
            
            # Identify bullish/bearish states
            if hasattr(self.hmm_model.model, 'means_') and self.hmm_model.model.means_.shape[1] > 0:
                mean_log_returns = self.hmm_model.model.means_[:, 0]
                bullish_state_idx = int(np.argmax(mean_log_returns))
                bearish_state_idx = int(np.argmin(mean_log_returns))
            else:
                bullish_state_idx = 2
                bearish_state_idx = 0
            
            # Get probabilities
            prob_bullish = float(state_probabilities[-1, bullish_state_idx]) if state_probabilities.shape[1] > bullish_state_idx else 0.0
            prob_bearish = float(state_probabilities[-1, bearish_state_idx]) if state_probabilities.shape[1] > bearish_state_idx else 0.0
            
            last_price = float(raw_price_data[-1])
            
            print(f"📊 HMM State: {current_market_context}, Bullish: {prob_bullish:.2%}, Bearish: {prob_bearish:.2%}")
            
            # Decision thresholds
            BULLISH_THRESHOLD = 0.6
            BEARISH_THRESHOLD = 0.6
            
            # Generate BUY signal
            if prob_bullish > BULLISH_THRESHOLD:
                print(f"✅ Generating BUY signal (confidence: {prob_bullish:.2%})")
                return self._generate_buy_signal(raw_price_data, last_price, prob_bullish, current_market_context, refined_prob)
            
            # Generate SELL signal
            elif prob_bearish > BEARISH_THRESHOLD:
                print(f"✅ Generating SELL signal (confidence: {prob_bearish:.2%})")
                return self._generate_sell_signal(raw_price_data, last_price, prob_bearish, current_market_context, refined_prob)
            
            # No clear signal
            else:
                print(f"⏸️ WAIT signal (Bullish: {prob_bullish:.2%}, Bearish: {prob_bearish:.2%})")
                return self._return_wait(f"Neutral state - Bullish: {prob_bullish:.2%}, Bearish: {prob_bearish:.2%}")
                
        except Exception as e:
            print(f"❌ Signal generation error: {e}")
            import traceback
            traceback.print_exc()
            return self._return_wait(f"Error: {str(e)}")
    
    def _generate_buy_signal(self, raw_price_data, last_price, confidence, context, refined_prob):
        """Generate BUY signal with ATR-based TP/SL"""
        # Use ATR for realistic TP/SL
        atr_result = self.atr_calc.calculate_tp_sl(raw_price_data, last_price, 'BUY')
        
        entry_point = atr_result['entry']
        tp = atr_result['tp']
        sl = atr_result['sl']
        
        return {
            "entry": float(entry_point),
            "tp": float(tp),
            "sl": float(sl),
            "market_context": context,
            "signal_type": "BUY",
            "confidence": float(confidence),
            "reasoning": f"Bullish HMM state (prob: {confidence:.2%}) | ATR-based stops | R:R {atr_result['risk_reward_ratio']:.1f}:1",
            "atr_info": {
                "atr_value": atr_result['atr'],
                "atr_pct": atr_result['atr_pct'],
                "risk_reward": atr_result['risk_reward_ratio'],
                "tp_pips": atr_result['tp_distance_pips'],
                "sl_pips": atr_result['sl_distance_pips']
            }
        }
    
    def _generate_sell_signal(self, raw_price_data, last_price, confidence, context, refined_prob):
        """Generate SELL signal with ATR-based TP/SL"""
        # Use ATR for realistic TP/SL
        atr_result = self.atr_calc.calculate_tp_sl(raw_price_data, last_price, 'SELL')
        
        entry_point = atr_result['entry']
        tp = atr_result['tp']
        sl = atr_result['sl']
        
        return {
            "entry": float(entry_point),
            "tp": float(tp),
            "sl": float(sl),
            "market_context": context,
            "signal_type": "SELL",
            "confidence": float(confidence),
            "reasoning": f"Bearish HMM state (prob: {confidence:.2%}) | ATR-based stops | R:R {atr_result['risk_reward_ratio']:.1f}:1",
            "atr_info": {
                "atr_value": atr_result['atr'],
                "atr_pct": atr_result['atr_pct'],
                "risk_reward": atr_result['risk_reward_ratio'],
                "tp_pips": atr_result['tp_distance_pips'],
                "sl_pips": atr_result['sl_distance_pips']
            }
        }
    
    def _return_wait(self, reason):
        """Return WAIT signal with reason"""
        return {
            "entry": None,
            "tp": None,
            "sl": None,
            "signal_type": "WAIT",
            "confidence": 0.0,
            "market_context": "Insufficient Data",
            "reasoning": reason
        }


if __name__ == '__main__':
    # Test
    np.random.seed(42)
    test_prices = np.cumsum(np.random.normal(0.001, 0.02, 300)) + 100
    
    signal_gen = SignalGenerator(n_hmm_components=3, wavelet_level=2, random_state=42)
    signals = signal_gen.generate_signals(test_prices)
    
    print("\n=== Generated Signal ===")
    print(f"Signal Type: {signals.get('signal_type')}")
    print(f"Entry: {signals.get('entry')}")
    print(f"TP: {signals.get('tp')}")
    print(f"SL: {signals.get('sl')}")
    print(f"Confidence: {signals.get('confidence')}")
    print(f"Reasoning: {signals.get('reasoning')}")
