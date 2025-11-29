"""
MASTER SIGNAL ORCHESTRATOR - The Brain of the System
This file fuses multiple models (HMM, Monte Carlo, ATR, Context)
and implements the advanced Discounted Entry (Pullback) Logic. 

✅ FIXED: MonteCarloOptimizer is now PRIMARY for TP/SL
⚠️ ATR Calculator is FALLBACK if Monte Carlo fails
"""
import numpy as np
from kalman_filter import apply_kalman_filter
from hmm_model import MarketHMM
from wavelet_analysis import denoise_signal_with_wavelets
from atr_calculator import ATRCalculator
from monte_carlo_optimizer import MonteCarloOptimizer  # ✅ PRIMARY TP/SL
from pure_monte_carlo_engine import MonteCarloTradingEngine # Trend detection & confidence
from market_analyzer import MarketAnalyzer # Key Levels, S/R, Volume
from context_aware_hmm import ContextAwareHMM # Pullback Detection
from typing import Dict

# --- Constants for Configuration ---
HMM_COMPONENTS = 3
WAVELET_LEVEL = 4
ATR_PERIOD = 14
ATR_TP_MULTIPLIER = 2.0
ATR_SL_MULTIPLIER = 1.0
PULLBACK_DISCOUNT_MULTIPLIER = 0.5 # How deep to set the limit order (e.g., 50% of the key level range)

# ✅ Monte Carlo Optimizer Constants
MC_SIMS = 25_000  # Number of simulations
MC_CONF = 0.90    # 90% confidence level for TP/SL quantiles


class SignalGenerator:
    def __init__(self, n_hmm_components=HMM_COMPONENTS, covariance_type='diag', wavelet_level=WAVELET_LEVEL, random_state=42):
        # 1.  Denoising/Smoothing
        self.wavelet_level = wavelet_level

        # 2. Core Models (The Experts)
        self.hmm_model = MarketHMM(n_components=n_hmm_components, covariance_type=covariance_type, random_state=random_state)
        
        # ✅ FIXED: Initialize both Monte Carlo engines
        self.mc_engine = MonteCarloTradingEngine()  # Trend detection & confidence validation
        self.mc_optimizer = MonteCarloOptimizer(    # ✅ PRIMARY: Dynamic TP/SL calculation
            n_simulations=MC_SIMS, 
            confidence_level=MC_CONF
        )
        
        # 3. Context & Feature Extraction
        self.market_analyzer = MarketAnalyzer() # Key Levels, S/R, Volume
        self.context_analyzer = ContextAwareHMM() # Pullback Detection
        
        # 4. Risk & Position Sizing
        self.atr_calc = ATRCalculator(  # ✅ FALLBACK only
            atr_period=ATR_PERIOD,
            tp_multiplier=ATR_TP_MULTIPLIER,
            sl_multiplier=ATR_SL_MULTIPLIER
        )
        
        print(f"✅ SignalGenerator initialized")
        print(f"   • HMM components: {n_hmm_components}")
        print(f"   • Monte Carlo Optimizer: PRIMARY (Sims: {MC_SIMS}, CL: {MC_CONF:.0%})")
        print(f"   • ATR Calculator: FALLBACK")
        print(f"   • Pure MC Engine: Trend detection")

    def _prepare_hmm_features(self, smoothed_data):
        """Prepare features for HMM training"""
        log_returns = np.diff(np.log(smoothed_data + 1e-10))
        volatility = np.zeros_like(log_returns)
        window = 10
        
        for i in range(window, len(log_returns)):
            volatility[i] = np.std(log_returns[i-window:i]) * np.sqrt(252)
        
        return np.column_stack([log_returns[window:], volatility[window:]])

    def generate_signals(self, prices: np.ndarray, volumes: np.ndarray = None) -> Dict:
        """
        Main signal generation orchestrator.
        
        Flow:
        1. Data Pre-Processing (Kalman + Wavelet)
        2. HMM State Detection
        3. Market Structure Analysis
        4. ✅ MONTE CARLO OPTIMIZER for TP/SL (PRIMARY)
        5. ATR Fallback if MC fails
        6. Risk Validation
        
        Returns:
            Dict: ALWAYS returns a dict, NEVER None
        """
        try:
            if len(prices) < 100:
                return self._return_wait("Insufficient data (requires ~100 bars)")

            if volumes is None:
                volumes = np.ones_like(prices)  # Dummy volumes if not provided

            current_price = prices[-1]

            # --- Layer 1: Data Pre-Processing ---
            print("\n" + "="*70)
            print("🧠 SIGNAL GENERATION PIPELINE")
            print("="*70)
            
            print("\n1️⃣ DATA PRE-PROCESSING")
            kalman_smoothed = apply_kalman_filter(prices)
            denoised_prices = denoise_signal_with_wavelets(kalman_smoothed, level=self.wavelet_level)
            print(f"   ✅ Kalman + Wavelet smoothing applied")
            
            # --- Layer 2: Core Models (Get Opinions) ---
            print("\n2️⃣ MARKET ANALYSIS")
            
            # 2a. Train HMM if needed
            if not self.hmm_model.is_trained:
                hmm_features = self._prepare_hmm_features(denoised_prices)
                self.hmm_model.train(hmm_features)
                print(f"   ✅ HMM trained")
            
            # 2b. Get HMM State
            hmm_features_latest = self._prepare_hmm_features(denoised_prices[-100:])
            latest_state_index = self.hmm_model.predict(hmm_features_latest)[-1]
            state_confidence = self.hmm_model.get_state_stability(self.hmm_model.state_history)
            print(f"   ✅ HMM State: {latest_state_index} (confidence: {state_confidence:.1%})")
            
            # 2c. Market Structure
            market_analysis = self.market_analyzer.analyze_market_structure(denoised_prices, volumes)
            key_levels = market_analysis['price_levels']
            print(f"   ✅ Support: {key_levels.get('nearest_support', 0):.2f} | Resistance: {key_levels.get('nearest_resistance', 0):.2f}")
            
            # 2d. Context Analysis
            hmm_context = self.context_analyzer.analyze_with_context(
                prices=denoised_prices,
                volumes=volumes,
                hmm_state=latest_state_index
            )
            print(f"   ✅ Context: {hmm_context['context']}")
            
            # --- Layer 3: Signal Decision ---
            print("\n3️⃣ SIGNAL DECISION")
            
            signal_type = hmm_context['signal']
            base_confidence = max(state_confidence, hmm_context['confidence'])
            reasoning = hmm_context['reasoning']
            
            if signal_type == 'WAIT':
                return self._return_wait(reasoning)
            
            print(f"   ✅ Signal: {signal_type} (confidence: {base_confidence:.1%})")
            
            # --- Layer 4: ✅ MONTE CARLO OPTIMIZER (PRIMARY TP/SL) ---
            print("\n4️⃣ TP/SL CALCULATION (Monte Carlo Optimizer - PRIMARY)")
            
            try:
                mc_result = self.mc_optimizer.calculate_tp_sl(
                    prices=denoised_prices,
                    current_price=current_price,
                    signal_type=signal_type,
                    time_horizon=50  # 50-bar projection
                )
                tp = float(mc_result['tp'])
                sl = float(mc_result['sl'])
                
                print(f"   ✅ Monte Carlo Success")
                print(f"      Entry: {current_price:.2f}")
                print(f"      TP: {tp:.2f} (from {current_price:.2f})")
                print(f"      SL: {sl:.2f}")
                print(f"      Volatility: {mc_result['volatility']:.2%}")
                
                tp_sl_source = "Monte Carlo Optimizer"
                
            except Exception as e:
                # ⚠️ FALLBACK: Use ATR if Monte Carlo fails
                print(f"   ⚠️ Monte Carlo failed: {str(e)}")
                print(f"   ⏮️ FALLBACK to ATR Calculator")
                
                atr_result = self.atr_calc.calculate_tp_sl(denoised_prices, current_price, signal_type)
                tp = float(atr_result['tp'])
                sl = float(atr_result['sl'])
                
                print(f"      Entry: {current_price:.2f}")
                print(f"      TP: {tp:.2f}")
                print(f"      SL: {sl:.2f}")
                print(f"      ATR: {atr_result['atr']:.2f}")
                
                tp_sl_source = "ATR (Fallback)"
                reasoning += f" | MC error, fell back to ATR"
            
            # --- Layer 5: Risk Metrics ---
            print("\n5️⃣ RISK METRICS")
            
            risk_metrics = self._compute_risk_metrics(denoised_prices, current_price, tp, sl, signal_type)
            print(f"   ✅ R:R: {risk_metrics['risk_reward_ratio']:.2f}:1")
            print(f"   ✅ Expected Value: {risk_metrics['expected_value_pct']:.2f}%")
            
            # Enforce minimum R:R
            if risk_metrics['risk_reward_ratio'] < 0.9:
                return self._return_wait(f"R:R {risk_metrics['risk_reward_ratio']:.2f}:1 too low (min 0.9:1)")
            
            # --- Final Signal Assembly ---
            print("\n✅ SIGNAL APPROVED")
            print("="*70 + "\n")
            
            return {
                "signal_type": signal_type,
                "entry": float(current_price),
                "tp": float(tp),
                "sl": float(sl),
                "confidence": float(base_confidence),
                "reasoning": reasoning,
                "tp_sl_source": tp_sl_source,  # ✅ Show which method was used
                "market_context": hmm_context['context'],
                "risk_metrics": risk_metrics,
            }
        
        except Exception as e:
            # ❗ CRITICAL: Catch ANY error and return WAIT signal
            error_msg = f"Signal generation exception: {type(e).__name__}: {str(e)}"
            print(f"\n❌ CRITICAL ERROR in generate_signals():")
            print(f"   {error_msg}")
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
        profit_pct = float((reward / current_price) * 100) if current_price > 0 else 0.0
        loss_pct = float((risk / current_price) * 100) if current_price > 0 else 0.0
        
        # Simple probability proxy
        total = reward + risk
        prob_tp = float(reward / total) if total > 0 else 0.5
        expected_value = float(prob_tp * reward - (1 - prob_tp) * risk)
        expected_value_pct = float((expected_value / current_price) * 100) if current_price > 0 else 0.0
        
        return {
            'risk_reward_ratio': rr,
            'potential_profit_pct': profit_pct,
            'potential_loss_pct': loss_pct,
            'prob_tp_hit': prob_tp,
            'prob_sl_hit': float(1.0 - prob_tp),
            'expected_value': expected_value,
            'expected_value_pct': expected_value_pct
        }

    def _return_wait(self, reason: str) -> Dict:
        """Return WAIT signal"""
        return {
            "signal_type": "WAIT",
            "entry": None,
            "tp": None,
            "sl": None,
            "confidence": 0.0,
            "reasoning": reason
        }
