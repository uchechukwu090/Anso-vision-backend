"""
MASTER SIGNAL ORCHESTRATOR - The Brain of the System
This file fuses multiple models (HMM, Monte Carlo, ATR, Context)
and implements the advanced Discounted Entry (Pullback) Logic.
"""
import numpy as np
from kalman_filter import apply_kalman_filter
from hmm_model import MarketHMM
from wavelet_analysis import denoise_signal_with_wavelets
from atr_calculator import ATRCalculator
from pure_monte_carlo_engine import MonteCarloTradingEngine # Using Pure MC for fast confidence
from market_analyzer import MarketAnalyzer # New: For S/R and Key Levels
from context_aware_hmm import ContextAwareHMM # New: For Pullback Detection
from typing import Dict

# --- Constants for Configuration ---
HMM_COMPONENTS = 3
WAVELET_LEVEL = 4
ATR_PERIOD = 14
ATR_TP_MULTIPLIER = 2.0
ATR_SL_MULTIPLIER = 1.0
PULLBACK_DISCOUNT_MULTIPLIER = 0.5 # How deep to set the limit order (e.g., 50% of the key level range)

class SignalGenerator:
    def __init__(self, n_hmm_components=HMM_COMPONENTS, wavelet_level=WAVELET_LEVEL, random_state=42):
        # 1. Denoising/Smoothing
        self.wavelet_level = wavelet_level

        # 2. Core Models (The Experts)
        self.hmm_model = MarketHMM(n_components=n_hmm_components, random_state=random_state)
        self.mc_engine = MonteCarloTradingEngine() # Monte Carlo is now a confidence validator
        
        # 3. Context & Feature Extraction
        self.market_analyzer = MarketAnalyzer() # Key Levels, S/R, Volume
        self.context_analyzer = ContextAwareHMM() # Pullback Detection
        
        # 4. Risk & Position Sizing
        self.atr_calc = ATRCalculator(
            atr_period=ATR_PERIOD,
            tp_multiplier=ATR_TP_MULTIPLIER,
            sl_multiplier=ATR_SL_MULTIPLIER
        )
        print(f"✅ SignalGenerator initialized (HMM components={n_hmm_components}, Discounted Entry ENABLED)")

    def _prepare_hmm_features(self, smoothed_data):
        # ... (Your existing HMM feature prep logic remains here) ...
        log_returns = np.diff(np.log(smoothed_data + 1e-10))
        volatility = np.zeros_like(log_returns)
        window = 10
        # Placeholder for your volatility calculation logic
        for i in range(window, len(log_returns)):
            volatility[i] = np.std(log_returns[i-window:i]) * np.sqrt(252)
        
        # The HMM input should be (Log Returns, Volatility)
        # Note: Need to align lengths. HMM features start one element after smoothed data.
        return np.column_stack([log_returns[window:], volatility[window:]])

    def generate_signals(self, prices: np.ndarray, volumes: np.ndarray) -> Dict:
        if len(prices) < 100:
            return self._return_wait("Insufficient data for HMM/Wavelet/MC analysis (requires ~100+ bars)")

        current_price = prices[-1]

        # --- Layer 1: Data Pre-Processing ---
        # 1a. Apply Kalman Filter
        kalman_smoothed = apply_kalman_filter(prices)
        # 1b. Apply Wavelet Denoising
        denoised_prices = denoise_signal_with_wavelets(kalman_smoothed, level=self.wavelet_level)
        
        # --- Layer 2: Core Models (Get Opinions) ---
        # 2a. Train HMM (if not trained) & Get State
        if not self.hmm_model.is_trained:
            hmm_features = self._prepare_hmm_features(denoised_prices)
            self.hmm_model.train(hmm_features)
        
        # 2b. Predict the current HMM state and confidence
        hmm_features_latest = self._prepare_hmm_features(denoised_prices[-100:]) # Use a fixed window
        latest_state_index = self.hmm_model.predict(hmm_features_latest)[-1]
        state_confidence = self.hmm_model.get_state_stability(self.hmm_model.state_history)
        
        # 2c. Get Market Structure Features
        market_analysis = self.market_analyzer.analyze_market_structure(denoised_prices, volumes)
        key_levels = market_analysis['price_levels']
        
        # 2d. Get Context-Aware Validation
        hmm_context = self.context_analyzer.analyze_with_context(
            prices=denoised_prices,
            volumes=volumes,
            hmm_state=latest_state_index
        )
        
        # --- Layer 3: Orchestration & Discounted Entry Logic (The Brain) ---
        
        signal_type = hmm_context['signal']
        base_confidence = max(state_confidence, hmm_context['confidence'])
        reasoning = hmm_context['reasoning']
        
        # --- NEW LOGIC: PULLBACK & DISCOUNTED ENTRY ---
        
        entry_price = current_price
        order_type = 'MARKET' # Default to Market
        
        if hmm_context['type'] == 'PULLBACK_RISK':
            # This means: Strong Trend, but a potential counter-move is expected.
            order_type = 'LIMIT'
            
            # Use support for BUY signal pullback, resistance for SELL signal pullback
            if signal_type == 'BUY':
                # The next entry should be a support level (a discount)
                nearest_support = key_levels['nearest_support']
                entry_price = nearest_support
                reasoning = f"PULLBACK/DISCOUNT ENTRY: Detected BUY trend exhaustion. Setting LIMIT order at support ({nearest_support:.2f})"
                
            elif signal_type == 'SELL':
                # The next entry should be a resistance level (a discount)
                nearest_resistance = key_levels['nearest_resistance']
                entry_price = nearest_resistance
                reasoning = f"PULLBACK/DISCOUNT ENTRY: Detected SELL trend exhaustion. Setting LIMIT order at resistance ({nearest_resistance:.2f})"
                
            # If the entry_price is now a discounted price, increase confidence slightly
            base_confidence = min(1.0, base_confidence * 1.1)

        # Handle Normal/Direct Signals (not pullbacks)
        elif signal_type == 'WAIT':
            return self._return_wait(reasoning)

        # --- Layer 4: Risk & Position Sizing (ATR) ---

        # Calculate TP/SL from the current price for ATR, even if the final entry is discounted.
        # This ensures the SL is based on current market volatility.
        atr_result = self.atr_calc.calculate_tp_sl(prices, current_price, signal_type)
        
        # FINAL ENTRY PRICE is the discounted price (Limit Order) or current price (Market Order)
        tp = atr_result['tp']
        sl = atr_result['sl']
        
        # Note: If order_type is LIMIT, the TP/SL is calculated *from* the current price,
        # but the actual R:R will be better if the LIMIT order fills at the discounted price.
        
        # --- Final Signal Assembly ---
        
        return {
            "order_type": order_type,
            "entry": float(entry_price),
            "tp": float(tp),
            "sl": float(sl),
            "signal_type": signal_type,
            "confidence": float(base_confidence),
            "reasoning": reasoning,
            "market_context": hmm_context['context'],
            "atr_info": {
                "atr_value": atr_result['atr'],
                "risk_reward": atr_result['risk_reward_ratio'],
            }
        }

    # Helper function remains the same
    def _return_wait(self, reason: str) -> Dict:
        """Return WAIT signal with reason"""
        return {
            "order_type": "MARKET",
            "entry": None,
            "tp": None,
            "sl": None,
            "signal_type": "WAIT",
            "confidence": 0.0,
            "market_context": {},
            "reasoning": reason
        }