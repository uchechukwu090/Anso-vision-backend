"""
MASTER SIGNAL ORCHESTRATOR - Enhanced with Learning & Breakout Detection
✅ NEW: Discounted entry detection, liquidity-based SL, S/R-based TP
✅ NEW: Volatility breakout signals
✅ NEW: Monte Carlo learning curve for parameter optimization
"""
import numpy as np
from kalman_filter import apply_kalman_filter
from hmm_model import MarketHMM
from wavelet_analysis import denoise_signal_with_wavelets
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
WAVELET_LEVEL = 2  # FIXED: Reduced from 4 to 2 for faster response (less lag)
ATR_PERIOD = 14
MC_SIMS = 25_000
MC_CONF = 0.90
MIN_CANDLES_FOR_TRAINING = 250  # ✅ Increased for stable learning
MIN_CANDLES_FOR_SIGNAL = 200

# Learning configuration
LEARNING_FILE = "mc_learning_state.pkl"
INITIAL_LEARNING_RATE = 0.1
MIN_LEARNING_RATE = 0.01


class SignalGenerator:
    def __init__(self, n_hmm_components=HMM_COMPONENTS, covariance_type='diag', 
                 wavelet_level=WAVELET_LEVEL, random_state=42):
        self.wavelet_level = wavelet_level
        
        # Core models
        self.hmm_model = MarketHMM(n_components=n_hmm_components, 
                                   covariance_type=covariance_type, 
                                   random_state=random_state)
        self.mc_engine = MonteCarloTradingEngine()
        self.mc_optimizer = MonteCarloOptimizer(n_simulations=MC_SIMS, confidence_level=MC_CONF)
        self.market_analyzer = MarketAnalyzer()
        self.context_analyzer = ContextAwareHMM()
        self.atr_calc = ATRCalculator(atr_period=ATR_PERIOD, tp_multiplier=2.0, sl_multiplier=1.0)
        
        # ✅ NEW: Learning system for Monte Carlo
        self.learning_state = self._load_learning_state()
        self.trade_history = []
        
        print(f"✅ SignalGenerator initialized")
        print(f"   • HMM components: {n_hmm_components}")
        print(f"   • Monte Carlo: {MC_SIMS} sims, {MC_CONF:.0%} confidence")
        print(f"   • Learning: Enabled (trades tracked: {len(self.learning_state['trades'])})")

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
    
    def _update_learning(self, signal_type: str, entry: float, tp: float, sl: float, 
                        outcome: str = None, exit_price: float = None):
        """
        Update learning parameters based on trade outcomes
        outcome: 'win', 'loss', or None (pending)
        """
        trade = {
            'timestamp': datetime.now().isoformat(),
            'signal': signal_type,
            'entry': entry,
            'tp': tp,
            'sl': sl,
            'outcome': outcome,
            'exit_price': exit_price
        }
        
        self.learning_state['trades'].append(trade)
        
        # If trade completed, update statistics
        if outcome:
            completed_trades = [t for t in self.learning_state['trades'] if t['outcome']]
            wins = len([t for t in completed_trades if t['outcome'] == 'win'])
            
            self.learning_state['win_rate'] = wins / len(completed_trades) if completed_trades else 0.0
            
            # Adjust Monte Carlo confidence based on win rate
            if self.learning_state['win_rate'] < 0.4:
                self.learning_state['mc_confidence'] = min(0.95, self.learning_state['mc_confidence'] + 0.01)
            elif self.learning_state['win_rate'] > 0.6:
                self.learning_state['mc_confidence'] = max(0.85, self.learning_state['mc_confidence'] - 0.01)
            
            # Update learning rate (decay)
            trades_count = len(completed_trades)
            self.learning_state['learning_rate'] = max(
                MIN_LEARNING_RATE,
                INITIAL_LEARNING_RATE * (0.99 ** trades_count)
            )
            
            self.learning_state['last_updated'] = datetime.now().isoformat()
            self._save_learning_state()
            
            print(f"   📊 Learning updated: WR={self.learning_state['win_rate']:.1%}, "
                  f"MC_Conf={self.learning_state['mc_confidence']:.2f}")

    def _prepare_hmm_features(self, smoothed_data):
        """Enhanced feature preparation with more indicators"""
        if len(smoothed_data) < 20:
            raise ValueError(f"Need at least 20 candles, got {len(smoothed_data)}")
        
        log_returns = np.diff(np.log(smoothed_data + 1e-10))
        
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

    def _calculate_volatility_adjusted_threshold(self, prices: np.ndarray) -> float:
        """
        ✅ Calculate adaptive discount threshold based on volatility
        High volatility = wider threshold (0.35-0.40)
        Low volatility = tighter threshold (0.25-0.30)
        """
        if len(prices) < 20:
            return 0.30  # Default
        
        # Calculate recent volatility (last 20 candles)
        returns = np.diff(np.log(prices[-20:]))
        volatility = np.std(returns)
        
        # Normalize to 0-1 range (typical crypto volatility: 0.01-0.05)
        norm_vol = min(1.0, max(0.0, (volatility - 0.01) / 0.04))
        
        # Map to threshold: low vol = 0.25, high vol = 0.40
        threshold = 0.25 + (norm_vol * 0.15)
        
        return float(threshold)
    
    def _detect_discounted_entry(self, prices: np.ndarray, signal_type: str, 
                                 support: float, resistance: float, trend_strength: float = 0.5) -> Dict:
        """
        ✅ ENHANCED: Intelligent entry with TWO modes:
        - FRESH TREND: Wait for deep discount (25-30% zone)
        - ESTABLISHED TREND: Enter on pullbacks (35-50% zone)
        
        Args:
            trend_strength: 0.0-1.0 from market_analyzer
        """
        current = prices[-1]
        price_range = resistance - support
        
        if price_range <= 0:
            return {
                'is_discounted': False,
                'entry_mode': 'INVALID_RANGE',
                'should_wait': True,
                'reason': 'Invalid support/resistance range'
            }
        
        # Get volatility-adjusted base threshold
        base_threshold = self._calculate_volatility_adjusted_threshold(prices)
        
        # Detect recent momentum (trend stage)
        recent_10_change = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
        recent_20_change = (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 else 0
        
        # Classify trend stage
        is_fresh_trend = trend_strength < 0.4 or abs(recent_20_change) < 0.015  # <1.5% move
        is_established_trend = trend_strength > 0.6 and abs(recent_20_change) > 0.02  # >2% move
        
        if signal_type == 'BUY':
            distance_from_support = (current - support) / price_range
            
            # MODE 1: Fresh trend - strict discount
            if is_fresh_trend:
                is_discounted = distance_from_support < base_threshold
                return {
                    'is_discounted': is_discounted,
                    'entry_mode': 'FRESH_TREND',
                    'discount_pct': float((1 - distance_from_support) * 100),
                    'threshold_used': base_threshold,
                    'distance_pct': float(distance_from_support * 100),
                    'should_wait': not is_discounted,
                    'reason': f"Fresh trend - {'✅ In discount zone' if is_discounted else f'⏳ Wait for pullback to {base_threshold:.1%}'} (currently {distance_from_support:.1%})"
                }
            
            # MODE 2: Established trend - pullback entry
            elif is_established_trend:
                pullback_threshold = base_threshold + 0.15
                recent_high = np.max(prices[-10:])
                pullback_depth = (recent_high - current) / recent_high if recent_high > 0 else 0
                
                is_pullback_zone = 0.20 < distance_from_support < pullback_threshold
                has_pullback = pullback_depth > 0.01  # 1% pullback minimum
                
                is_valid = is_pullback_zone and has_pullback
                
                return {
                    'is_discounted': is_valid,
                    'entry_mode': 'MOMENTUM_PULLBACK',
                    'pullback_depth_pct': float(pullback_depth * 100),
                    'distance_pct': float(distance_from_support * 100),
                    'threshold_used': pullback_threshold,
                    'should_wait': not is_valid,
                    'reason': f"Trend moving - {'✅ Pullback entry' if is_valid else f'⏳ Need {pullback_depth*100:.1f}% pullback or position <{pullback_threshold:.1%}'} (at {distance_from_support:.1%})"
                }
            
            # MODE 3: Moderate trend - standard discount
            else:
                moderate_threshold = base_threshold + 0.05
                is_discounted = distance_from_support < moderate_threshold
                
                return {
                    'is_discounted': is_discounted,
                    'entry_mode': 'MODERATE_TREND',
                    'distance_pct': float(distance_from_support * 100),
                    'threshold_used': moderate_threshold,
                    'should_wait': not is_discounted,
                    'reason': f"Moderate trend - {'✅ Fair entry' if is_discounted else f'⏳ Wait for <{moderate_threshold:.1%}'} (at {distance_from_support:.1%})"
                }
        
        else:  # SELL
            distance_from_resistance = (resistance - current) / price_range
            
            # MODE 1: Fresh trend - strict premium
            if is_fresh_trend:
                is_premium = distance_from_resistance < base_threshold
                return {
                    'is_discounted': is_premium,
                    'entry_mode': 'FRESH_TREND',
                    'premium_pct': float((1 - distance_from_resistance) * 100),
                    'threshold_used': base_threshold,
                    'distance_pct': float(distance_from_resistance * 100),
                    'should_wait': not is_premium,
                    'reason': f"Fresh downtrend - {'✅ In premium zone' if is_premium else f'⏳ Wait for rally to {base_threshold:.1%}'} (currently {distance_from_resistance:.1%})"
                }
            
            # MODE 2: Established trend - rally entry
            elif is_established_trend:
                pullback_threshold = base_threshold + 0.15
                recent_low = np.min(prices[-10:])
                rally_depth = (current - recent_low) / recent_low if recent_low > 0 else 0
                
                is_pullback_zone = 0.20 < distance_from_resistance < pullback_threshold
                has_rally = rally_depth > 0.01
                
                is_valid = is_pullback_zone and has_rally
                
                return {
                    'is_discounted': is_valid,
                    'entry_mode': 'MOMENTUM_PULLBACK',
                    'rally_depth_pct': float(rally_depth * 100),
                    'distance_pct': float(distance_from_resistance * 100),
                    'threshold_used': pullback_threshold,
                    'should_wait': not is_valid,
                    'reason': f"Downtrend moving - {'✅ Rally entry' if is_valid else f'⏳ Need {rally_depth*100:.1f}% rally or position <{pullback_threshold:.1%}'} (at {distance_from_resistance:.1%})"
                }
            
            # MODE 3: Moderate trend
            else:
                moderate_threshold = base_threshold + 0.05
                is_premium = distance_from_resistance < moderate_threshold
                
                return {
                    'is_discounted': is_premium,
                    'entry_mode': 'MODERATE_TREND',
                    'distance_pct': float(distance_from_resistance * 100),
                    'threshold_used': moderate_threshold,
                    'should_wait': not is_premium,
                    'reason': f"Moderate downtrend - {'✅ Fair entry' if is_premium else f'⏳ Wait for <{moderate_threshold:.1%}'} (at {distance_from_resistance:.1%})"
                }

    def _find_liquidity_zones(self, prices: np.ndarray, lookback: int = 50) -> Dict:
        """
        ✅ NEW: Find liquidity zones (swing highs/lows) for better SL placement
        """
        recent = prices[-lookback:] if len(prices) > lookback else prices
        
        swing_highs = []
        swing_lows = []
        
        # Find swing points (3-bar patterns)
        for i in range(2, len(recent) - 2):
            # Swing high
            if recent[i] > recent[i-1] and recent[i] > recent[i-2] and \
               recent[i] > recent[i+1] and recent[i] > recent[i+2]:
                swing_highs.append(recent[i])
            
            # Swing low
            if recent[i] < recent[i-1] and recent[i] < recent[i-2] and \
               recent[i] < recent[i+1] and recent[i] < recent[i+2]:
                swing_lows.append(recent[i])
        
        current = prices[-1]
        
        # Find nearest swing low (for BUY SL) and swing high (for SELL SL)
        nearest_swing_low = min(swing_lows, key=lambda x: abs(current - x)) if swing_lows else current * 0.99
        nearest_swing_high = min(swing_highs, key=lambda x: abs(current - x)) if swing_highs else current * 1.01
        
        return {
            'swing_lows': swing_lows,
            'swing_highs': swing_highs,
            'nearest_swing_low': float(nearest_swing_low),
            'nearest_swing_high': float(nearest_swing_high),
            'liquidity_clusters': len(swing_highs) + len(swing_lows)
        }

    def _detect_volatility_breakout(self, prices: np.ndarray, volumes: np.ndarray) -> Dict:
        """
        ✅ NEW: Detect volatility breakouts for range expansion trades
        """
        if len(prices) < 30:
            return {'is_breakout': False, 'strength': 0.0}
        
        # Calculate volatility (ATR-like)
        recent_range = np.mean([prices[i] - prices[i-1] for i in range(-20, -1)])
        current_range = prices[-1] - prices[-2]
        
        # Volume confirmation
        avg_volume = np.mean(volumes[-20:-1])
        current_volume = volumes[-1]
        volume_surge = current_volume > avg_volume * 1.5
        
        # Volatility surge
        volatility_ratio = current_range / recent_range if recent_range > 0 else 1.0
        is_breakout = volatility_ratio > 1.5 and volume_surge
        
        # Determine direction
        direction = 'BUY' if prices[-1] > prices[-2] else 'SELL'
        
        return {
            'is_breakout': is_breakout,
            'strength': float(volatility_ratio),
            'direction': direction,
            'volume_surge': volume_surge,
            'reasoning': f"Volatility {volatility_ratio:.1f}x, Volume {current_volume/avg_volume:.1f}x"
        }

    def _calculate_tp_with_sr(self, current_price: float, signal_type: str, 
                              mc_tp: float, support: float, resistance: float) -> float:
        """
        ✅ NEW: Adjust TP to align with S/R levels
        """
        if signal_type == 'BUY':
            # For BUY, TP should be near resistance but not beyond
            if mc_tp > resistance:
                adjusted_tp = resistance * 0.995  # 0.5% before resistance
            else:
                adjusted_tp = mc_tp
        else:  # SELL
            # For SELL, TP should be near support but not beyond
            if mc_tp < support:
                adjusted_tp = support * 1.005  # 0.5% after support
            else:
                adjusted_tp = mc_tp
        
        return float(adjusted_tp)

    def generate_signals(self, prices: np.ndarray, volumes: np.ndarray = None) -> Dict:
        """
        Enhanced signal generation with all new features
        """
        try:
            if len(prices) < MIN_CANDLES_FOR_SIGNAL:
                return self._return_wait(
                    f"Need {MIN_CANDLES_FOR_SIGNAL} candles, got {len(prices)}"
                )

            if volumes is None:
                volumes = np.ones_like(prices)

            current_price = prices[-1]

            print("\n" + "="*70)
            print("🧠 ENHANCED SIGNAL GENERATION")
            print("="*70)
            
            # 1. Data preprocessing
            print("\n1️⃣ DATA PRE-PROCESSING")
            kalman_smoothed = apply_kalman_filter(prices)
            denoised_prices = denoise_signal_with_wavelets(kalman_smoothed, level=self.wavelet_level)
            print(f"   ✅ Smoothing applied")
            
            # 2. Market analysis
            print("\n2️⃣ MARKET ANALYSIS")
            
            if not self.hmm_model.is_trained:
                hmm_features = self._prepare_hmm_features(denoised_prices)
                self.hmm_model.train(hmm_features)
                print(f"   ✅ HMM trained")
            
            hmm_features_latest = self._prepare_hmm_features(denoised_prices[-100:])
            latest_state_index = self.hmm_model.predict_states(hmm_features_latest)[-1]
            self.hmm_model.state_history = self.hmm_model.predict_states(hmm_features_latest)
            state_confidence = self.hmm_model.get_state_stability(self.hmm_model.state_history)
            print(f"   ✅ HMM State: {latest_state_index} (conf: {state_confidence:.1%})")
            
            market_analysis = self.market_analyzer.analyze_market_structure(denoised_prices, volumes)
            key_levels = market_analysis['price_levels']
            support = key_levels.get('nearest_support', 0)
            resistance = key_levels.get('nearest_resistance', 0)
            print(f"   ✅ S/R: {support:.2f} / {resistance:.2f}")
            
            hmm_context = self.context_analyzer.analyze_with_context(
                prices=denoised_prices,
                volumes=volumes,
                hmm_state=latest_state_index
            )
            print(f"   ✅ Context: {hmm_context['context']}")
            
            # 3. ✅ NEW: Volatility breakout detection
            print("\n3️⃣ VOLATILITY BREAKOUT CHECK")
            breakout = self._detect_volatility_breakout(denoised_prices, volumes)
            if breakout['is_breakout']:
                print(f"   🔥 BREAKOUT DETECTED: {breakout['direction']} "
                      f"(strength: {breakout['strength']:.2f}x)")
                signal_type = breakout['direction']
                base_confidence = 0.75  # High confidence for breakouts
                reasoning = f"Volatility breakout: {breakout['reasoning']}"
            else:
                print(f"   ℹ️ No breakout (volatility: {breakout['strength']:.2f}x)")
                signal_type = hmm_context['signal']
                base_confidence = max(state_confidence, hmm_context['confidence'])
                reasoning = hmm_context['reasoning']
            
            if signal_type == 'WAIT' and not breakout['is_breakout']:
                return self._return_wait(reasoning)
            
            # 4. ✅ NEW: Discounted entry check (with trend strength)
            print("\n4️⃣ ENTRY PRICE ANALYSIS")
            trend_strength = market_analysis.get('trend_strength', 0.5)
            discount_check = self._detect_discounted_entry(denoised_prices, signal_type, 
                                                           support, resistance, trend_strength)
            print(f"   📊 {discount_check['reason']}")
            
            if discount_check['should_wait'] and not breakout['is_breakout']:
                # Return WAIT with distance info
                return self._return_wait_with_distance(
                    f"Price not at discounted level. {discount_check['reason']}",
                    discount_check['distance_from_discount'],
                    discount_check['distance_pct'],
                    signal_type
                )
            
            if discount_check['is_discounted']:
                print(f"   ✅ Discounted entry confirmed ({discount_check['discount_pct']:.1f}%)")
                base_confidence = min(1.0, base_confidence + 0.1)
            
            print(f"   ✅ Signal: {signal_type} (confidence: {base_confidence:.1%})")
            
            # 5. ✅ ENHANCED: TP/SL with learning-adjusted confidence
            print("\n5️⃣ TP/SL CALCULATION (Learning-Enhanced)")
            
            try:
                # Use learned confidence level
                mc_confidence = self.learning_state['mc_confidence']
                print(f"   📚 Using learned confidence: {mc_confidence:.2%}")
                
                self.mc_optimizer.confidence_level = mc_confidence
                mc_result = self.mc_optimizer.calculate_tp_sl(
                    prices=denoised_prices,
                    current_price=current_price,
                    signal_type=signal_type,
                    time_horizon=50
                )
                
                # ✅ NEW: Adjust TP based on S/R
                mc_tp = mc_result['tp']
                adjusted_tp = self._calculate_tp_with_sr(current_price, signal_type, 
                                                         mc_tp, support, resistance)
                
                # ✅ NEW: Adjust SL based on liquidity zones
                liquidity = self._find_liquidity_zones(denoised_prices)
                
                if signal_type == 'BUY':
                    # SL below nearest swing low
                    sl = liquidity['nearest_swing_low'] * 0.999
                else:
                    # SL above nearest swing high
                    sl = liquidity['nearest_swing_high'] * 1.001
                
                print(f"   ✅ Monte Carlo Success")
                print(f"      Entry: {current_price:.2f}")
                print(f"      TP: {adjusted_tp:.2f} (adjusted from {mc_tp:.2f})")
                print(f"      SL: {sl:.2f} (liquidity-based)")
                print(f"      Liquidity zones: {liquidity['liquidity_clusters']}")
                
                tp_sl_source = "MC + S/R + Liquidity"
                
            except Exception as e:
                print(f"   ⚠️ Monte Carlo failed: {str(e)}")
                print(f"   ⏮️ FALLBACK to ATR")
                
                atr_result = self.atr_calc.calculate_tp_sl(denoised_prices, current_price, signal_type)
                adjusted_tp = float(atr_result['tp'])
                sl = float(atr_result['sl'])
                
                tp_sl_source = "ATR (Fallback)"
                reasoning += " | MC failed, used ATR"
            
            # 6. Risk validation
            print("\n6️⃣ RISK METRICS")
            
            risk_metrics = self._compute_risk_metrics(denoised_prices, current_price, 
                                                      adjusted_tp, sl, signal_type)
            print(f"   ✅ R:R: {risk_metrics['risk_reward_ratio']:.2f}:1")
            print(f"   ✅ Expected Value: {risk_metrics['expected_value_pct']:.2f}%")
            
            if risk_metrics['risk_reward_ratio'] < 0.9:
                return self._return_wait(f"R:R too low: {risk_metrics['risk_reward_ratio']:.2f}:1")
            
            # ✅ NEW: Track trade for learning
            self._update_learning(signal_type, current_price, adjusted_tp, sl)
            
            print("\n✅ SIGNAL APPROVED")
            print("="*70 + "\n")
            
            return {
                "signal_type": signal_type,
                "entry": float(current_price),
                "tp": float(adjusted_tp),
                "sl": float(sl),
                "confidence": float(base_confidence),
                "reasoning": reasoning,
                "tp_sl_source": tp_sl_source,
                "market_context": hmm_context['context'],
                "risk_metrics": risk_metrics,
                "is_breakout": breakout['is_breakout'],
                "is_discounted": discount_check['is_discounted'],
                "learning_stats": {
                    "win_rate": self.learning_state['win_rate'],
                    "trades_count": len(self.learning_state['trades']),
                    "mc_confidence": self.learning_state['mc_confidence']
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
        profit_pct = float((reward / current_price) * 100) if current_price > 0 else 0.0
        loss_pct = float((risk / current_price) * 100) if current_price > 0 else 0.0
        
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
            "entry": 0.0,
            "tp": 0.0,
            "sl": 0.0,
            "confidence": 0.0,
            "reasoning": reason,
            "market_context": "N/A",
            "risk_metrics": {
                "risk_reward_ratio": 0.0,
                "potential_profit_pct": 0.0,
                "potential_loss_pct": 0.0,
                "prob_tp_hit": 0.0,
                "prob_sl_hit": 0.0,
                "expected_value": 0.0,
                "expected_value_pct": 0.0
            }
        }
    
    def _return_wait_with_distance(self, reason: str, distance: float, 
                                   distance_pct: float, signal_type: str) -> Dict:
        """
        Return WAIT signal with distance to discount zone
        """
        return {
            "signal_type": "WAIT",
            "entry": 0.0,
            "tp": 0.0,
            "sl": 0.0,
            "confidence": 0.0,
            "reasoning": reason,
            "market_context": "N/A",
            "distance_info": {
                "distance_from_discount": float(distance),
                "distance_pct": float(distance_pct),
                "direction": signal_type,
                "message": f"Price needs to move ${abs(distance):.2f} ({distance_pct:.1f}% of range) to reach discount zone"
            },
            "risk_metrics": {
                "risk_reward_ratio": 0.0,
                "potential_profit_pct": 0.0,
                "potential_loss_pct": 0.0,
                "prob_tp_hit": 0.0,
                "prob_sl_hit": 0.0,
                "expected_value": 0.0,
                "expected_value_pct": 0.0
            }
        }
