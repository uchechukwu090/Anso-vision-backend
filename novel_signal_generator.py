"""
NOVEL SIGNAL GENERATOR - NO RETAIL INDICATORS
Based on Information Theory, Market Microstructure, and Physics

This is NOT your typical RSI/MACD bot. This uses:
- Shannon Entropy (information theory)
- Order Flow Imbalance (market microstructure)
- Phase Transitions (physics)
- Complexity Measures (chaos theory)
- Multi-scale Analysis (wavelets, but not as indicator)

Goal: 75-85% accuracy through first principles, not memorized patterns
"""

import numpy as np
from scipy import stats, signal
from scipy.stats import entropy
import pywt


class NovelSignalGenerator:
    """
    Revolutionary signal generator using physics and information theory
    NO traditional indicators!
    """
    
    def __init__(self):
        self.name = "Information Theory + Market Microstructure Hybrid"
        print(f"🚀 {self.name} initialized")
        print("📊 Using: Entropy, Order Flow, Phase Transitions, Complexity")
    
    def calculate_shannon_entropy(self, prices, bins=50):
        """
        Shannon Entropy: H(X) = -Σ p(x) * log(p(x))
        Measures uncertainty/chaos in price distribution
        
        Low entropy = Trending (predictable)
        High entropy = Ranging (unpredictable)
        """
        # Create histogram of returns
        returns = np.diff(np.log(prices))
        hist, _ = np.histogram(returns, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zeros
        
        # Calculate Shannon entropy
        shannon_ent = -np.sum(hist * np.log2(hist))
        
        # Normalize (0 to 1)
        max_entropy = np.log2(bins)
        normalized = shannon_ent / max_entropy
        
        return normalized
    
    def detect_order_flow_imbalance(self, prices, volumes):
        """
        Order Flow Imbalance: Where is smart money?
        
        Theory: When price goes up on low volume vs down on high volume,
        institutions are distributing (bearish).
        
        Returns: -1 to +1
        -1 = Strong sell pressure
        +1 = Strong buy pressure
        """
        if len(prices) < 20 or len(volumes) < 20:
            return 0.0
        
        # Calculate directional volume
        price_changes = np.diff(prices[-20:])
        volume_recent = volumes[-19:]  # Match length
        
        # Volume on up moves vs down moves
        up_volume = np.sum(volume_recent[price_changes > 0])
        down_volume = np.sum(volume_recent[price_changes < 0])
        
        total_volume = up_volume + down_volume
        if total_volume == 0:
            return 0.0
        
        # Imbalance ratio
        imbalance = (up_volume - down_volume) / total_volume
        
        return float(imbalance)
    
    def detect_phase_transition(self, prices):
        """
        Phase Transition Detection (from physics)
        
        Markets behave like physical systems:
        - Stable phase = Trending
        - Unstable phase = Ranging
        - Transition = Breakout opportunity
        
        Uses variance ratio test and Hurst exponent
        """
        if len(prices) < 50:
            return 0.5, "insufficient_data"
        
        returns = np.diff(np.log(prices[-50:]))
        
        # Hurst Exponent (measures long-term memory)
        # H < 0.5 = Mean reverting
        # H = 0.5 = Random walk
        # H > 0.5 = Trending
        
        lags = range(2, 20)
        tau = [np.std(np.subtract(returns[lag:], returns[:-lag])) for lag in lags]
        
        # Linear fit in log-log space
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = reg[0]
        
        # Classify phase
        if hurst < 0.4:
            phase = "mean_reverting"
        elif hurst > 0.6:
            phase = "trending"
        else:
            phase = "random_walk"
        
        return float(hurst), phase
    
    def calculate_complexity(self, prices):
        """
        Kolmogorov Complexity Approximation
        
        Measures how "random" vs "structured" the price series is
        Uses compression ratio as proxy
        
        Low complexity = Structured pattern (tradeable)
        High complexity = Random noise (avoid)
        """
        # Normalize prices to 0-255 for compression approximation
        normalized = ((prices - np.min(prices)) / (np.max(prices) - np.min(prices)) * 255).astype(int)
        
        # Use run-length encoding as complexity proxy
        # Count consecutive same values
        changes = np.where(np.diff(normalized) != 0)[0]
        num_runs = len(changes) + 1
        
        # Compression ratio
        complexity = num_runs / len(normalized)
        
        # 0 = Highly structured (all same value)
        # 1 = Maximum complexity (every value different)
        
        return float(complexity)
    
    def multi_scale_analysis(self, prices):
        """
        Wavelet Multi-Scale Analysis
        
        NOT using wavelets as indicator, but to detect:
        - Which time scales are dominant (1h, 4h, 1d trends)
        - Energy distribution across scales
        - Coherence between scales
        """
        # Discrete Wavelet Transform
        coeffs = pywt.wavedec(prices, 'db4', level=3)
        
        # Calculate energy at each scale
        energies = [np.sum(c**2) for c in coeffs]
        total_energy = np.sum(energies)
        
        # Normalize
        energy_distribution = [e / total_energy for e in energies]
        
        # Dominant scale
        dominant_scale = np.argmax(energy_distribution)
        
        # Scale interpretation:
        # 0 = Long-term trend dominant
        # 1 = Medium-term swing dominant
        # 2-3 = Short-term noise dominant
        
        return {
            'dominant_scale': dominant_scale,
            'energy_dist': energy_distribution,
            'trend_strength': energy_distribution[0]  # Long-term component
        }
    
    def correlation_network_analysis(self, prices_1m, prices_5m, prices_15m, prices_1h):
        """
        Network Topology Analysis
        
        Build correlation network between timeframes
        Strong correlation = Trend is coherent across scales
        Weak correlation = Conflicting signals, avoid
        """
        # Ensure same length for all timeframes
        min_len = min(len(prices_1m), len(prices_5m), len(prices_15m), len(prices_1h))
        
        # Take last N points
        p1 = prices_1m[-min_len:]
        p5 = prices_5m[-min_len:]
        p15 = prices_15m[-min_len:]
        p1h = prices_1h[-min_len:]
        
        # Calculate correlations
        corr_1m_5m = np.corrcoef(p1, p5)[0, 1]
        corr_5m_15m = np.corrcoef(p5, p15)[0, 1]
        corr_15m_1h = np.corrcoef(p15, p1h)[0, 1]
        
        # Average correlation (network coherence)
        coherence = np.mean([abs(corr_1m_5m), abs(corr_5m_15m), abs(corr_15m_1h)])
        
        return {
            'coherence': float(coherence),
            'aligned': coherence > 0.7
        }
    
    def generate_signal(self, prices, volumes):
        """
        MASTER SIGNAL GENERATOR
        
        Combines all novel methods to produce signal
        """
        if len(prices) < 100:
            return self._wait_signal("Insufficient data")
        
        print("\n" + "="*60)
        print("🧠 NOVEL SIGNAL ANALYSIS")
        print("="*60)
        
        # 1. Shannon Entropy
        entropy_val = self.calculate_shannon_entropy(prices)
        print(f"📊 Shannon Entropy: {entropy_val:.3f}")
        if entropy_val > 0.8:
            print("   → High chaos (ranging market)")
        elif entropy_val < 0.4:
            print("   → Low chaos (trending market)")
        
        # 2. Order Flow Imbalance
        flow_imbalance = self.detect_order_flow_imbalance(prices, volumes)
        print(f"💰 Order Flow Imbalance: {flow_imbalance:+.3f}")
        if flow_imbalance > 0.3:
            print("   → Bullish (buy pressure)")
        elif flow_imbalance < -0.3:
            print("   → Bearish (sell pressure)")
        
        # 3. Phase Transition
        hurst, phase = self.detect_phase_transition(prices)
        print(f"🌊 Hurst Exponent: {hurst:.3f} ({phase})")
        
        # 4. Complexity
        complexity = self.calculate_complexity(prices[-100:])
        print(f"🔬 Market Complexity: {complexity:.3f}")
        if complexity < 0.3:
            print("   → Structured pattern detected")
        elif complexity > 0.7:
            print("   → Random noise detected")
        
        # 5. Multi-Scale Analysis
        wavelet_result = self.multi_scale_analysis(prices[-100:])
        print(f"🌀 Dominant Scale: {wavelet_result['dominant_scale']}")
        print(f"🌀 Trend Strength: {wavelet_result['trend_strength']:.3f}")
        
        # === DECISION LOGIC ===
        
        # BUY Conditions:
        # 1. Low entropy (trending)
        # 2. Positive order flow
        # 3. Trending phase (Hurst > 0.5)
        # 4. Low complexity (structured)
        # 5. Strong trend in wavelet
        
        buy_score = 0
        sell_score = 0
        
        # Entropy contribution
        if entropy_val < 0.5:
            buy_score += (0.5 - entropy_val) * 2  # Max +1.0
            sell_score += (0.5 - entropy_val) * 2
        
        # Order flow contribution
        buy_score += max(0, flow_imbalance) * 2  # 0 to +2.0
        sell_score += max(0, -flow_imbalance) * 2
        
        # Phase contribution
        if phase == "trending":
            if flow_imbalance > 0:
                buy_score += 1.0
            else:
                sell_score += 1.0
        elif phase == "mean_reverting":
            # Fade the move
            if flow_imbalance > 0:
                sell_score += 0.5
            else:
                buy_score += 0.5
        
        # Complexity contribution
        if complexity < 0.4:  # Structured
            buy_score += (0.4 - complexity) * 2
            sell_score += (0.4 - complexity) * 2
        
        # Wavelet contribution
        if wavelet_result['trend_strength'] > 0.5:
            if flow_imbalance > 0:
                buy_score += wavelet_result['trend_strength']
            else:
                sell_score += wavelet_result['trend_strength']
        
        # Normalize scores
        total_score = buy_score + sell_score
        if total_score > 0:
            buy_confidence = buy_score / total_score
            sell_confidence = sell_score / total_score
        else:
            buy_confidence = 0.5
            sell_confidence = 0.5
        
        print("\n" + "="*60)
        print(f"📈 BUY Score: {buy_score:.2f} | Confidence: {buy_confidence:.1%}")
        print(f"📉 SELL Score: {sell_score:.2f} | Confidence: {sell_confidence:.1%}")
        print("="*60)
        
        # Decision threshold
        THRESHOLD = 0.65
        
        if buy_confidence > THRESHOLD:
            return self._buy_signal(prices, buy_confidence, 
                                   entropy_val, flow_imbalance, hurst, complexity)
        elif sell_confidence > THRESHOLD:
            return self._sell_signal(prices, sell_confidence,
                                    entropy_val, flow_imbalance, hurst, complexity)
        else:
            return self._wait_signal(f"Insufficient confidence (Buy: {buy_confidence:.1%}, Sell: {sell_confidence:.1%})")
    
    def _calculate_atr_simple(self, prices, period=14):
        """Simple ATR for TP/SL"""
        changes = np.abs(np.diff(prices[-period:]))
        atr = np.mean(changes)
        return max(atr, prices[-1] * 0.001)  # At least 0.1%
    
    def _buy_signal(self, prices, confidence, entropy, flow, hurst, complexity):
        """Generate BUY signal"""
        current = prices[-1]
        atr = self._calculate_atr_simple(prices)
        
        entry = current * 1.0005
        tp = entry + (2.0 * atr)
        sl = entry - (1.0 * atr)
        
        reasoning = f"BUY: Entropy={entropy:.2f}, Flow={flow:+.2f}, Hurst={hurst:.2f}, Complexity={complexity:.2f}"
        
        return {
            'signal_type': 'BUY',
            'entry': float(entry),
            'tp': float(tp),
            'sl': float(sl),
            'confidence': float(confidence),
            'reasoning': reasoning,
            'novel_metrics': {
                'entropy': float(entropy),
                'order_flow': float(flow),
                'hurst': float(hurst),
                'complexity': float(complexity)
            }
        }
    
    def _sell_signal(self, prices, confidence, entropy, flow, hurst, complexity):
        """Generate SELL signal"""
        current = prices[-1]
        atr = self._calculate_atr_simple(prices)
        
        entry = current * 0.9995
        tp = entry - (2.0 * atr)
        sl = entry + (1.0 * atr)
        
        reasoning = f"SELL: Entropy={entropy:.2f}, Flow={flow:+.2f}, Hurst={hurst:.2f}, Complexity={complexity:.2f}"
        
        return {
            'signal_type': 'SELL',
            'entry': float(entry),
            'tp': float(tp),
            'sl': float(sl),
            'confidence': float(confidence),
            'reasoning': reasoning,
            'novel_metrics': {
                'entropy': float(entropy),
                'order_flow': float(flow),
                'hurst': float(hurst),
                'complexity': float(complexity)
            }
        }
    
    def _wait_signal(self, reason):
        """Generate WAIT signal"""
        return {
            'signal_type': 'WAIT',
            'entry': None,
            'tp': None,
            'sl': None,
            'confidence': 0.0,
            'reasoning': reason
        }


# Test
if __name__ == '__main__':
    print("\n🚀 TESTING NOVEL SIGNAL GENERATOR\n")
    
    # Generate test data
    np.random.seed(42)
    
    # Trending market
    trend = np.linspace(100, 110, 200)
    noise = np.random.normal(0, 0.5, 200)
    prices = trend + noise
    volumes = np.random.uniform(800, 1200, 200)
    
    # Test generator
    generator = NovelSignalGenerator()
    signal = generator.generate_signal(prices, volumes)
    
    print("\n" + "="*60)
    print("📊 FINAL SIGNAL")
    print("="*60)
    print(f"Type: {signal['signal_type']}")
    print(f"Entry: {signal.get('entry')}")
    print(f"TP: {signal.get('tp')}")
    print(f"SL: {signal.get('sl')}")
    print(f"Confidence: {signal.get('confidence', 0):.1%}")
    print(f"Reasoning: {signal.get('reasoning')}")
