"""
TEST: Ultra-Clean Signal Generator
Verify the new system works correctly
"""
import numpy as np
from signal_generator import SignalGenerator

# Generate realistic test data
np.random.seed(42)
base_price = 1.0850
prices = base_price + np.cumsum(np.random.randn(300) * 0.0001)
volumes = np.abs(np.random.randn(300) * 1000 + 5000)

print("\n" + "="*70)
print("TESTING ULTRA-CLEAN SIGNAL GENERATOR")
print("="*70)
print(f"Test Data: {len(prices)} candles")
print(f"Price Range: {prices.min():.5f} - {prices.max():.5f}")
print(f"Current Price: {prices[-1]:.5f}")

# Initialize generator
sig_gen = SignalGenerator()

# Generate signal
signal = sig_gen.generate_signals(prices, volumes)

# Display results
print("\n" + "="*70)
print("SIGNAL OUTPUT")
print("="*70)
print(f"Signal: {signal['signal']}")
print(f"Entry: {signal['entry']:.5f}")
print(f"TP: {signal['tp']:.5f}")
print(f"SL: {signal['sl']:.5f}")
print(f"Confidence: {signal['confidence']:.1%}")
print(f"\nHMM State: {signal['hmm_state']}")
print(f"HMM Confidence: {signal['hmm_confidence']:.1%}")
print(f"Stop Hunt Risk: {signal['stop_hunt_risk']}")
print(f"Whipsaw Risk: {signal['whipsaw_risk']}")
print(f"Liquidity Adjusted: {signal['liquidity_adjusted']}")
print(f"\nRisk Metrics:")
print(f"  R:R: {signal['risk_metrics']['risk_reward_ratio']:.2f}:1")
print(f"  Potential Profit: {signal['risk_metrics']['potential_profit_pct']:.2f}%")
print(f"  Potential Loss: {signal['risk_metrics']['potential_loss_pct']:.2f}%")
print(f"  Volatility: {signal['risk_metrics']['volatility']:.6f}")
print(f"\nReasoning: {signal['reasoning']}")
print("="*70)
