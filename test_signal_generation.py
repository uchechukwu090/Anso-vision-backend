"""
Test script to identify the exact error in signal generation
Run this locally to see what's failing
"""
import numpy as np
import sys
import traceback

print("="*70)
print("TESTING SIGNAL GENERATION LOCALLY")
print("="*70)

# Test 1: Import all modules
print("\n1️⃣ Testing imports...")
try:
    from model_manager import get_model_manager
    from signal_generator import SignalGenerator
    from hmm_model import MarketHMM
    print("   ✅ All imports successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 2: Create SignalGenerator with covariance_type
print("\n2️⃣ Testing SignalGenerator creation...")
try:
    sg = SignalGenerator(
        n_hmm_components=3,
        covariance_type='diag',
        random_state=42
    )
    print("   ✅ SignalGenerator created successfully")
except Exception as e:
    print(f"   ❌ SignalGenerator creation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 3: Generate test data
print("\n3️⃣ Generating test data...")
np.random.seed(42)
prices = np.cumsum(np.random.normal(0.001, 0.02, 150)) + 100
volumes = np.random.uniform(1000, 10000, 150)
print(f"   ✅ Generated {len(prices)} candles")
print(f"   Price range: {prices.min():.2f} - {prices.max():.2f}")

# Test 4: Create ModelManager
print("\n4️⃣ Testing ModelManager...")
try:
    manager = get_model_manager()
    print("   ✅ ModelManager created")
except Exception as e:
    print(f"   ❌ ModelManager creation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 5: Train model
print("\n5️⃣ Testing model training...")
try:
    success = manager.train_model("TESTBTC", prices, volumes)
    print(f"   Training result: {success}")
    if not success:
        print("   ❌ Training returned False")
        sys.exit(1)
    print("   ✅ Training successful")
except Exception as e:
    print(f"   ❌ Training failed with exception: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 6: Generate signal
print("\n6️⃣ Testing signal generation...")
try:
    signal = manager.generate_signal("TESTBTC", prices, volumes)
    
    if signal is None:
        print("   ❌ Signal is None!")
        
        # Check model state
        state = manager.get_model_state("TESTBTC")
        if state:
            print(f"   Model is_trained: {state.is_trained}")
            print(f"   Signal generator exists: {state.signal_generator is not None}")
        sys.exit(1)
    
    print("   ✅ Signal generated successfully!")
    print(f"   Signal type: {signal.get('signal_type')}")
    print(f"   Entry: {signal.get('entry')}")
    print(f"   TP: {signal.get('tp')}")
    print(f"   SL: {signal.get('sl')}")
    print(f"   Confidence: {signal.get('confidence')}")
    
except Exception as e:
    print(f"   ❌ Signal generation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 7: Direct SignalGenerator test
print("\n7️⃣ Testing SignalGenerator directly...")
try:
    sg2 = SignalGenerator(n_hmm_components=3, covariance_type='diag', random_state=42)
    direct_signal = sg2.generate_signals(prices, volumes)
    
    print(f"   ✅ Direct signal: {direct_signal.get('signal_type')}")
    
except Exception as e:
    print(f"   ❌ Direct signal generation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nIf this works locally but fails on Render, check:")
print("1. Render logs for Python errors")
print("2. Missing dependencies in requirements.txt")
print("3. Environment variables")
print("4. Memory/CPU limits on Render")
