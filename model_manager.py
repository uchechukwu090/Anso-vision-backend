"""
Model Manager - Persistent, Pre-trained Models
Prevents retraining HMM on every API call

✅ BULLETPROOF VERSION with comprehensive error handling
"""
import threading
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
from signal_generator import SignalGenerator
from hmm_model import MarketHMM


class ModelState:
    """Tracks the state of a trained model"""
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.signal_generator = None
        self.last_trained = None
        self.train_count = 0
        self.last_signal = None
        self.last_signal_time = None
        self.lock = threading.Lock()
        self.is_trained = False
        self.last_error = None
        
    def needs_retraining(self, candles_since_train: int = 50, max_age_minutes: int = 60) -> bool:
        """Check if model needs retraining"""
        if not self.is_trained:
            return True
        
        # Retrain every 50 candles
        if self.train_count >= candles_since_train:
            return True
        
        # Retrain if model is too old (stale)
        if self.last_trained:
            age = datetime.now() - self.last_trained
            if age > timedelta(minutes=max_age_minutes):
                return True
        
        return False


class ModelManager:
    """
    Manages trained models for multiple symbols
    Handles training, caching, and signal generation
    """
    def __init__(self, n_hmm_components: int = 3, 
                 covariance_type: str = 'diag',
                 random_state: int = 42):
        self.models: Dict[str, ModelState] = {}
        self.n_hmm_components = n_hmm_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.global_lock = threading.Lock()
        
        # Performance tracking
        self.signal_history = []
        self.performance_metrics = {}
        
        print(f"✅ ModelManager initialized (HMM components={n_hmm_components})")
    
    def get_or_create_model(self, symbol: str) -> ModelState:
        """Get existing model or create new one"""
        with self.global_lock:
            if symbol not in self.models:
                self.models[symbol] = ModelState(symbol)
                print(f"📊 Created new model state for {symbol}")
            return self.models[symbol]
    
    def train_model(self, symbol: str, prices: np.ndarray, volumes: np.ndarray = None, force: bool = False) -> tuple[bool, str]:
        """
        Train or retrain model for a symbol
        
        Args:
            symbol: Trading symbol
            prices: Price array (minimum 100 candles)
            volumes: Volume array (optional)
            force: Force retraining even if not needed
            
        Returns:
            tuple[bool, str]: (success, error_message)
        """
        print(f"\n{'='*70}")
        print(f"🎯 TRAINING MODEL: {symbol}")
        print(f"{'='*70}")
        print(f"   Prices: {len(prices)} candles (range: {prices.min():.2f} - {prices.max():.2f})")
        print(f"   Volumes: {len(volumes) if volumes is not None else 'None'}")
        print(f"   Force: {force}")
        
        model_state = self.get_or_create_model(symbol)
        
        with model_state.lock:
            # Check if training is needed
            if not force and not model_state.needs_retraining():
                print(f"   ℹ️ Model already trained and fresh")
                return True, "Model already trained"
            
            try:
                # Validate data - REQUIRE 250 CANDLES FOR STABLE HMM
                if len(prices) < 250:
                    error_msg = f"Insufficient data: Need 250 candles for stable HMM, got {len(prices)}"
                    print(f"   ❌ {error_msg}")
                    model_state.last_error = error_msg
                    return False, error_msg
                
                # Check for invalid prices
                if np.any(prices <= 0) or np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
                    error_msg = "Invalid price data (zero, negative, NaN or Inf values)"
                    print(f"   ❌ {error_msg}")
                    model_state.last_error = error_msg
                    return False, error_msg
                
                # Use last 250 candles for training (or all if less)
                train_data = prices[-250:] if len(prices) > 250 else prices
                print(f"   📊 Using {len(train_data)} candles for training")
                
                # Initialize signal generator if needed
                if model_state.signal_generator is None:
                    print(f"   🆕 Creating new SignalGenerator...")
                    try:
                        model_state.signal_generator = SignalGenerator(
                            n_hmm_components=self.n_hmm_components,
                            random_state=self.random_state
                        )
                        print(f"   ✅ SignalGenerator created successfully")
                    except Exception as sg_error:
                        error_msg = f"SignalGenerator creation failed: {str(sg_error)}"
                        print(f"   ❌ {error_msg}")
                        import traceback
                        traceback.print_exc()
                        model_state.last_error = error_msg
                        return False, error_msg
                
                # Train HMM
                print(f"   🧠 Preparing HMM features...")
                try:
                    hmm_features = model_state.signal_generator._prepare_hmm_features(train_data)
                    print(f"   ✅ Features prepared: shape {hmm_features.shape}")
                except Exception as feat_error:
                    error_msg = f"Feature preparation failed: {str(feat_error)}"
                    print(f"   ❌ {error_msg}")
                    import traceback
                    traceback.print_exc()
                    model_state.last_error = error_msg
                    return False, error_msg
                
                if len(hmm_features) < self.n_hmm_components:
                    error_msg = f"Insufficient features for HMM: {len(hmm_features)} < {self.n_hmm_components}"
                    print(f"   ❌ {error_msg}")
                    model_state.last_error = error_msg
                    return False, error_msg
                
                print(f"   🎯 Training HMM model...")
                try:
                    model_state.signal_generator.hmm_model.train(hmm_features)
                    print(f"   ✅ HMM training complete")
                except Exception as train_error:
                    error_msg = f"HMM training failed: {str(train_error)}"
                    print(f"   ❌ {error_msg}")
                    import traceback
                    traceback.print_exc()
                    model_state.last_error = error_msg
                    return False, error_msg
                
                # Update state
                model_state.is_trained = True
                model_state.last_trained = datetime.now()
                model_state.train_count = 0
                model_state.last_error = None
                
                print(f"   ✅ {symbol}: Model trained successfully")
                print(f"{'='*70}\n")
                return True, "Training successful"
                
            except Exception as e:
                error_msg = f"Unexpected training error: {type(e).__name__}: {str(e)}"
                print(f"   ❌ {error_msg}")
                import traceback
                traceback.print_exc()
                model_state.last_error = error_msg
                print(f"{'='*70}\n")
                return False, error_msg
    
    def generate_signal(self, symbol: str, prices: np.ndarray, 
                       volumes: np.ndarray = None,
                       auto_train: bool = True) -> dict:
        """
        Generate trading signal for a symbol
        
        Args:
            symbol: Trading symbol
            prices: Price array
            volumes: Volume array (optional)
            auto_train: Automatically train if needed
            
        Returns:
            dict: ALWAYS returns a dict, never None. Returns WAIT signal if failed.
        """
        print(f"\n{'='*70}")
        print(f"🔮 GENERATING SIGNAL: {symbol}")
        print(f"{'='*70}")
        print(f"   Prices: {len(prices)} candles")
        print(f"   Auto-train: {auto_train}")
        
        model_state = self.get_or_create_model(symbol)
        
        # Validate input data
        if len(prices) < 200:
            error_msg = f"Insufficient data: Need 200 candles, got {len(prices)}"
            print(f"   ❌ {error_msg}")
            print(f"{'='*70}\n")
            return self._error_signal(error_msg)
        
        if np.any(prices <= 0) or np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
            error_msg = "Invalid price data (zero, negative, NaN or Inf values)"
            print(f"   ❌ {error_msg}")
            print(f"{'='*70}\n")
            return self._error_signal(error_msg)
        
        # Train if needed
        if auto_train and model_state.needs_retraining():
            print(f"   🔄 Model needs retraining...")
            success, train_error = self.train_model(symbol, prices, volumes)
            if not success:
                error_msg = f"Training failed: {train_error}"
                print(f"   ❌ {error_msg}")
                print(f"{'='*70}\n")
                return self._error_signal(error_msg)
        
        # Check if model is trained
        if not model_state.is_trained:
            error_msg = f"Model not trained and auto-train disabled"
            print(f"   ❌ {error_msg}")
            if model_state.last_error:
                error_msg += f" (last error: {model_state.last_error})"
            print(f"{'='*70}\n")
            return self._error_signal(error_msg)
        
        # Generate signal with comprehensive error handling
        try:
            with model_state.lock:
                print(f"   📊 Calling signal_generator.generate_signals()...")
                
                # Use last 250 candles (or all if less)
                signal_data = prices[-250:] if len(prices) > 250 else prices
                signal_volumes = volumes[-250:] if volumes is not None and len(volumes) > 250 else volumes
                
                try:
                    signal = model_state.signal_generator.generate_signals(signal_data, signal_volumes)
                except Exception as gen_error:
                    error_msg = f"Signal generation failed: {type(gen_error).__name__}: {str(gen_error)}"
                    print(f"   ❌ {error_msg}")
                    import traceback
                    traceback.print_exc()
                    print(f"{'='*70}\n")
                    return self._error_signal(error_msg)
                
                if signal is None:
                    error_msg = "Signal generator returned None (internal error)"
                    print(f"   ❌ {error_msg}")
                    print(f"{'='*70}\n")
                    return self._error_signal(error_msg)
                
                # Validate signal structure
                if not isinstance(signal, dict):
                    error_msg = f"Invalid signal type: expected dict, got {type(signal)}"
                    print(f"   ❌ {error_msg}")
                    print(f"{'='*70}\n")
                    return self._error_signal(error_msg)
                
                # Increment train counter
                model_state.train_count += 1
                
                # Track signal
                model_state.last_signal = signal
                model_state.last_signal_time = datetime.now()
                
                # Add metadata
                signal['symbol'] = symbol
                signal['timestamp'] = datetime.now().isoformat()
                signal['model_age_minutes'] = (datetime.now() - model_state.last_trained).total_seconds() / 60 if model_state.last_trained else 0
                signal['candles_since_train'] = model_state.train_count
                
                # Ensure all required fields exist
                signal.setdefault('signal_type', 'WAIT')
                signal.setdefault('entry', float(prices[-1]))
                signal.setdefault('tp', 0.0)
                signal.setdefault('sl', 0.0)
                signal.setdefault('confidence', 0.0)
                signal.setdefault('reasoning', 'No reasoning provided')
                signal.setdefault('market_context', 'N/A')
                signal.setdefault('risk_metrics', {})
                
                # Store in history
                self.signal_history.append({
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'signal': signal.get('signal_type'),
                    'confidence': signal.get('confidence', 0),
                    'entry': signal.get('entry'),
                    'tp': signal.get('tp'),
                    'sl': signal.get('sl')
                })
                
                # Limit history size
                if len(self.signal_history) > 1000:
                    self.signal_history = self.signal_history[-1000:]
                
                print(f"   ✅ Signal generated: {signal['signal_type']}")
                print(f"{'='*70}\n")
                return signal
                
        except Exception as e:
            error_msg = f"Unexpected signal generation error: {type(e).__name__}: {str(e)}"
            print(f"   ❌ {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"{'='*70}\n")
            return self._error_signal(error_msg)
    
    def _error_signal(self, error_message: str) -> dict:
        """
        Return a safe WAIT signal with error information
        NEVER returns None - always returns a valid signal dict
        """
        return {
            'signal_type': 'WAIT',
            'entry': 0.0,
            'tp': 0.0,
            'sl': 0.0,
            'confidence': 0.0,
            'reasoning': f'ERROR: {error_message}',
            'market_context': 'Error',
            'risk_metrics': {
                'risk_reward_ratio': 0,
                'potential_profit_pct': 0,
                'potential_loss_pct': 0,
                'prob_tp_hit': 0,
                'prob_sl_hit': 0,
                'expected_value': 0,
                'expected_value_pct': 0
            },
            'error': True,
            'error_message': error_message
        }
    
    def get_model_state(self, symbol: str) -> Optional[ModelState]:
        """Get model state object for a symbol"""
        return self.models.get(symbol)
    
    def get_model_stats(self, symbol: str) -> dict:
        """Get statistics about a model"""
        model_state = self.models.get(symbol)
        if not model_state:
            return {'error': 'Model not found'}
        
        with model_state.lock:
            return {
                'symbol': symbol,
                'is_trained': model_state.is_trained,
                'last_trained': model_state.last_trained.isoformat() if model_state.last_trained else None,
                'candles_since_train': model_state.train_count,
                'last_signal_type': model_state.last_signal.get('signal_type') if model_state.last_signal else None,
                'last_signal_time': model_state.last_signal_time.isoformat() if model_state.last_signal_time else None,
                'needs_retraining': model_state.needs_retraining(),
                'last_error': model_state.last_error
            }
    
    def get_all_stats(self) -> dict:
        """Get statistics for all models"""
        return {
            'total_models': len(self.models),
            'total_signals': len(self.signal_history),
            'models': {symbol: self.get_model_stats(symbol) for symbol in self.models.keys()}
        }
    
    def force_retrain_all(self, prices_dict: Dict[str, np.ndarray]):
        """Force retrain all models"""
        print("🔄 Force retraining all models...")
        for symbol, prices in prices_dict.items():
            self.train_model(symbol, prices, force=True)
        print("✅ All models retrained")
    
    def clear_model(self, symbol: str):
        """Clear a specific model"""
        with self.global_lock:
            if symbol in self.models:
                del self.models[symbol]
                print(f"🗑️ Cleared model for {symbol}")
    
    def clear_all_models(self):
        """Clear all models"""
        with self.global_lock:
            self.models.clear()
            self.signal_history.clear()
            print("🗑️ Cleared all models")


# Singleton instance
_model_manager_instance = None
_model_manager_lock = threading.Lock()

def get_model_manager() -> ModelManager:
    """Get singleton ModelManager instance"""
    global _model_manager_instance
    
    if _model_manager_instance is None:
        with _model_manager_lock:
            if _model_manager_instance is None:
                _model_manager_instance = ModelManager()
    
    return _model_manager_instance


if __name__ == '__main__':
    # Test the model manager
    manager = get_model_manager()
    
    # Generate test data
    np.random.seed(42)
    test_prices = np.cumsum(np.random.normal(0.001, 0.02, 300)) + 100
    
    # Train and generate signal
    print("\n--- Training Model ---")
    success, error = manager.train_model("BTCUSD", test_prices)
    print(f"Training success: {success}")
    if not success:
        print(f"Error: {error}")
    
    # Generate signal
    print("\n--- Generating Signal ---")
    signal = manager.generate_signal("BTCUSD", test_prices)
    print(f"Signal: {signal.get('signal_type')}")
    print(f"Entry: {signal.get('entry')}")
    print(f"TP: {signal.get('tp')}")
    print(f"SL: {signal.get('sl')}")
    if signal.get('error'):
        print(f"Error: {signal.get('error_message')}")
    
    # Check stats
    print("\n--- Model Stats ---")
    stats = manager.get_model_stats("BTCUSD")
    print(stats)
