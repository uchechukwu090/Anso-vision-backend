"""
FIXED MODEL MANAGER
===================
Simplified manager for FixedSignalGenerator.
No HMM training needed - just initializes the confluence engine.
"""

import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
from fixed_signal_generator import FixedSignalGenerator


class ModelState:
    """Tracks state of signal generator per symbol."""
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.signal_generator: Optional[FixedSignalGenerator] = None
        self.last_trained = None
        self.train_count = 0
        self.last_signal = None
        self.last_signal_time = None
        self.lock = threading.Lock()
        self.is_trained = False
        self.last_error = None

    def needs_retraining(self, candles_since_train: int = 50, max_age_minutes: int = 60) -> bool:
        if not self.is_trained:
            return True
        if self.train_count >= candles_since_train:
            return True
        if self.last_trained and (datetime.now() - self.last_trained) > timedelta(minutes=max_age_minutes):
            return True
        return False


class ModelManager:
    """Manages FixedSignalGenerator instances for multiple symbols."""

    def __init__(self, confidence_threshold: float = 0.50):
        self.models: Dict[str, ModelState] = {}
        self.confidence_threshold = confidence_threshold
        self.global_lock = threading.Lock()
        self.signal_history = []
        print(f"✅ Fixed ModelManager initialized (threshold={confidence_threshold})")

    def get_or_create_model(self, symbol: str) -> ModelState:
        with self.global_lock:
            if symbol not in self.models:
                self.models[symbol] = ModelState(symbol)
                print(f"📊 Created model state for {symbol}")
            return self.models[symbol]

    def train_model(self, symbol: str, prices: np.ndarray, 
                   volumes: np.ndarray = None, force: bool = False) -> tuple:
        """
        Initialize signal generator for symbol.
        No actual training needed - FixedSignalGenerator uses rule-based logic.
        """
        model_state = self.get_or_create_model(symbol)

        with model_state.lock:
            if not force and model_state.is_trained:
                return True, "Model already initialized"

            if len(prices) < 250:
                error_msg = f"Need 250 candles, got {len(prices)}"
                model_state.last_error = error_msg
                return False, error_msg

            try:
                model_state.signal_generator = FixedSignalGenerator(
                    confidence_threshold=self.confidence_threshold
                )
                model_state.is_trained = True
                model_state.last_trained = datetime.now()
                model_state.train_count = 0
                model_state.last_error = None
                return True, "Initialized successfully"
            except Exception as e:
                error_msg = f"Initialization failed: {str(e)}"
                model_state.last_error = error_msg
                return False, error_msg

    def generate_signal(self, symbol: str, prices: np.ndarray,
                       volumes: np.ndarray = None, auto_train: bool = True) -> dict:
        """Generate trading signal using fixed confluence engine."""
        model_state = self.get_or_create_model(symbol)

        # Validate input
        if len(prices) < 250:
            return self._error_signal(f"Need 250 candles, got {len(prices)}")

        if np.any(prices <= 0) or np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
            return self._error_signal("Invalid price data")

        # Initialize if needed
        if auto_train and not model_state.is_trained:
            success, error = self.train_model(symbol, prices, volumes)
            if not success:
                return self._error_signal(f"Init failed: {error}")

        if not model_state.is_trained:
            return self._error_signal("Model not initialized")

        try:
            with model_state.lock:
                signal = model_state.signal_generator.generate_signals(
                    prices[-250:] if len(prices) > 250 else prices,
                    volumes[-250:] if volumes is not None and len(volumes) > 250 else volumes
                )

                if signal is None or not isinstance(signal, dict):
                    return self._error_signal("Invalid signal output")

                model_state.train_count += 1
                model_state.last_signal = signal
                model_state.last_signal_time = datetime.now()

                # Add metadata
                signal['symbol'] = symbol
                signal['timestamp'] = datetime.now().isoformat()
                signal['model_age_minutes'] = (datetime.now() - model_state.last_trained).total_seconds() / 60 if model_state.last_trained else 0
                signal['candles_since_train'] = model_state.train_count

                # Ensure required fields
                signal.setdefault('signal_type', 'WAIT')
                signal.setdefault('entry', float(prices[-1]))
                signal.setdefault('tp', 0.0)
                signal.setdefault('sl', 0.0)
                signal.setdefault('confidence', 0.0)
                signal.setdefault('reasoning', 'No reasoning')
                signal.setdefault('regime', 'UNKNOWN')
                signal.setdefault('risk_metrics', {})

                return signal

        except Exception as e:
            return self._error_signal(f"Signal error: {type(e).__name__}: {str(e)}")

    def get_model_stats(self, symbol: str) -> dict:
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
                'needs_retraining': model_state.needs_retraining(),
                'last_error': model_state.last_error
            }

    def _error_signal(self, error_message: str) -> dict:
        return {
            'signal_type': 'WAIT',
            'entry': 0.0, 'tp': 0.0, 'sl': 0.0,
            'confidence': 0.0,
            'reasoning': f'ERROR: {error_message}',
            'regime': 'UNKNOWN',
            'risk_metrics': {},
            'error': True,
            'error_message': error_message
        }


# Singleton instance
_model_manager_instance = None
_model_manager_lock = threading.Lock()

def get_model_manager() -> ModelManager:
    global _model_manager_instance
    if _model_manager_instance is None:
        with _model_manager_lock:
            if _model_manager_instance is None:
                _model_manager_instance = ModelManager()
    return _model_manager_instance


if __name__ == '__main__':
    # Test
    manager = get_model_manager()
    np.random.seed(42)
    test_prices = np.cumsum(np.random.normal(0.001, 0.02, 300)) + 100

    success, msg = manager.train_model("BTCUSD", test_prices)
    print(f"Training: {success} - {msg}")

    signal = manager.generate_signal("BTCUSD", test_prices)
    print(f"Signal: {signal.get('signal_type')}")
    print(f"Confidence: {signal.get('confidence', 0):.1%}")
