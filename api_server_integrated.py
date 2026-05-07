"""
FIXED API SERVER - Integrated with Fixed Signal Generator
==========================================================
Drop-in replacement for api_server_integrated.py

Key changes:
1. Uses FixedSignalGenerator instead of original SignalGenerator
2. Updated model_manager to use new generator
3. Same Flask API endpoints, same MT5 posting logic
4. Better signal quality, fewer false positives

Environment variables:
- COMMUNITY_TRADING_URL: MT5 backend URL
- COMMUNITY_API_KEY: API key for MT5 backend
- DATA_FETCHER_URL: Data fetcher service URL
- NEWS_MODEL_URL: News model service URL
"""

import os
import requests
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sock import Sock
from dotenv import load_dotenv
import threading
import json
import time
from typing import Dict, Set, List, Optional
from datetime import datetime, timedelta
import logging
import concurrent.futures

# Import fixed signal generator
from fixed_signal_generator import FixedSignalGenerator, SupervisedRegimeDetector

load_dotenv()

# Flask setup
app = Flask(__name__)
sock = Sock(app)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')
CORS(app, resources={
    r"/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# External services
DATA_FETCHER_URL = os.getenv('DATA_FETCHER_URL', 'https://anso-vision-data-fetcher.onrender.com')
NEWS_MODEL_URL = os.getenv('NEWS_MODEL_URL', 'https://anso-vision-news-model.onrender.com')
COMMUNITY_TRADING_URL = os.getenv('COMMUNITY_TRADING_URL', 'https://ansorade-backend.onrender.com')
COMMUNITY_API_KEY = os.getenv('COMMUNITY_API_KEY', 'Mr.creative090')

logger.info(f"🎯 FIXED SYSTEM INITIALIZED")
logger.info(f"   Signal Generator: Fixed Confluence Engine v3")
logger.info(f"   MT5 Backend: {COMMUNITY_TRADING_URL}/api/signal")

# Thread pool for background posting
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)


# =============================================================================
# MODEL MANAGER (Fixed)
# =============================================================================
class ModelState:
    """Tracks state of trained model per symbol."""
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
        if not self.is_trained:
            return True
        if self.train_count >= candles_since_train:
            return True
        if self.last_trained and (datetime.now() - self.last_trained) > timedelta(minutes=max_age_minutes):
            return True
        return False


class ModelManager:
    """Manages FixedSignalGenerator instances per symbol."""

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
        """Initialize signal generator for symbol (no training needed for fixed system)."""
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
        """Generate signal using fixed confluence engine."""
        model_state = self.get_or_create_model(symbol)

        # Validate
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


# Singleton
_model_manager = None
_model_lock = threading.Lock()

def get_model_manager() -> ModelManager:
    global _model_manager
    if _model_manager is None:
        with _model_lock:
            if _model_manager is None:
                _model_manager = ModelManager()
    return _model_manager


# =============================================================================
# RISK MANAGER (Unchanged - still works well)
# =============================================================================
class RiskManager:
    """Prevents overtrading and dangerous signals."""

    def __init__(self, max_daily_signals: int = 20, 
                 max_signals_per_hour: int = 5,
                 min_signal_spacing_minutes: int = 15):
        self.max_daily_signals = max_daily_signals
        self.max_signals_per_hour = max_signals_per_hour
        self.min_signal_spacing_minutes = min_signal_spacing_minutes
        self.signal_history: Dict[str, list] = {}
        self.last_signal_time: Dict[str, datetime] = {}
        self.daily_signal_count: Dict[str, int] = {}
        self.last_reset_date: Dict[str, datetime] = {}
        self.lock = threading.Lock()

    def should_allow_signal(self, symbol: str, signal_type: str) -> tuple:
        with self.lock:
            now = datetime.now()

            if symbol not in self.signal_history:
                self.signal_history[symbol] = []
                self.daily_signal_count[symbol] = 0
                self.last_reset_date[symbol] = now.date()

            if self.last_reset_date[symbol] != now.date():
                self.daily_signal_count[symbol] = 0
                self.last_reset_date[symbol] = now.date()

            if self.daily_signal_count[symbol] >= self.max_daily_signals:
                return False, f"Daily limit ({self.max_daily_signals}) reached"

            hour_ago = now - timedelta(hours=1)
            recent_hour = [t for t in self.signal_history[symbol] if t > hour_ago]
            if len(recent_hour) >= self.max_signals_per_hour:
                return False, f"Hourly limit ({self.max_signals_per_hour}) reached"

            if symbol in self.last_signal_time:
                mins_since = (now - self.last_signal_time[symbol]).total_seconds() / 60
                if mins_since < self.min_signal_spacing_minutes:
                    return False, f"Too soon ({mins_since:.0f}min < {self.min_signal_spacing_minutes}min)"

            return True, "Signal approved"

    def record_signal(self, symbol: str, signal_type: str, confidence: float):
        with self.lock:
            now = datetime.now()
            if symbol not in self.signal_history:
                self.signal_history[symbol] = []
                self.daily_signal_count[symbol] = 0

            self.signal_history[symbol].append(now)
            self.last_signal_time[symbol] = now
            self.daily_signal_count[symbol] += 1


_risk_manager = None
_risk_lock = threading.Lock()

def get_risk_manager() -> RiskManager:
    global _risk_manager
    if _risk_manager is None:
        with _risk_lock:
            if _risk_manager is None:
                _risk_manager = RiskManager()
    return _risk_manager


# =============================================================================
# SIGNAL QUEUE & WEBSOCKET (Unchanged)
# =============================================================================
class SignalQueue:
    def __init__(self):
        self.signals: Dict[str, Dict] = {}
        self.lock = threading.Lock()
        self.last_posted: Dict[str, datetime] = {}

    def add_signal(self, symbol: str, signal: Dict):
        with self.lock:
            signal_id = f"{symbol}_{datetime.now().timestamp()}"
            self.signals[signal_id] = {**signal, 'symbol': symbol, 
                                       'timestamp': datetime.now().isoformat(), 'status': 'pending'}
            return signal_id

    def mark_posted(self, signal_id: str):
        with self.lock:
            if signal_id in self.signals:
                self.signals[signal_id]['status'] = 'posted'
                self.signals[signal_id]['posted_at'] = datetime.now().isoformat()
                self.last_posted[self.signals[signal_id]['symbol']] = datetime.now()

    def should_post_signal(self, symbol: str, signal_type: str) -> bool:
        with self.lock:
            if symbol not in self.last_posted:
                return True
            time_since = datetime.now() - self.last_posted[symbol]
            if time_since < timedelta(minutes=5):
                return False
            return True

signal_queue = SignalQueue()


class WebSocketManager:
    def __init__(self):
        self.connections: Dict[str, Set] = {}
        self.last_signals: Dict[str, Dict] = {}
        self.lock = threading.Lock()

    def subscribe(self, ws, symbol: str):
        with self.lock:
            if symbol not in self.connections:
                self.connections[symbol] = set()
            self.connections[symbol].add(ws)

    def broadcast_signal(self, symbol: str, signal: Dict):
        with self.lock:
            self.last_signals[symbol] = signal
            if symbol not in self.connections:
                return
            message = {"type": "signal_update", "symbol": symbol, "signal": signal,
                      "timestamp": datetime.now().isoformat()}
            disconnected = set()
            for ws in self.connections[symbol]:
                try:
                    ws.send(json.dumps(message))
                except:
                    disconnected.add(ws)
            for ws in disconnected:
                self.connections[symbol].discard(ws)

ws_manager = WebSocketManager()


# =============================================================================
# API ROUTES
# =============================================================================
model_manager = get_model_manager()
risk_manager = get_risk_manager()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'Anso Vision Backend (FIXED)',
        'version': '4.0.0',
        'signal_generator': 'Fixed Confluence Engine v3',
        'mt5_backend': COMMUNITY_TRADING_URL
    }), 200

@app.route('/api/candle-complete', methods=['POST'])
def candle_complete():
    """Called when new candle completes - auto-posts signals."""
    try:
        data = request.json
        symbol = data.get('symbol')
        candles = data.get('candles')

        if not symbol or not candles:
            return jsonify({'error': 'Missing symbol or candles'}), 400

        prices = np.array([c.get('close', 0) for c in candles])
        volumes = np.array([c.get('volume', 1.0) for c in candles])

        if len(prices) < 250:
            return jsonify({'status': 'insufficient_data', 'required': 250, 'received': len(prices)}), 200

        # Check news
        trade_allowed, news_reason = check_news_before_trade()
        if not trade_allowed:
            return jsonify({'status': 'news_block', 'reason': news_reason}), 200

        # Generate signal
        signal_result = model_manager.generate_signal(symbol, prices, volumes)

        if signal_result.get('signal_type') != 'WAIT':
            risk_manager.record_signal(symbol, signal_result['signal_type'], 
                                      signal_result.get('confidence', 0.0))

            if signal_queue.should_post_signal(symbol, signal_result['signal_type']):
                signal_id = signal_queue.add_signal(symbol, signal_result)
                thread_pool.submit(post_to_mt5, symbol, signal_result, signal_id)

        ws_manager.broadcast_signal(symbol, signal_result)

        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'signal_type': signal_result.get('signal_type'),
            'posted_to_mt5': signal_result.get('signal_type') != 'WAIT'
        }), 200

    except Exception as e:
        logger.error(f"Error in candle_complete: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_signal():
    """Main endpoint for frontend signal analysis."""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        data = request.json
        symbol = data.get('symbol')
        candles = data.get('candles', [])

        if not symbol or len(candles) < 250:
            return jsonify({'success': False, 'error': 'Need 250 candles', 'signal': 'HOLD'}), 400

        prices = np.array([c.get('close', 0) for c in candles])
        volumes = np.array([c.get('volume', 0) for c in candles])

        trade_allowed, news_reason = check_news_before_trade()
        if not trade_allowed:
            return build_wait_response(symbol, prices[-1], f'News block: {news_reason}')

        allowed, risk_reason = risk_manager.should_allow_signal(symbol, 'UNKNOWN')
        if not allowed:
            return build_wait_response(symbol, prices[-1], f'Risk limit: {risk_reason}')

        signal_result = model_manager.generate_signal(symbol, prices, volumes)

        if signal_result is None or signal_result.get('error', False):
            error_msg = signal_result.get('error_message', 'Unknown error') if signal_result else 'Signal generation failed'
            return build_wait_response(symbol, prices[-1], error_msg)

        signal_type = signal_result.get('signal_type', 'WAIT')

        if signal_type != 'WAIT':
            risk_manager.record_signal(symbol, signal_type, signal_result.get('confidence', 0.0))

            if signal_queue.should_post_signal(symbol, signal_type):
                signal_id = signal_queue.add_signal(symbol, signal_result)
                thread_pool.submit(post_to_mt5, symbol, signal_result, signal_id)

        response = build_signal_response(symbol, signal_result, prices[-1])
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in /analyze: {e}")
        return jsonify({'success': False, 'error': str(e), 'signal': 'HOLD'}), 500


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def check_news_before_trade() -> tuple:
    """Check if high-impact news allows trading."""
    try:
        response = requests.get(f"{NEWS_MODEL_URL}/should-trade", timeout=3)
        if response.status_code == 200:
            news_data = response.json()
            if not news_data.get('should_trade', True):
                return False, news_data.get('reason', 'High-impact news')
            return True, 'News check passed'
        return True, 'News API failure, proceeding'
    except:
        return True, 'News check unreachable, proceeding'

def post_to_mt5(symbol: str, signal: Dict, signal_id: Optional[str] = None):
    """Post signal to MT5 backend."""
    try:
        signal_type = signal.get('signal_type', 'WAIT')
        if signal_type == 'WAIT':
            return

        payload = {
            "symbol": symbol,
            "action": signal_type,
            "entry": float(signal.get('entry', 0)),
            "tp": float(signal.get('tp', 0)),
            "sl": float(signal.get('sl', 0)),
            "volume": 0.01,
            "confidence": float(signal.get('confidence', 0)),
            "reasoning": signal.get('reasoning', 'No reasoning'),
            "regime": signal.get('regime', 'UNKNOWN'),
            "timeframe": "1h"
        }

        response = requests.post(
            f"{COMMUNITY_TRADING_URL}/api/signal",
            json=payload,
            headers={"X-API-Key": COMMUNITY_API_KEY, "Content-Type": "application/json"},
            timeout=10
        )

        if response.status_code == 200:
            if signal_id:
                signal_queue.mark_posted(signal_id)
            logger.info(f"✅ Signal posted to MT5: {symbol} {signal_type}")
        else:
            logger.error(f"❌ MT5 error: {response.status_code}")

    except Exception as e:
        logger.error(f"❌ Error posting to MT5: {e}")

def build_wait_response(symbol: str, price: float, reason: str):
    return jsonify({
        'success': True, 'symbol': symbol, 'signal': 'HOLD', 'signal_type': 'WAIT',
        'entry': float(price), 'tp': 0, 'sl': 0, 'confidence': 0,
        'reasoning': reason, 'regime': 'N/A'
    }), 200

def build_signal_response(symbol: str, signal_result: Dict, current_price: float):
    signal_type = signal_result.get('signal_type', 'WAIT')
    return {
        'success': True, 'symbol': symbol,
        'signal': 'BUY' if signal_type == 'BUY' else 'SELL' if signal_type == 'SELL' else 'HOLD',
        'signal_type': signal_type,
        'entry': float(signal_result.get('entry', current_price)),
        'tp': float(signal_result.get('tp', 0)),
        'sl': float(signal_result.get('sl', 0)),
        'confidence': float(signal_result.get('confidence', 0)),
        'reasoning': signal_result.get('reasoning', ''),
        'regime': signal_result.get('regime', 'N/A'),
        'posted_to_mt5': signal_type != 'WAIT'
    }


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    print(f"\n{'='*70}")
    print(f"🚀 STARTING FIXED API SERVER v4.0.0")
    print(f"{'='*70}")
    print(f"   • Signal Generator: Fixed Confluence Engine v3")
    print(f"   • 2/5 Confluence Requirement")
    print(f"   • Supervised Regime Detection (replaces HMM)")
    print(f"   • Proper ATR + Mean Reversion + Breakout Detection")
    print(f"{'='*70}")

    app.run(host='0.0.0.0', port=port, debug=debug)
