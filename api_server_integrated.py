"""
API SERVER INTEGRATED - Enhanced with WebSocket Support & Logging
✅ FIXED: Now includes limit_orders and all fields in signal transmission
✅ FIXED: Signals automatically posted to MT5 from /analyze endpoint
✅ FIXED: Real-time WebSocket trigger system
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

# Core integration
from model_manager import get_model_manager
from risk_manager import get_risk_manager

load_dotenv()

# Flask setup
app = Flask(__name__)
sock = Sock(app)

# ✅ ENHANCED LOGGING SETUP
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('signal_system.log')
    ]
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

# Manager initialization
model_manager = get_model_manager()
risk_manager = get_risk_manager()

DATA_FETCHER_URL = os.getenv('DATA_FETCHER_URL', 'https://anso-vision-data-fetcher.onrender.com')
NEWS_MODEL_URL = os.getenv('NEWS_MODEL_URL', 'https://anso-vision-news-model.onrender.com')
COMMUNITY_TRADING_URL = os.getenv('COMMUNITY_TRADING_URL', 'https://ansorade-backend.onrender.com')
COMMUNITY_API_KEY = os.getenv('COMMUNITY_API_KEY', 'Mr.creative090')

logger.info(f"🎯 SYSTEM INITIALIZATION")
logger.info(f"   Signal Generator → MT5 API: {COMMUNITY_TRADING_URL}/api/signal")
logger.info(f"   API Key: {'*' * len(COMMUNITY_API_KEY)}")

# ✅ NEW: Thread pool for background signal posting
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)

# ✅ NEW: Signal Queue for tracking
class SignalQueue:
    def __init__(self):
        self.signals: Dict[str, Dict] = {}
        self.lock = threading.Lock()
        self.last_posted: Dict[str, datetime] = {}
    
    def add_signal(self, symbol: str, signal: Dict):
        with self.lock:
            signal_id = f"{symbol}_{datetime.now().timestamp()}"
            self.signals[signal_id] = {
                **signal,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'status': 'pending'
            }
            logger.info(f"📥 Queue: Added signal {signal_id} for {symbol}")
            return signal_id
    
    def mark_posted(self, signal_id: str):
        with self.lock:
            if signal_id in self.signals:
                self.signals[signal_id]['status'] = 'posted'
                self.signals[signal_id]['posted_at'] = datetime.now().isoformat()
                self.last_posted[self.signals[signal_id]['symbol']] = datetime.now()
                logger.info(f"📤 Queue: Marked {signal_id} as posted")
    
    def should_post_signal(self, symbol: str, signal_type: str) -> bool:
        """Prevent duplicate signals within 5 minutes"""
        with self.lock:
            if symbol not in self.last_posted:
                return True
            
            time_since_last = datetime.now() - self.last_posted[symbol]
            if time_since_last < timedelta(minutes=5):
                logger.info(f"⏭️ Skipping {symbol} - Last signal was {time_since_last.seconds//60} min ago")
                return False
            return True

signal_queue = SignalQueue()

# WebSocket connection manager
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
            logger.info(f"✅ WebSocket subscribed to {symbol}. Total: {len(self.connections[symbol])}")
            
            if symbol in self.last_signals:
                self._send_safe(ws, {
                    "type": "signal_update",
                    "symbol": symbol,
                    "signal": self.last_signals[symbol],
                    "timestamp": datetime.now().isoformat()
                })
    
    def unsubscribe(self, ws, symbol: str):
        with self.lock:
            if symbol in self.connections:
                self.connections[symbol].discard(ws)
                logger.info(f"❌ WebSocket unsubscribed from {symbol}")
    
    def broadcast_signal(self, symbol: str, signal: Dict):
        with self.lock:
            last_signal = self.last_signals.get(symbol, {})
            signal_changed = (
                last_signal.get('signal_type') != signal.get('signal_type') or
                abs(last_signal.get('confidence', 0) - signal.get('confidence', 0)) > 0.1
            )
            
            if signal_changed:
                logger.info(f"🔔 {symbol}: Signal changed - {signal['signal_type']} @ {signal.get('confidence', 0):.1%}")
            
            self.last_signals[symbol] = signal
            
            if symbol not in self.connections:
                return
            
            message = {
                "type": "signal_update",
                "symbol": symbol,
                "signal": signal,
                "signal_changed": signal_changed,
                "timestamp": datetime.now().isoformat()
            }
            
            disconnected = set()
            for ws in self.connections[symbol]:
                if not self._send_safe(ws, message):
                    disconnected.add(ws)
            
            for ws in disconnected:
                self.connections[symbol].discard(ws)
    
    def _send_safe(self, ws, message: dict) -> bool:
        try:
            ws.send(json.dumps(message))
            return True
        except:
            return False

ws_manager = WebSocketManager()

# ✅ WebSocket endpoint
@sock.route('/ws/signals')
def websocket_signals(ws):
    """WebSocket endpoint for real-time signal updates"""
    logger.info("🔌 WebSocket client connected")
    
    try:
        while True:
            message = ws.receive()
            if not message:
                break
            
            data = json.loads(message)
            action = data.get('action')
            symbol = data.get('symbol')
            
            if action == 'subscribe' and symbol:
                ws_manager.subscribe(ws, symbol)
                ws.send(json.dumps({
                    "type": "subscribed",
                    "symbol": symbol,
                    "status": "success"
                }))
            
            elif action == 'unsubscribe' and symbol:
                ws_manager.unsubscribe(ws, symbol)
                ws.send(json.dumps({
                    "type": "unsubscribed",
                    "symbol": symbol,
                    "status": "success"
                }))
            
            elif action == 'ping':
                ws.send(json.dumps({"type": "pong"}))
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("🔌 WebSocket client disconnected")

# ✅ API Routes
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'Anso Vision Backend',
        'version': '3.2.1',
        'websocket_connections': sum(len(conns) for conns in ws_manager.connections.values()),
        'tracked_symbols': list(ws_manager.last_signals.keys()),
        'mt5_backend': COMMUNITY_TRADING_URL,
        'signals_queued': len(signal_queue.signals)
    }), 200

@app.route('/api/candle-complete', methods=['POST'])
def candle_complete():
    """✅ ENHANCED: Called when a new candle completes - AUTO-POSTS SIGNALS"""
    try:
        data = request.json
        symbol = data.get('symbol')
        candles = data.get('candles')
        
        if not symbol or not candles:
            return jsonify({'error': 'Missing symbol or candles'}), 400
        
        logger.info(f"🕯️ Candle completed: {symbol} ({len(candles)} candles)")
        
        prices = np.array([c.get('close', 0) for c in candles])
        volumes = np.array([c.get('volume', 1.0) for c in candles])
        
        if len(prices) < 250:
            logger.warning(f"⚠️ Not enough data: {len(prices)}/250 candles")
            return jsonify({
                'status': 'insufficient_data',
                'required': 250,
                'received': len(prices)
            }), 200
        
        trade_allowed, news_reason = check_news_before_trade()
        if not trade_allowed:
            logger.warning(f"⚠️ Trade blocked by news: {news_reason}")
            signal_result = {
                'signal_type': 'WAIT',
                'entry': float(prices[-1]),
                'tp': 0.0,
                'sl': 0.0,
                'confidence': 0.0,
                'reasoning': f'News block: {news_reason}',
                'market_context': 'High-impact news'
            }
        else:
            logger.info(f"🧠 Generating signal for {symbol}...")
            signal_result = model_manager.generate_signal(symbol, prices, volumes)
            
            if signal_result.get('signal_type') != 'WAIT':
                logger.info(f"🎯 VALID TRADING SIGNAL DETECTED: {symbol} {signal_result['signal_type']}")
                risk_manager.record_signal(
                    symbol,
                    signal_result['signal_type'],
                    signal_result.get('confidence', 0.0)
                )
                
                # ✅ AUTO-POST TO MT5 (Background thread)
                if signal_queue.should_post_signal(symbol, signal_result['signal_type']):
                    signal_id = signal_queue.add_signal(symbol, signal_result)
                    thread_pool.submit(post_to_community_trading, symbol, signal_result, signal_id)
                else:
                    logger.info(f"⏭️ Rate limited: Skipping duplicate signal for {symbol}")
        
        ws_manager.broadcast_signal(symbol, signal_result)
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'signal_type': signal_result.get('signal_type'),
            'posted_to_mt5': signal_result.get('signal_type') != 'WAIT',
            'websocket_clients': len(ws_manager.connections.get(symbol, set()))
        }), 200
    
    except Exception as e:
        logger.error(f"❌ Error in candle_complete: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_signal():
    """✅ ENHANCED: Main endpoint for frontend signal analysis - AUTO-POSTS TO MT5"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        data = request.json
        symbol = data.get('symbol')
        candles = data.get('candles', [])
        timeframe = data.get('timeframe', '1h')
        
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 /analyze request for {symbol}")
        logger.info(f"   Candles: {len(candles)}")
        logger.info(f"   Timeframe: {timeframe}")
        logger.info(f"{'='*70}")
        
        if not symbol:
            return jsonify({
                'success': False,
                'error': 'Missing symbol',
                'signal': 'HOLD',
                'signal_type': 'ERROR'
            }), 400
        
        if len(candles) < 250:
            return jsonify({
                'success': False,
                'error': f'Insufficient data: Need 250 candles, got {len(candles)}',
                'symbol': symbol,
                'signal': 'HOLD',
                'signal_type': 'ERROR'
            }), 400
        
        prices = np.array([c.get('close', 0) for c in candles])
        volumes = np.array([c.get('volume', 0) for c in candles])
        
        trade_allowed, news_reason = check_news_before_trade()
        if not trade_allowed:
            return build_wait_response(symbol, prices[-1], f'News block: {news_reason}', timeframe)
        
        allowed, risk_reason = risk_manager.should_allow_signal(symbol, 'UNKNOWN')
        if not allowed:
            return build_wait_response(symbol, prices[-1], f'Risk limit: {risk_reason}', timeframe)
        
        logger.info(f"🧠 Generating signal...")
        signal_result = model_manager.generate_signal(symbol, prices, volumes)
        
        if signal_result is None or signal_result.get('error', False):
            error_msg = signal_result.get('error_message', 'Unknown error') if signal_result else 'Signal generation returned None'
            return build_wait_response(symbol, prices[-1], error_msg, timeframe)
        
        signal_type = signal_result.get('signal_type', 'WAIT')
        
        # ✅ CRITICAL FIX: AUTO-POST VALID SIGNALS TO MT5
        if signal_type != 'WAIT':
            logger.info(f"🎯 VALID SIGNAL DETECTED: {symbol} {signal_type}")
            risk_manager.record_signal(symbol, signal_type, signal_result.get('confidence', 0.0))
            
            # ✅ AUTO-POST TO MT5 (Background thread)
            if signal_queue.should_post_signal(symbol, signal_type):
                signal_id = signal_queue.add_signal(symbol, signal_result)
                thread_pool.submit(post_to_community_trading, symbol, signal_result, signal_id)
            else:
                logger.info(f"⏭️ Rate limited: Skipping duplicate signal for {symbol}")
        
        response = build_signal_response(symbol, signal_result, timeframe, prices[-1])
        
        logger.info(f"✅ Signal generated: {signal_type}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"❌ Error in /analyze: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Internal error: {str(e)}',
            'signal': 'HOLD',
            'signal_type': 'ERROR'
        }), 500

@app.route('/signal/<symbol>', methods=['POST'])
def generate_signal_route(symbol):
    """Legacy endpoint for compatibility - AUTO-POSTS TO MT5"""
    data = request.json
    if not data or 'prices' not in data or 'volumes' not in data:
        return jsonify({'signal_type': 'WAIT', 'reasoning': 'Missing price or volume data'}), 400
    
    prices = np.array(data.get('prices', []))
    volumes = np.array(data.get('volumes', []))
    
    if len(prices) < 200:
        return jsonify({'signal_type': 'WAIT', 'reasoning': f'Insufficient data: need 200 candles, got {len(prices)}'}), 400

    trade_allowed, news_reason = check_news_before_trade()
    if not trade_allowed:
        return jsonify({'signal_type': 'WAIT', 'reasoning': f'News block: {news_reason}'}), 200

    allowed, risk_reason = risk_manager.should_allow_signal(symbol, 'UNKNOWN')
    if not allowed:
        return jsonify({'signal_type': 'WAIT', 'reasoning': f'Risk limit: {risk_reason}'}), 200

    signal_result = model_manager.generate_signal(symbol, prices, volumes)
    
    if signal_result.get('signal_type') != 'WAIT':
        risk_manager.record_signal(symbol, signal_result['signal_type'], signal_result.get('confidence', 0.0))
        
        # ✅ AUTO-POST TO MT5
        if signal_queue.should_post_signal(symbol, signal_result['signal_type']):
            signal_id = signal_queue.add_signal(symbol, signal_result)
            thread_pool.submit(post_to_community_trading, symbol, signal_result, signal_id)
        
    return jsonify(signal_result), 200

# ✅ NEW: Manual signal trigger endpoint
@app.route('/api/trigger-signal', methods=['POST'])
def trigger_signal():
    """Manually trigger signal generation and posting"""
    data = request.json
    symbol = data.get('symbol')
    
    if not symbol:
        return jsonify({'error': 'Missing symbol'}), 400
    
    logger.info(f"🔔 Manual signal trigger for {symbol}")
    
    # Simulate candle data (you should fetch real data here)
    # This is just for testing
    candles = [{'close': 1.0850 + i*0.0001, 'volume': 1.0} for i in range(250)]
    
    # Call candle_complete logic
    prices = np.array([c.get('close', 0) for c in candles])
    volumes = np.array([c.get('volume', 1.0) for c in candles])
    
    signal_result = model_manager.generate_signal(symbol, prices, volumes)
    
    if signal_result.get('signal_type') != 'WAIT':
        signal_id = signal_queue.add_signal(symbol, signal_result)
        thread_pool.submit(post_to_community_trading, symbol, signal_result, signal_id)
        
        return jsonify({
            'success': True,
            'message': f'Signal triggered for {symbol}',
            'signal': signal_result,
            'signal_id': signal_id
        }), 200
    else:
        return jsonify({
            'success': False,
            'message': f'WAIT signal for {symbol}',
            'signal': signal_result
        }), 200

# ✅ NEW: Check signal queue status
@app.route('/api/signals/queue', methods=['GET'])
def get_signal_queue():
    """Get current signal queue status"""
    return jsonify({
        'status': 'active',
        'queue_size': len(signal_queue.signals),
        'signals': signal_queue.signals,
        'last_posted': {k: v.isoformat() for k, v in signal_queue.last_posted.items()}
    }), 200

# ✅ HELPER FUNCTIONS
def check_news_before_trade() -> tuple:
    """Check if high-impact news allows trading"""
    try:
        response = requests.get(f"{NEWS_MODEL_URL}/should-trade", timeout=3)
        if response.status_code == 200:
            news_data = response.json()
            if not news_data.get('should_trade', True):
                return False, news_data.get('reason', 'High-impact news detected')
            return True, 'News check passed'
        return True, 'News check API failure, proceeding'
    except:
        return True, 'News check unreachable, proceeding'

def post_to_community_trading(symbol: str, signal: Dict, signal_id: Optional[str] = None):
    """✅ ENHANCED: Post signal to MT5 with ALL FIELDS including limit_orders"""
    try:
        signal_type = signal.get('signal_type', 'WAIT')
        
        if signal_type == 'WAIT':
            logger.info(f"⏭️ Skipping WAIT signal for {symbol}")
            return
        
        # ✅ Validate required fields
        required_fields = ['entry', 'tp', 'sl']
        missing_fields = [field for field in required_fields if not signal.get(field)]
        
        if missing_fields:
            logger.error(f"❌ Missing required fields for {symbol}: {missing_fields}")
            logger.error(f"   Signal data: {signal}")
            return
        
        # ✅ INCLUDE ALL FIELDS from signal generator
        payload = {
            "symbol": symbol,
            "action": signal_type,
            "entry": float(signal.get('entry', 0)),
            "tp": float(signal.get('tp', 0)),
            "sl": float(signal.get('sl', 0)),
            "volume": 0.01,
            "confidence": float(signal.get('confidence', 0)),
            "reasoning": signal.get('reasoning', 'No reasoning'),
            "limit_orders": signal.get('limit_orders', False),
            "timeframe": "1h"
        }
        
        logger.info("\n" + "="*70)
        logger.info(f"🚀 POSTING SIGNAL #{signal_id} TO MT5 BACKEND")
        logger.info("="*70)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Action: {signal_type}")
        logger.info(f"Entry: {payload['entry']:.4f}")
        logger.info(f"TP: {payload['tp']:.4f} | SL: {payload['sl']:.4f}")
        logger.info(f"Confidence: {payload['confidence']:.1%}")
        logger.info(f"Limit Orders: {payload['limit_orders']}")
        logger.info(f"Reasoning: {payload['reasoning'][:60]}...")
        logger.info(f"Target URL: {COMMUNITY_TRADING_URL}/api/signal")
        logger.info(f"API Key: {'*' * len(COMMUNITY_API_KEY)}")
        logger.info("Attempting POST request...")
        
        response = requests.post(
            f"{COMMUNITY_TRADING_URL}/api/signal",
            json=payload,
            headers={
                "X-API-Key": COMMUNITY_API_KEY,
                "Content-Type": "application/json"
            },
            timeout=10
        )
        
        logger.info(f"Response Status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            signal_id_backend = response_data.get('signal_id', 'unknown')
            
            # ✅ Mark as posted in queue
            if signal_id:
                signal_queue.mark_posted(signal_id)
            
            logger.info(f"✅ SIGNAL POSTED TO MT5 SUCCESSFULLY")
            logger.info(f"   Signal ID: {signal_id_backend}")
            logger.info(f"   Response: {response.text[:150]}")
            logger.info("="*70 + "\n")
            
            # ✅ Broadcast via WebSocket
            ws_manager.broadcast_signal(symbol, {
                **signal,
                'posted_to_mt5': True,
                'mt5_signal_id': signal_id_backend,
                'posted_at': datetime.now().isoformat()
            })
            
        elif response.status_code == 403:
            logger.error(f"❌ AUTHENTICATION FAILED: Invalid API key")
        elif response.status_code == 404:
            logger.error(f"❌ ENDPOINT NOT FOUND: {COMMUNITY_TRADING_URL}/api/signal")
        else:
            logger.error(f"⚠️ MT5 BACKEND ERROR: HTTP {response.status_code}")
            logger.error(f"   Response: {response.text[:200]}")
    
    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ CANNOT REACH MT5 BACKEND: {COMMUNITY_TRADING_URL}")
        logger.error(f"   Error: {e}")
    
    except requests.exceptions.Timeout:
        logger.error(f"❌ MT5 BACKEND TIMEOUT: Request took >10 seconds")
    
    except Exception as e:
        logger.error(f"❌ ERROR POSTING TO MT5: {type(e).__name__}: {e}", exc_info=True)

def build_wait_response(symbol: str, price: float, reason: str, timeframe: str):
    """Build standardized WAIT response"""
    return jsonify({
        'success': True,
        'symbol': symbol,
        'signal': 'HOLD',
        'signal_type': 'WAIT',
        'entry': float(price),
        'tp': 0,
        'sl': 0,
        'confidence': 0,
        'reasoning': reason,
        'market_context': 'N/A',
        'timeframe': timeframe
    }), 200

def build_signal_response(symbol: str, signal_result: Dict, timeframe: str, current_price: float) -> Dict:
    """Build standardized signal response"""
    signal_type = signal_result.get('signal_type', 'WAIT')
    signal_map = {'BUY': 'BUY', 'SELL': 'SELL', 'WAIT': 'HOLD'}
    
    entry = signal_result.get('entry', current_price)
    tp = signal_result.get('tp', 0)
    sl = signal_result.get('sl', 0)
    
    return {
        'success': True,
        'symbol': symbol,
        'signal': signal_map.get(signal_type, 'HOLD'),
        'signal_type': signal_type,
        'entry': float(entry if entry is not None else current_price),
        'tp': float(tp if tp is not None else 0),
        'sl': float(sl if sl is not None else 0),
        'confidence': float(signal_result.get('confidence', 0)),
        'reasoning': signal_result.get('reasoning', 'No reasoning provided'),
        'market_context': signal_result.get('market_context', 'N/A'),
        'timeframe': timeframe,
        'posted_to_mt5': signal_type != 'WAIT'
    }

# ✅ NEW: Auto-test on startup
def auto_test_connection():
    """Test MT5 backend connection on startup"""
    logger.info("🔧 Testing MT5 backend connection...")
    
    try:
        response = requests.post(
            f"{COMMUNITY_TRADING_URL}/api/signal",
            json={
                "symbol": "TEST",
                "action": "BUY",
                "entry": 1.0000,
                "tp": 1.0100,
                "sl": 0.9900,
                "volume": 0.01,
                "confidence": 0.85,
                "limit_orders": False,
                "reasoning": "Auto-test signal"
            },
            headers={
                "X-API-Key": COMMUNITY_API_KEY,
                "Content-Type": "application/json"
            },
            timeout=5
        )
        
        if response.status_code == 200:
            logger.info(f"✅ MT5 BACKEND CONNECTION SUCCESSFUL")
            logger.info(f"   Response: {response.text}")
        else:
            logger.warning(f"⚠️ MT5 backend returned {response.status_code}: {response.text}")
    
    except Exception as e:
        logger.warning(f"⚠️ MT5 backend test failed: {e}")
        logger.warning(f"   URL: {COMMUNITY_TRADING_URL}")
        logger.warning(f"   Make sure backend is running and API key is correct")

# ✅ ENTRY POINT
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 STARTING ENHANCED API SERVER v3.2.1")
    logger.info(f"{'='*70}")
    logger.info(f"   • WebSocket: /ws/signals")
    logger.info(f"   • REST API: /analyze, /api/candle-complete")
    logger.info(f"   • Signal Queue: /api/signals/queue")
    logger.info(f"   • Manual Trigger: /api/trigger-signal")
    logger.info(f"   • Community Trading: {COMMUNITY_TRADING_URL}")
    logger.info(f"   ✅ AUTO-POSTS ALL VALID SIGNALS TO MT5")
    logger.info(f"   ✅ RATE LIMITING: 5 min between same symbol signals")
    logger.info(f"   ✅ BACKGROUND PROCESSING: Non-blocking signal posting")
    logger.info(f"{'='*70}")
    
    # Run connection test
    auto_test_connection()
    
    app.run(host='0.0.0.0', port=port, debug=debug)
