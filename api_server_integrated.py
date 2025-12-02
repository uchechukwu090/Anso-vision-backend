"""
API SERVER INTEGRATED - Enhanced with WebSocket Support
Single deployment point with real-time analysis capabilities
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
from typing import Dict, Set
from datetime import datetime

# Core integration
from model_manager import get_model_manager
from risk_manager import get_risk_manager

load_dotenv()

# Flask setup
app = Flask(__name__)
sock = Sock(app)

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

# WebSocket connection manager
class WebSocketManager:
    def __init__(self):
        self.connections: Dict[str, Set] = {}  # symbol -> set of websockets
        self.last_signals: Dict[str, Dict] = {}  # symbol -> last signal
        self.lock = threading.Lock()
    
    def subscribe(self, ws, symbol: str):
        with self.lock:
            if symbol not in self.connections:
                self.connections[symbol] = set()
            self.connections[symbol].add(ws)
            print(f"✅ WebSocket subscribed to {symbol}. Total: {len(self.connections[symbol])}")
            
            # Send last signal if exists
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
                print(f"❌ WebSocket unsubscribed from {symbol}")
    
    def broadcast_signal(self, symbol: str, signal: Dict):
        with self.lock:
            # Check if signal changed
            last_signal = self.last_signals.get(symbol, {})
            signal_changed = (
                last_signal.get('signal_type') != signal.get('signal_type') or
                abs(last_signal.get('confidence', 0) - signal.get('confidence', 0)) > 0.1
            )
            
            if signal_changed:
                print(f"🔔 {symbol}: Signal changed - {signal['signal_type']} @ {signal.get('confidence', 0):.1%}")
            
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
            
            # Clean up disconnected
            for ws in disconnected:
                self.connections[symbol].discard(ws)
    
    def _send_safe(self, ws, message: dict) -> bool:
        try:
            ws.send(json.dumps(message))
            return True
        except:
            return False

ws_manager = WebSocketManager()

# ----------------------------------------------------------------------
# WEBSOCKET ENDPOINT
# ----------------------------------------------------------------------

@sock.route('/ws/signals')
def websocket_signals(ws):
    """
    WebSocket endpoint for real-time signal updates
    Client sends: {"action": "subscribe", "symbol": "BTCUSD"}
    """
    print("🔌 WebSocket client connected")
    
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
        print(f"WebSocket error: {e}")
    finally:
        print("🔌 WebSocket client disconnected")

# ----------------------------------------------------------------------
# API ROUTES
# ----------------------------------------------------------------------

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'Anso Vision Backend',
        'version': '3.0.0',
        'websocket_connections': sum(len(conns) for conns in ws_manager.connections.values()),
        'tracked_symbols': list(ws_manager.last_signals.keys())
    }), 200


@app.route('/api/candle-complete', methods=['POST'])
def candle_complete():
    """
    Called by data fetcher when a new candle completes
    Triggers automatic analysis and broadcasts to WebSocket clients
    """
    try:
        data = request.json
        symbol = data.get('symbol')
        candles = data.get('candles')
        
        if not symbol or not candles:
            return jsonify({'error': 'Missing symbol or candles'}), 400
        
        print(f"\n🕯️ Candle completed: {symbol} ({len(candles)} candles)")
        
        # Extract prices and volumes
        prices = np.array([c.get('close', 0) for c in candles])
        volumes = np.array([c.get('volume', 1.0) for c in candles])
        
        if len(prices) < 250:
            print(f"⚠️ Not enough data: {len(prices)}/250 candles")
            return jsonify({
                'status': 'insufficient_data',
                'required': 250,
                'received': len(prices)
            }), 200
        
        # Check news
        trade_allowed, news_reason = check_news_before_trade()
        if not trade_allowed:
            print(f"⚠️ Trade blocked by news: {news_reason}")
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
            # Generate signal
            print(f"🧠 Generating signal for {symbol}...")
            signal_result = model_manager.generate_signal(symbol, prices, volumes)
            
            # Record if not WAIT
            if signal_result.get('signal_type') != 'WAIT':
                risk_manager.record_signal(
                    symbol,
                    signal_result['signal_type'],
                    signal_result.get('confidence', 0.0)
                )
        
        # Broadcast to WebSocket clients
        ws_manager.broadcast_signal(symbol, signal_result)
        
        # Post to community trading if not WAIT
        if signal_result.get('signal_type') != 'WAIT':
            post_to_community_trading(symbol, signal_result)
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'signal_type': signal_result.get('signal_type'),
            'websocket_clients': len(ws_manager.connections.get(symbol, set()))
        }), 200
    
    except Exception as e:
        print(f"❌ Error in candle_complete: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_signal():
    """
    Main endpoint for frontend signal analysis
    Expects: { symbol, candles: [{timestamp, open, high, low, close, volume}], timeframe }
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        data = request.json
        symbol = data.get('symbol')
        candles = data.get('candles', [])
        timeframe = data.get('timeframe', '1h')
        
        print(f"\n{'='*70}")
        print(f"📊 /analyze request for {symbol}")
        print(f"   Candles: {len(candles)}")
        print(f"   Timeframe: {timeframe}")
        print(f"{'='*70}")
        
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
                'error': f'Insufficient data: Need 250 candles for stable HMM training, got {len(candles)}',
                'symbol': symbol,
                'signal': 'HOLD',
                'signal_type': 'ERROR',
                'entry': 0,
                'tp': 0,
                'sl': 0,
                'confidence': 0,
                'reasoning': f'Insufficient data: HMM requires 250 candles. Got {len(candles)}. Please fetch more historical data.',
                'market_context': 'N/A'
            }), 400
        
        # Extract prices and volumes
        prices = np.array([c.get('close', 0) for c in candles])
        volumes = np.array([c.get('volume', 0) for c in candles])
        
        if np.any(prices <= 0) or np.any(volumes < 0):
            return jsonify({
                'success': False,
                'error': 'Invalid candle data',
                'symbol': symbol,
                'signal': 'HOLD',
                'signal_type': 'ERROR'
            }), 400
        
        # Check news
        trade_allowed, news_reason = check_news_before_trade()
        if not trade_allowed:
            return build_wait_response(symbol, prices[-1], f'News block: {news_reason}', timeframe)
        
        # Check risk manager
        allowed, risk_reason = risk_manager.should_allow_signal(symbol, 'UNKNOWN')
        if not allowed:
            return build_wait_response(symbol, prices[-1], f'Risk limit: {risk_reason}', timeframe)
        
        # Generate signal
        print(f"🧠 Generating signal...")
        signal_result = model_manager.generate_signal(symbol, prices, volumes)
        
        if signal_result is None or signal_result.get('error', False):
            error_msg = signal_result.get('error_message', 'Unknown error') if signal_result else 'Signal generation returned None'
            return build_wait_response(symbol, prices[-1], error_msg, timeframe)
        
        # Record signal if not WAIT
        signal_type = signal_result.get('signal_type', 'WAIT')
        if signal_type != 'WAIT':
            risk_manager.record_signal(symbol, signal_type, signal_result.get('confidence', 0.0))
        
        # Build response
        response = build_signal_response(symbol, signal_result, timeframe, prices[-1])
        
        print(f"✅ Signal generated: {signal_type}")
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Error in /analyze: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'Internal error: {str(e)}',
            'signal': 'HOLD',
            'signal_type': 'ERROR'
        }), 500


@app.route('/signal/<symbol>', methods=['POST'])
def generate_signal_route(symbol):
    """Legacy endpoint for compatibility"""
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
        
    return jsonify(signal_result), 200


# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------

def check_news_before_trade() -> tuple[bool, str]:
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


def post_to_community_trading(symbol: str, signal: Dict):
    """Post signal to MT5 community trading platform"""
    try:
        payload = {
            "symbol": symbol,
            "action": signal['signal_type'],
            "entry": signal['entry'],
            "tp": signal['tp'],
            "sl": signal['sl'],
            "confidence": signal['confidence'],
            "reasoning": signal['reasoning']
        }
        
        response = requests.post(
            f"{COMMUNITY_TRADING_URL}/api/signals/external",
            json=payload,
            headers={"X-API-Key": COMMUNITY_API_KEY},
            timeout=5
        )
        
        if response.status_code == 200:
            print(f"✅ Signal posted to community platform")
        else:
            print(f"⚠️ Community platform returned {response.status_code}")
    
    except Exception as e:
        print(f"❌ Failed to post to community: {e}")


def build_wait_response(symbol: str, price: float, reason: str, timeframe: str) -> tuple:
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
        'timeframe': timeframe,
        'risk_metrics': {
            'risk_reward_ratio': 0,
            'potential_profit_pct': 0,
            'potential_loss_pct': 0,
            'prob_tp_hit': 0,
            'prob_sl_hit': 0,
            'expected_value': 0,
            'expected_value_pct': 0
        }
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
        'distance_info': signal_result.get('distance_info'),  # NEW: Distance tracking
        'is_breakout': signal_result.get('is_breakout', False),  # NEW: Breakout flag
        'is_discounted': signal_result.get('is_discounted', False),  # NEW: Discount flag
        'learning_stats': signal_result.get('learning_stats'),  # NEW: Learning metrics
        'risk_metrics': signal_result.get('risk_metrics', {
            'risk_reward_ratio': 0,
            'potential_profit_pct': 0,
            'potential_loss_pct': 0,
            'prob_tp_hit': 0,
            'prob_sl_hit': 0,
            'expected_value': 0,
            'expected_value_pct': 0
        })
    }


# ----------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    print(f"🚀 Starting Enhanced API Server on port {port}")
    print(f"   • WebSocket: /ws/signals")
    print(f"   • REST API: /analyze, /api/candle-complete")
    print(f"   • Community Trading: {COMMUNITY_TRADING_URL}")
    app.run(host='0.0.0.0', port=port, debug=debug)
