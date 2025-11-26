"""
PRODUCTION-READY API Server with:
- WebSocket real-time data
- Model persistence (no retraining every request)
- Ensemble validation
- Risk management & circuit breakers
- Proper error handling
"""
import os
import requests
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from datetime import datetime

# Import new components
from websocket_manager import WebSocketManager, SimulatedWebSocketManager
from model_manager import get_model_manager
from ensemble_validator import EnsembleValidator
from risk_manager import get_risk_manager
from market_analyzer import MarketAnalyzer

load_dotenv()

app = Flask(__name__)

# CORS Configuration
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')
CORS(app, resources={
    r"/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize components
model_manager = get_model_manager()
ensemble_validator = EnsembleValidator(min_confirmation_score=0.65)
risk_manager = get_risk_manager()
market_analyzer = MarketAnalyzer()

# WebSocket Manager (choose real or simulated)
USE_SIMULATED_WS = os.getenv('USE_SIMULATED_WEBSOCKET', 'false').lower() == 'true'
if USE_SIMULATED_WS:
    ws_manager = SimulatedWebSocketManager()
    print("🧪 Using SIMULATED WebSocket (for testing)")
else:
    ws_manager = WebSocketManager()
    print("📡 Using REAL WebSocket connections")

# Global state
MONITORED_SYMBOLS = []
WS_RUNNING = False

# Environment variables
NEWS_MODEL_URL = os.getenv('NEWS_MODEL_URL', 'https://anso-vision-news-model.onrender.com')
TRADING_BACKEND_URL = os.getenv('TRADING_BACKEND_URL', 'https://anso-vision-backend.onrender.com')
TRADING_API_KEY = os.getenv('TRADING_API_KEY', 'Mr.creative090')


def on_new_candle_callback(symbol: str, candle: dict):
    """Callback triggered when new candle arrives via WebSocket"""
    try:
        # Get candle buffer
        buffer = ws_manager.get_buffer(symbol)
        
        if not buffer or not buffer.is_ready(min_candles=250):
            return  # Not enough data yet
        
        # Get prices and volumes
        prices = buffer.get_prices(n=300)
        volumes = buffer.get_volumes(n=300)
        
        # Generate signal (model manager handles training automatically)
        signal = model_manager.generate_signal(symbol, prices, volumes, auto_train=True)
        
        if not signal or signal.get('entry') is None:
            return
        
        signal_type = signal.get('signal_type')
        
        if signal_type not in ['BUY', 'SELL']:
            return
        
        # Validate with ensemble
        validation = ensemble_validator.validate_signal(signal, prices, volumes)
        
        if not validation['approved']:
            print(f"⚠️ {symbol}: Signal rejected by ensemble - {', '.join(validation['warnings'])}")
            return
        
        # Check risk limits
        allowed, reason = risk_manager.should_allow_signal(symbol, signal_type)
        
        if not allowed:
            print(f"🛑 {symbol}: Signal blocked by risk manager - {reason}")
            return
        
        # Check market conditions
        volatility = signal.get('monte_carlo', {}).get('volatility', 0)
        volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 else 1
        
        suitable, condition_reason = risk_manager.check_market_conditions(volatility, volume_ratio)
        
        if not suitable:
            print(f"🌪️ {symbol}: Market conditions unsuitable - {condition_reason}")
            return
        
        # Record signal
        risk_manager.record_signal(symbol, signal_type, validation['confidence'])
        
        # Send to trading backend
        send_signal_to_trading_backend({
            'symbol': symbol,
            'signal': signal_type,
            'entry': signal['entry'],
            'tp': signal['tp'],
            'sl': signal['sl'],
            'confidence': validation['confidence'],
            'ensemble_strength': validation['strength'],
            'timeframe': '1m',
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"✅ {symbol}: {signal_type} signal sent (confidence: {validation['confidence']:.2f}, strength: {validation['strength']})")
        
    except Exception as e:
        print(f"❌ Error in candle callback for {symbol}: {e}")


def send_signal_to_trading_backend(signal_data: dict):
    """Send generated signal to trading backend"""
    try:
        headers = {
            "X-API-Key": TRADING_API_KEY,
            "Content-Type": "application/json"
        }
        
        payload = {
            "symbol": signal_data.get("symbol"),
            "action": signal_data.get("signal"),
            "volume": 0.01,
            "sl": signal_data.get("sl"),
            "tp": signal_data.get("tp"),
            "confidence": signal_data.get("confidence"),
            "timeframe": signal_data.get("timeframe", "1m"),
            "ensemble_strength": signal_data.get("ensemble_strength")
        }
        
        response = requests.post(
            f"{TRADING_BACKEND_URL}/api/signal",
            json=payload,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"⚠️ Trading backend returned {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error sending to trading backend: {e}")
        return None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Anso Vision Backend v3.0',
        'version': '3.0.0',
        'websocket_active': WS_RUNNING,
        'monitored_symbols': MONITORED_SYMBOLS,
        'models_loaded': len(model_manager.models),
        'trading_backend_url': TRADING_BACKEND_URL
    }), 200


@app.route('/analyze', methods=['POST'])
def analyze_signal():
    """
    Analyze market data and generate trading signals (REST endpoint)
    
    Expected payload:
    {
        "symbol": "EURUSD",
        "candles": [...],
        "timeframe": "1h",
        "send_to_backend": true
    }
    """
    try:
        data = request.json or {}
        symbol = data.get('symbol')
        candles = data.get('candles')
        timeframe = data.get('timeframe', '1h')
        send_to_backend = data.get('send_to_backend', True)

        # Validation
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol is required'}), 400

        if not candles or len(candles) < 250:
            return jsonify({
                'success': False,
                'error': 'Insufficient candles',
                'required': 250,
                'provided': len(candles) if candles else 0
            }), 400

        # Extract data
        try:
            prices = np.array([float(c.get('close', 0)) for c in candles], dtype=float)
            volumes = np.array([float(c.get('volume', 0)) for c in candles], dtype=float)
        except (ValueError, TypeError) as e:
            return jsonify({
                'success': False,
                'error': 'Invalid candle data format',
                'message': str(e)
            }), 400

        # Generate signal using model manager (handles training automatically)
        signal = model_manager.generate_signal(symbol, prices, volumes, auto_train=True)
        
        if not signal or signal.get('entry') is None:
            return jsonify({
                'success': False,
                'error': 'Failed to generate signal',
                'message': 'Model not ready or insufficient data'
            }), 500
        
        signal_type = signal.get('signal_type', 'WAIT')
        
        # Validate with ensemble
        validation = ensemble_validator.validate_signal(signal, prices, volumes)
        
        # Check risk limits
        if signal_type in ['BUY', 'SELL']:
            allowed, reason = risk_manager.should_allow_signal(symbol, signal_type)
            if not allowed:
                return jsonify({
                    'success': False,
                    'error': 'Signal blocked by risk manager',
                    'reason': reason,
                    'statistics': risk_manager.get_statistics(symbol)
                }), 429  # Too Many Requests
            
            # Record signal
            risk_manager.record_signal(symbol, signal_type, validation['confidence'])
        
        # Market structure
        market_structure = market_analyzer.analyze_market_structure(prices, volumes)
        
        # Build response
        response = {
            'success': True,
            'symbol': symbol,
            'timeframe': timeframe,
            'signal': signal_type,
            'entry': float(signal.get('entry', 0)),
            'tp': float(signal.get('tp', 0)),
            'sl': float(signal.get('sl', 0)),
            'original_confidence': float(signal.get('confidence', 0)),
            'ensemble_validation': {
                'approved': validation['approved'],
                'confidence': validation['confidence'],
                'strength': validation['strength'],
                'confirmations': validation['confirmations'],
                'warnings': validation['warnings'],
                'checks': validation['checks']
            },
            'market_structure': market_structure,
            'risk_metrics': signal.get('risk_metrics', {}),
            'model_info': model_manager.get_model_stats(symbol),
            'timestamp': datetime.now().isoformat()
        }
        
        # Send to trading backend if requested and signal is valid
        if send_to_backend and signal_type in ['BUY', 'SELL'] and validation['approved']:
            backend_result = send_signal_to_trading_backend(response)
            response['backend_transmission'] = {
                'sent': True,
                'result': backend_result
            }
        
        return jsonify(response), 200

    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/websocket/start', methods=['POST'])
def start_websocket():
    """Start WebSocket monitoring for symbols"""
    global WS_RUNNING, MONITORED_SYMBOLS
    
    try:
        data = request.json or {}
        symbols = data.get('symbols', [])
        timeframe = data.get('timeframe', '1m')
        
        if not symbols:
            return jsonify({
                'success': False,
                'error': 'No symbols provided'
            }), 400
        
        # Register callbacks
        for symbol in symbols:
            ws_manager.add_symbol(symbol)
            ws_manager.register_callback(symbol, on_new_candle_callback)
        
        # Start WebSocket
        if not WS_RUNNING:
            ws_manager.start(symbols, timeframe=timeframe)
            WS_RUNNING = True
        
        MONITORED_SYMBOLS = symbols
        
        return jsonify({
            'success': True,
            'message': f'WebSocket started for {len(symbols)} symbols',
            'symbols': symbols,
            'timeframe': timeframe
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/websocket/stop', methods=['POST'])
def stop_websocket():
    """Stop WebSocket monitoring"""
    global WS_RUNNING
    
    try:
        ws_manager.stop()
        WS_RUNNING = False
        
        return jsonify({
            'success': True,
            'message': 'WebSocket stopped'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/stats/models', methods=['GET'])
def get_model_stats():
    """Get statistics for all models"""
    return jsonify(model_manager.get_all_stats()), 200


@app.route('/stats/risk', methods=['GET'])
def get_risk_stats():
    """Get risk management statistics"""
    symbol = request.args.get('symbol')
    return jsonify(risk_manager.get_statistics(symbol)), 200


@app.route('/admin/reset-limits', methods=['POST'])
def reset_limits():
    """Reset risk limits (admin only)"""
    data = request.json or {}
    symbol = data.get('symbol')
    
    risk_manager.reset_limits(symbol)
    
    return jsonify({
        'success': True,
        'message': f'Limits reset for {symbol if symbol else "all symbols"}'
    }), 200


if __name__ == '__main__':
    print(f"🚀 Anso Vision Backend v3.0 starting...")
    print(f"📊 Model Manager: Enabled")
    print(f"🤝 Ensemble Validator: Enabled")
    print(f"🛡️ Risk Manager: Enabled")
    print(f"📡 WebSocket: {'Simulated' if USE_SIMULATED_WS else 'Real'}")
    print(f"🔗 Trading Backend: {TRADING_BACKEND_URL}")
    
    app.run(debug=False, port=5000, host='0.0.0.0')
