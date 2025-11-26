"""
INTEGRATED API Server v3.0
Works with Data Fetcher and News Model services
"""
import os
import requests
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from datetime import datetime

# Import components
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

# Service URLs (from environment or defaults)
DATA_FETCHER_URL = os.getenv('DATA_FETCHER_URL', 'https://anso-vision-data-fetcher.onrender.com')
NEWS_MODEL_URL = os.getenv('NEWS_MODEL_URL', 'https://anso-vision-news-model.onrender.com')
TRADING_BACKEND_URL = os.getenv('TRADING_BACKEND_URL', 'https://anso-vision-backend.onrender.com')
TRADING_API_KEY = os.getenv('TRADING_API_KEY', 'Mr.creative090')


def check_news_before_trade() -> tuple[bool, str]:
    """
    Check if trading should be allowed based on high-impact news
    Calls the News Model service
    
    Returns:
        (can_trade: bool, reason: str)
    """
    try:
        response = requests.get(
            f"{NEWS_MODEL_URL}/should-trade",
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            can_trade = data.get("can_trade", True)
            reason = data.get("reason", "Unknown")
            
            if not can_trade:
                blocked_events = data.get("blocked_events", [])
                if blocked_events:
                    reason = f"High-impact news: {blocked_events[0].get('title', 'Unknown event')}"
            
            return can_trade, reason
        else:
            # If news service fails, allow trading by default (don't block unnecessarily)
            return True, "News service unavailable - trading allowed by default"
            
    except Exception as e:
        print(f"⚠️ News check failed: {str(e)}")
        return True, f"News check error: {str(e)} - trading allowed by default"


def send_signal_to_trading_backend(signal_data: dict):
    """Send generated signal to trading backend for storage and distribution"""
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
            "timeframe": signal_data.get("timeframe", "1h"),
            "ensemble_strength": signal_data.get("ensemble_strength")
        }
        
        response = requests.post(
            f"{TRADING_BACKEND_URL}/api/signal",
            json=payload,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"✅ Signal sent to trading backend: {signal_data['symbol']} {signal_data.get('signal')}")
            return response.json()
        else:
            print(f"⚠️ Trading backend returned {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error sending to trading backend: {e}")
        return None


@app.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check"""
    # Check other services
    services_status = {}
    
    # Check Data Fetcher
    try:
        df_response = requests.get(f"{DATA_FETCHER_URL}/health", timeout=5)
        services_status['data_fetcher'] = 'healthy' if df_response.status_code == 200 else 'unhealthy'
    except:
        services_status['data_fetcher'] = 'unreachable'
    
    # Check News Model
    try:
        news_response = requests.get(f"{NEWS_MODEL_URL}/health", timeout=5)
        services_status['news_model'] = 'healthy' if news_response.status_code == 200 else 'unhealthy'
    except:
        services_status['news_model'] = 'unreachable'
    
    return jsonify({
        'status': 'healthy',
        'service': 'Anso Vision Backend v3.0',
        'version': '3.0.0',
        'models_loaded': len(model_manager.models),
        'services': services_status,
        'endpoints': {
            'analyze': '/analyze - Analyze trading signals',
            'health': '/health - Health check',
            'model_stats': '/stats/models - Model statistics',
            'risk_stats': '/stats/risk - Risk management stats'
        },
        'external_services': {
            'data_fetcher': DATA_FETCHER_URL,
            'news_model': NEWS_MODEL_URL,
            'trading_backend': TRADING_BACKEND_URL
        }
    }), 200


@app.route('/analyze', methods=['POST'])
def analyze_signal():
    """
    Analyze market data and generate trading signals
    
    Flow:
    1. Frontend → Data Fetcher (gets candles)
    2. Frontend → Backend (sends candles for analysis)
    3. Backend → News Model (checks if trading allowed)
    4. Backend generates signal
    5. Backend → Trading Backend (sends signal if approved)
    
    Expected payload:
    {
        "symbol": "EURUSD",
        "candles": [...],  // From Data Fetcher
        "timeframe": "1h",
        "send_to_backend": true
    }
    """
    try:
        data = request.json or {}
        symbol = data.get('symbol')
        candles = data.get('candles')
        timeframe = data.get('timeframe', '1h')
        send_to_backend = data.get('send_to_backend', False)

        # Validation
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol is required'}), 400

        if not candles or len(candles) < 100:
            return jsonify({
                'success': False,
                'error': 'Insufficient candles',
                'required': 100,
                'provided': len(candles) if candles else 0,
                'message': 'Please ensure Data Fetcher provides at least 100 candles'
            }), 400

        # Extract price and volume data
        try:
            # Handle both Data Fetcher format and direct format
            if 'close' in candles[0]:
                prices = np.array([float(c.get('close', 0)) for c in candles], dtype=float)
                volumes = np.array([float(c.get('volume', 0)) for c in candles], dtype=float)
            elif 'timestamp' in candles[0]:
                prices = np.array([float(c.get('close', 0)) for c in candles], dtype=float)
                volumes = np.array([float(c.get('volume', 0)) for c in candles], dtype=float)
            else:
                raise ValueError("Invalid candle format")
                
        except (ValueError, TypeError, KeyError) as e:
            return jsonify({
                'success': False,
                'error': 'Invalid candle data format',
                'message': str(e),
                'expected_format': {
                    'timestamp': 'Unix timestamp or ISO string',
                    'open': 'float',
                    'high': 'float',
                    'low': 'float',
                    'close': 'float',
                    'volume': 'float'
                }
            }), 400

        # Check news before generating signal
        can_trade, news_reason = check_news_before_trade()
        
        if not can_trade:
            return jsonify({
                'success': False,
                'error': 'Trading blocked due to high-impact news',
                'reason': news_reason,
                'signal': 'WAIT',
                'can_retry_in_minutes': 30,
                'news_check': {
                    'can_trade': False,
                    'reason': news_reason
                }
            }), 200  # Return 200 but with WAIT signal

        # Generate signal using model manager
        signal = model_manager.generate_signal(symbol, prices, volumes, auto_train=True)
        
        if not signal or signal.get('entry') is None:
            return jsonify({
                'success': False,
                'error': 'Failed to generate signal',
                'message': 'Model not ready or insufficient data quality'
            }), 500
        
        signal_type = signal.get('signal_type', 'WAIT')
        
        # Validate with ensemble
        validation = ensemble_validator.validate_signal(signal, prices, volumes)
        
        # Check risk limits (only for actionable signals)
        if signal_type in ['BUY', 'SELL']:
            allowed, risk_reason = risk_manager.should_allow_signal(symbol, signal_type)
            
            if not allowed:
                return jsonify({
                    'success': False,
                    'error': 'Signal blocked by risk manager',
                    'reason': risk_reason,
                    'signal': 'WAIT',
                    'statistics': risk_manager.get_statistics(symbol)
                }), 429  # Too Many Requests
            
            # Record signal
            risk_manager.record_signal(symbol, signal_type, validation['confidence'])
        
        # Market structure analysis
        market_structure = market_analyzer.analyze_market_structure(prices, volumes)
        
        # Build comprehensive response
        response = {
            'success': True,
            'symbol': symbol,
            'timeframe': timeframe,
            'signal': signal_type,
            'entry': float(signal.get('entry', 0)),
            'tp': float(signal.get('tp', 0)),
            'sl': float(signal.get('sl', 0)),
            
            # Confidence metrics
            'confidence': float(signal.get('confidence', 0)),  # Original HMM confidence
            
            # Ensemble validation
            'ensemble_validation': {
                'approved': validation['approved'],
                'confidence': validation['confidence'],  # Combined confidence
                'strength': validation['strength'],
                'confirmations': validation['confirmations'],
                'warnings': validation['warnings'],
                'checks': validation['checks']
            },
            
            # Market analysis
            'market_context': signal.get('market_context', {}),
            'market_structure': market_structure,
            'risk_metrics': signal.get('risk_metrics', {}),
            
            # Model info
            'model_info': model_manager.get_model_stats(symbol),
            
            # News check
            'news_check': {
                'can_trade': can_trade,
                'reason': news_reason
            },
            
            # Reasoning
            'reasoning': signal.get('reasoning', validation.get('confirmations', ['Signal generated'])[0] if validation.get('confirmations') else 'Signal generated'),
            
            # Timestamp
            'timestamp': datetime.now().isoformat()
        }
        
        # Send to trading backend if requested and approved
        if send_to_backend and signal_type in ['BUY', 'SELL'] and validation['approved']:
            backend_result = send_signal_to_trading_backend({
                'symbol': symbol,
                'signal': signal_type,
                'entry': response['entry'],
                'tp': response['tp'],
                'sl': response['sl'],
                'confidence': validation['confidence'],
                'ensemble_strength': validation['strength'],
                'timeframe': timeframe
            })
            
            response['backend_transmission'] = {
                'sent': backend_result is not None,
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
            'message': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/stats/models', methods=['GET'])
def get_model_stats():
    """Get statistics for all trained models"""
    return jsonify(model_manager.get_all_stats()), 200


@app.route('/stats/risk', methods=['GET'])
def get_risk_stats():
    """Get risk management statistics"""
    symbol = request.args.get('symbol')
    return jsonify(risk_manager.get_statistics(symbol)), 200


@app.route('/admin/reset-limits', methods=['POST'])
def reset_limits():
    """Reset risk limits (admin only - for testing)"""
    data = request.json or {}
    symbol = data.get('symbol')
    
    risk_manager.reset_limits(symbol)
    
    return jsonify({
        'success': True,
        'message': f'Limits reset for {symbol if symbol else "all symbols"}'
    }), 200


@app.route('/admin/retrain', methods=['POST'])
def force_retrain():
    """Force retrain models (admin only - for testing)"""
    data = request.json or {}
    symbol = data.get('symbol')
    
    if symbol:
        # Get data from Data Fetcher
        try:
            response = requests.get(
                f"{DATA_FETCHER_URL}/candles/{symbol}",
                params={'interval': '1h', 'outputsize': 300},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                candles = data.get('candles', [])
                
                if len(candles) >= 250:
                    prices = np.array([c['close'] for c in candles])
                    success = model_manager.train_model(symbol, prices, force=True)
                    
                    return jsonify({
                        'success': success,
                        'symbol': symbol,
                        'message': 'Model retrained' if success else 'Training failed'
                    }), 200
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Insufficient data for training',
                        'candles': len(candles)
                    }), 400
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to fetch candles from Data Fetcher'
                }), 502
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    else:
        return jsonify({
            'success': False,
            'error': 'Symbol required'
        }), 400


if __name__ == '__main__':
    print(f"🚀 Anso Vision Backend v3.0 (Integrated) starting...")
    print(f"📊 Model Manager: Enabled")
    print(f"🤝 Ensemble Validator: Enabled")
    print(f"🛡️ Risk Manager: Enabled")
    print(f"\n🔗 External Services:")
    print(f"   Data Fetcher: {DATA_FETCHER_URL}")
    print(f"   News Model: {NEWS_MODEL_URL}")
    print(f"   Trading Backend: {TRADING_BACKEND_URL}")
    
    app.run(debug=False, port=5000, host='0.0.0.0')
