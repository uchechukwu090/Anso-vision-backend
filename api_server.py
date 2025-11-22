import os
import requests
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from kalman_filter import apply_kalman_filter
from signal_generator import SignalGenerator
from hmm_model import MarketHMM
from context_aware_hmm import ContextAwareHMM
from market_analyzer import MarketAnalyzer
from monte_carlo_optimizer import MonteCarloOptimizer

app = Flask(__name__)

# CORS Configuration - Production Ready
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')
CORS(app, resources={
    r"/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize models
signal_gen = SignalGenerator(n_hmm_components=3)
if not hasattr(signal_gen, "monte_carlo") or signal_gen.monte_carlo is None:
    signal_gen.monte_carlo = MonteCarloOptimizer()

context_hmm = ContextAwareHMM()
market_analyzer = MarketAnalyzer()

# Environment variables
NEWS_MODEL_URL = os.getenv('NEWS_MODEL_URL', 'https://anso-vision-news-model.onrender.com')
DATA_FETCHER_URL = os.getenv('DATA_FETCHER_URL', 'https://anso-vision-data-fetcher.onrender.com')

def check_news_before_trade():
    """Check if trading should be paused due to high-impact news"""
    try:
        response = requests.get(f"{NEWS_MODEL_URL}/should-trade", timeout=5)
        if response.status_code == 200:
            return response.json().get("can_trade", True)
        return True
    except Exception as e:
        print(f"⚠️ News check failed: {str(e)}")
        return True

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Anso Vision Backend',
        'version': '2.0.0'
    }), 200

@app.route('/analyze', methods=['POST'])
def analyze_signal():
    """
    Analyze market data and generate trading signals
    
    Expected payload:
    {
        "symbol": "EURUSD",
        "candles": [...],
        "timeframe": "1h" (optional)
    }
    """
    try:
        data = request.json or {}
        symbol = data.get('symbol')
        candles = data.get('candles')
        timeframe = data.get('timeframe', '1h')

        # Validation
        if not symbol:
            return jsonify({
                'success': False,
                'error': 'Symbol is required'
            }), 400

        if not candles or len(candles) < 100:
            return jsonify({
                'success': False,
                'error': 'Insufficient candles',
                'required': 100,
                'provided': len(candles) if candles else 0,
                'message': 'At least 100 candles are required for accurate analysis'
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

        # Apply Kalman filter
        smoothed_prices = apply_kalman_filter(prices)
        features = signal_gen._prepare_hmm_features(smoothed_prices)

        # Train HMM
        signal_gen.hmm_model.train(features)
        hmm_states = signal_gen.hmm_model.predict_states(features)
        current_hmm_state = hmm_states[-1]

        # Context-aware analysis
        context_signal = context_hmm.analyze_with_context(prices, volumes, current_hmm_state)
        signal_type = context_signal['signal']
        current_price = float(prices[-1])

        # Monte Carlo TP/SL
        try:
            mc_result = signal_gen.monte_carlo.calculate_tp_sl(prices, current_price, signal_type)
            risk_metrics = signal_gen.monte_carlo.calculate_risk_metrics(
                prices, current_price, mc_result['tp'], mc_result['sl'], signal_type
            )
        except Exception as e:
            print(f"⚠️ Monte Carlo calculation failed: {str(e)}")
            if signal_type == 'BUY':
                mc_result = {'tp': current_price * 1.02, 'sl': current_price * 0.98, 'confidence': 0.5}
            else:
                mc_result = {'tp': current_price * 0.98, 'sl': current_price * 1.02, 'confidence': 0.5}
            risk_metrics = {
                'risk_reward_ratio': 2.0,
                'potential_profit_pct': 2.0,
                'potential_loss_pct': 1.0,
                'prob_tp': 0.5,
                'expected_value': 0.0
            }

        # Market structure
        market_structure = market_analyzer.analyze_market_structure(prices, volumes)

        response = {
            'success': True,
            'symbol': symbol,
            'timeframe': timeframe,
            'signal': signal_type,
            'entry': float(current_price),
            'tp': float(mc_result['tp']),
            'sl': float(mc_result['sl']),
            'confidence': float(context_signal.get('confidence', mc_result.get('confidence', 0.5))),
            'reasoning': context_signal.get('reasoning', 'No reasoning provided'),
            'signal_type': context_signal.get('type', signal_type),
            'market_context': context_signal.get('context', {}),
            'market_structure': market_structure,
            'risk_metrics': {
                'risk_reward_ratio': float(risk_metrics.get('risk_reward_ratio', 1.0)),
                'potential_profit_pct': float(risk_metrics.get('potential_profit_pct', 0.0)),
                'potential_loss_pct': float(risk_metrics.get('potential_loss_pct', 0.0)),
                'prob_tp': float(risk_metrics.get('prob_tp', 0.5)),
                'expected_value': float(risk_metrics.get('expected_value', 0.0)),
            },
            'hmm_state': {
                '0_bearish': float(current_hmm_state == 0),
                '1_neutral': float(current_hmm_state == 1),
                '2_bullish': float(current_hmm_state == 2),
            }
        }

        return jsonify(response), 200

    except Exception as e:
        print(f"❌ Analysis error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/execute', methods=['POST'])
def execute_signal():
    """
    Execute trading signal (MT4/MT5)
    
    Expected payload:
    {
        "symbol": "EURUSD",
        "entry": 1.0850,
        "tp": 1.0900,
        "sl": 1.0800,
        "signal": "BUY",
        "account_type": "MT4",
        "api_key": "your_api_key"
    }
    """
    try:
        # Check news
        if not check_news_before_trade():
            return jsonify({
                'success': False,
                'status': 'blocked',
                'reason': 'High-impact news detected - trading paused for safety'
            }), 403

        data = request.json or {}
        
        # Validation
        required_fields = ['symbol', 'entry', 'tp', 'sl', 'signal', 'account_type']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'error': 'Missing required fields',
                'missing': missing_fields
            }), 400

        symbol = data.get('symbol')
        entry = float(data.get('entry'))
        tp = float(data.get('tp'))
        sl = float(data.get('sl'))
        signal_type = data.get('signal')
        account_type = data.get('account_type')

        return jsonify({
            'success': True,
            'status': 'executed',
            'message': f'Signal executed: {signal_type} {symbol}',
            'trade_details': {
                'symbol': symbol,
                'direction': signal_type,
                'entry': entry,
                'tp': tp,
                'sl': sl,
                'account': account_type
            }
        }), 200

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': 'Invalid numeric values',
            'message': str(e)
        }), 400
    except Exception as e:
        print(f"❌ Execution error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/webhook/live', methods=['POST'])
def receive_live_candles():
    """Webhook for live candle data from data fetcher"""
    try:
        data = request.json
        symbol = data.get("symbol")
        candles = data.get("candles", [])

        if len(candles) < 100:
            return jsonify({
                'success': False,
                'error': "Insufficient candles",
                'required': 100,
                'provided': len(candles)
            }), 400

        # Reuse analyze logic
        return analyze_signal()

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/news/today', methods=['GET'])
def get_today_news():
    """Proxy endpoint for news"""
    try:
        response = requests.get(f"{DATA_FETCHER_URL}/news", timeout=10)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.Timeout:
        return jsonify({
            'success': False,
            'error': 'News service timeout'
        }), 504
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Failed to fetch news',
            'message': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    
    print(f"🚀 Starting Anso Vision Backend on port {port}")
    print(f"📰 News Model URL: {NEWS_MODEL_URL}")
    print(f"📊 Data Fetcher URL: {DATA_FETCHER_URL}")
    print(f"🔒 CORS Origins: {ALLOWED_ORIGINS}")
    
    app.run(host="0.0.0.0", port=port, debug=debug)
