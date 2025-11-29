"""
API SERVER INTEGRATED - The Central Deployment Point
This server is the single entry point for all signal requests. 
It delegates all complex logic (HMM, Context, Fusion, Risk) to the ModelManager.
"""
import os
import requests
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import threading
from typing import Dict

# --- CORE INTEGRATION ---
from model_manager import get_model_manager
from risk_manager import get_risk_manager

# Load environment variables
load_dotenv()

# --- FLASK SETUP ---
app = Flask(__name__)

ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')
CORS(app, resources={
    r"/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# --- MANAGER INITIALIZATION ---
model_manager = get_model_manager()
risk_manager = get_risk_manager()

DATA_FETCHER_URL = os.getenv('DATA_FETCHER_URL', 'https://anso-vision-data-fetcher.onrender.com')
NEWS_MODEL_URL = os.getenv('NEWS_MODEL_URL', 'https://anso-vision-news-model.onrender.com')

# ----------------------------------------------------------------------
# API ROUTES
# ----------------------------------------------------------------------

@app.route('/signal/<symbol>', methods=['POST'])
def generate_signal_route(symbol):
    data = request.json
    if not data or 'prices' not in data or 'volumes' not in data:
        return jsonify({'signal_type': 'WAIT', 'reasoning': 'Missing price or volume data in request body.'}), 400
    
    prices = np.array(data.get('prices', []))
    volumes = np.array(data.get('volumes', []))
    
    if len(prices) < 100 or len(volumes) < 100:
        return jsonify({'signal_type': 'WAIT', 'reasoning': 'Insufficient data (requires ~100 bars) for integrated HMM/Context analysis.'}), 400

    trade_allowed, news_reason = check_news_before_trade()
    if not trade_allowed:
        return jsonify({'signal_type': 'WAIT', 'reasoning': f'Trade Blocked by News Model: {news_reason}'}), 200

    allowed, risk_reason = risk_manager.should_allow_signal(symbol, 'UNKNOWN')
    if not allowed:
        return jsonify({'signal_type': 'WAIT', 'reasoning': f'Risk Manager Blocked: {risk_reason}'}), 200

    try:
        signal_result = model_manager.generate_signal(symbol, prices, volumes)
    except Exception as e:
        print(f"Error during signal generation for {symbol}: {e}")
        return jsonify({'signal_type': 'WAIT', 'reasoning': f'Internal Model Error: {str(e)}'}), 500

    final_signal_type = signal_result.get('signal_type', 'WAIT')
    if final_signal_type != 'WAIT':
        risk_manager.record_signal(symbol, final_signal_type, signal_result.get('confidence', 0.0))
        
    return jsonify(signal_result), 200


@app.route('/train/<symbol>', methods=['POST'])
def train_model_route(symbol):
    data = request.json
    prices = np.array(data.get('prices', []))
    
    if len(prices) < 100:
        return jsonify({'success': False, 'reasoning': 'Insufficient data for training.'}), 400
    
    def run_training():
        success = model_manager.train_model(symbol, prices, volumes=np.array(data.get('volumes', [])))
        if success:
            print(f"✅ Training for {symbol} succeeded.")
        else:
            print(f"❌ Training for {symbol} failed.")
    
    thread = threading.Thread(target=run_training)
    thread.start()
    
    return jsonify({'success': True, 'message': f'Training started for {symbol}. Check logs for completion.'}), 200


@app.route('/status/<symbol>', methods=['GET'])
def get_model_status(symbol):
    state = model_manager.get_model_state(symbol)
    if state:
        return jsonify({
            'symbol': symbol,
            'is_trained': state.is_trained,
            'last_trained': state.last_trained.isoformat() if state.last_trained else 'Never',
            'train_count': state.train_count,
            'needs_retraining': state.needs_retraining(),
        }), 200
    return jsonify({'symbol': symbol, 'is_trained': False, 'reasoning': 'Model not yet initialized.'}), 404


# --- Extra Endpoints from second file ---

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'Anso Vision Backend', 'version': '2.0.0'}), 200


@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_signal():
    """
    Main endpoint for frontend signal analysis
    Expects: { symbol, candles: [{timestamp, open, high, low, close, volume}], timeframe }
    Returns: Full signal analysis with BUY/SELL/WAIT
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
        
        if len(candles) < 200:
            return jsonify({
                'success': False,
                'error': f'Insufficient data: Need 200 candles for stable HMM training, got {len(candles)}',
                'symbol': symbol,
                'signal': 'HOLD',
                'signal_type': 'ERROR',
                'entry': 0,
                'tp': 0,
                'sl': 0,
                'confidence': 0,
                'reasoning': f'Insufficient data: HMM requires 200 candles for reliable predictions. Got {len(candles)}. Please fetch more historical data or use a smaller timeframe.',
                'market_context': 'N/A',
                'market_structure': 'N/A',
                'risk_metrics': {
                    'risk_reward_ratio': 0,
                    'potential_profit_pct': 0,
                    'potential_loss_pct': 0,
                    'prob_tp': 0,
                    'expected_value': 0
                }
            }), 400
        
        # Extract prices and volumes from candles
        prices = np.array([c.get('close', 0) for c in candles])
        volumes = np.array([c.get('volume', 0) for c in candles])
        
        # Check for invalid data
        if np.any(prices <= 0) or np.any(volumes < 0):
            return jsonify({
                'success': False,
                'error': 'Invalid candle data (zero or negative values)',
                'symbol': symbol,
                'signal': 'HOLD',
                'signal_type': 'ERROR'
            }), 400
        
        # Check news before trading
        trade_allowed, news_reason = check_news_before_trade()
        if not trade_allowed:
            return jsonify({
                'success': True,
                'symbol': symbol,
                'signal': 'HOLD',
                'signal_type': 'WAIT',
                'entry': float(prices[-1]),
                'tp': 0,
                'sl': 0,
                'confidence': 0,
                'reasoning': f'Trade Blocked by News: {news_reason}',
                'market_context': 'High-impact news detected',
                'market_structure': 'N/A',
                'timeframe': timeframe,
                'risk_metrics': {
                    'risk_reward_ratio': 0,
                    'potential_profit_pct': 0,
                    'potential_loss_pct': 0,
                    'prob_tp': 0,
                    'expected_value': 0
                }
            }), 200
        
        # Check risk manager
        allowed, risk_reason = risk_manager.should_allow_signal(symbol, 'UNKNOWN')
        if not allowed:
            return jsonify({
                'success': True,
                'symbol': symbol,
                'signal': 'HOLD',
                'signal_type': 'WAIT',
                'entry': float(prices[-1]),
                'tp': 0,
                'sl': 0,
                'confidence': 0,
                'reasoning': f'Risk Manager Blocked: {risk_reason}',
                'market_context': 'Risk limits exceeded',
                'market_structure': 'N/A',
                'timeframe': timeframe,
                'risk_metrics': {
                    'risk_reward_ratio': 0,
                    'potential_profit_pct': 0,
                    'potential_loss_pct': 0,
                    'prob_tp': 0,
                    'expected_value': 0
                }
            }), 200
        
        # Generate signal using model manager
        print(f"\n🔄 Calling model_manager.generate_signal()...")
        print(f"   Symbol: {symbol}")
        print(f"   Prices shape: {prices.shape}")
        print(f"   Volumes shape: {volumes.shape}")
        print(f"   Price range: {prices.min():.2f} - {prices.max():.2f}")
        
        signal_result = model_manager.generate_signal(symbol, prices, volumes)
        
        # model_manager.generate_signal() ALWAYS returns a dict, never None
        if signal_result is None:
            # This should NEVER happen with the new bulletproof model_manager
            print(f"❌ CRITICAL: generate_signal returned None (should be impossible)")
            return jsonify({
                'success': False,
                'error': 'Signal generation returned None (critical error)',
                'symbol': symbol,
                'signal': 'HOLD',
                'signal_type': 'ERROR',
                'entry': float(prices[-1]),
                'tp': 0,
                'sl': 0,
                'confidence': 0,
                'reasoning': 'Critical internal error: signal generator returned None',
                'market_context': 'N/A',
                'market_structure': 'N/A',
                'timeframe': timeframe,
                'risk_metrics': {
                    'risk_reward_ratio': 0,
                    'potential_profit_pct': 0,
                    'potential_loss_pct': 0,
                    'prob_tp': 0,
                    'expected_value': 0
                }
            }), 500
        
        # Check if signal indicates an error
        if signal_result.get('error', False):
            print(f"⚠️ Signal generation error: {signal_result.get('error_message', 'Unknown')}")
            return jsonify({
                'success': False,
                'error': signal_result.get('error_message', 'Signal generation failed'),
                'symbol': symbol,
                'signal': 'HOLD',
                'signal_type': 'ERROR',
                'entry': float(prices[-1]),
                'tp': 0,
                'sl': 0,
                'confidence': 0,
                'reasoning': signal_result.get('reasoning', 'Error during signal generation'),
                'market_context': 'Error',
                'market_structure': 'N/A',
                'timeframe': timeframe,
                'risk_metrics': signal_result.get('risk_metrics', {
                    'risk_reward_ratio': 0,
                    'potential_profit_pct': 0,
                    'potential_loss_pct': 0,
                    'prob_tp': 0,
                    'expected_value': 0
                })
            }), 400
        
        # Map signal_type to signal for frontend compatibility
        signal_type = signal_result.get('signal_type', 'WAIT')
        signal_map = {'BUY': 'BUY', 'SELL': 'SELL', 'WAIT': 'HOLD'}
        signal = signal_map.get(signal_type, 'HOLD')
        
        # Record signal if not WAIT
        if signal_type != 'WAIT':
            risk_manager.record_signal(symbol, signal_type, signal_result.get('confidence', 0.0))
        
        # Build response matching frontend expectations
        entry_value = signal_result.get('entry', prices[-1])
        tp_value = signal_result.get('tp', 0)
        sl_value = signal_result.get('sl', 0)
        
        # Ensure values are not None before converting to float
        entry_value = float(entry_value if entry_value is not None else prices[-1])
        tp_value = float(tp_value if tp_value is not None else 0)
        sl_value = float(sl_value if sl_value is not None else 0)
        
        response = {
            'success': True,
            'symbol': symbol,
            'signal': signal,
            'signal_type': signal_type,
            'entry': entry_value,
            'tp': tp_value,
            'sl': sl_value,
            'confidence': float(signal_result.get('confidence', 0)),
            'reasoning': signal_result.get('reasoning', 'No reasoning provided'),
            'market_context': signal_result.get('market_context', 'N/A'),
            'market_structure': signal_result.get('market_structure', 'N/A'),
            'timeframe': timeframe,
            'risk_metrics': signal_result.get('risk_metrics', {
                'risk_reward_ratio': 0,
                'potential_profit_pct': 0,
                'potential_loss_pct': 0,
                'prob_tp': 0,
                'expected_value': 0
            }),
            'hmm_state': {
                '0_bearish': 0.33,
                '1_neutral': 0.34,
                '2_bullish': 0.33
            }
        }
        
        print(f"✅ Signal generated for {symbol}: {signal_type}")
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Error in /analyze endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'Internal error: {str(e)}',
            'signal': 'HOLD',
            'signal_type': 'ERROR'
        }), 500


@app.route('/webhook/live', methods=['POST'])
def receive_live_candles():
    data = request.json
    symbol = data.get("symbol")
    candles = data.get("candles", [])
    if len(candles) < 100:
        return jsonify({'success': False, 'error': "Insufficient candles", 'required': 100, 'provided': len(candles)}), 400
    return analyze_signal()


@app.route('/news/today', methods=['GET'])
def get_today_news():
    try:
        response = requests.get(f"{DATA_FETCHER_URL}/news", timeout=10)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.Timeout:
        return jsonify({'success': False, 'error': 'News service timeout'}), 504
    except Exception as e:
        return jsonify({'success': False, 'error': 'Failed to fetch news', 'message': str(e)}), 500


# --- Helper ---
def check_news_before_trade() -> tuple[bool, str]:
    try:
        response = requests.get(f"{NEWS_MODEL_URL}/should-trade", timeout=3)
        if response.status_code == 200:
            news_data = response.json()
            if not news_data.get('should_trade', True):
                return False, news_data.get('reason', 'High-impact news detected.')
            return True, 'News check passed.'
        return True, 'News check API failure, proceeding with warning.'
    except requests.exceptions.RequestException:
        return True, 'News check unreachable, proceeding with warning.'


# --- Entry Point ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    print(f"🚀 Starting API Server Integrated on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
