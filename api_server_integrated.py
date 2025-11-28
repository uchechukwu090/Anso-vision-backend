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
from datetime import datetime
import json
import threading
from typing import Dict

# --- CORE INTEGRATION ---
# The API sees the intelligence through the ModelManager layer. 
from model_manager import get_model_manager
from risk_manager import get_risk_manager  # Import Layer 4 Safety

# Load environment variables (e.g., ALLOWED_ORIGINS, DATA_FETCHER_URL)
load_dotenv()

# --- FLASK SETUP ---
app = Flask(__name__)

# CORS Setup
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')
CORS(app, resources={
    r"/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# --- MANAGER INITIALIZATION (Singletons) ---
# This initializes the HMM and other models (via SignalGenerator) behind the scenes.
# It ensures they are trained only when necessary.
model_manager = get_model_manager()
risk_manager = get_risk_manager()

# Service URLs (Used for external checks like news)
DATA_FETCHER_URL = os.getenv('DATA_FETCHER_URL', 'https://anso-vision-data-fetcher.onrender.com')
NEWS_MODEL_URL = os.getenv('NEWS_MODEL_URL', 'https://anso-vision-news-model.onrender.com')


# ----------------------------------------------------------------------
# API ROUTES
# ----------------------------------------------------------------------

@app.route('/signal/<symbol>', methods=['POST'])
def generate_signal_route(symbol):
    """
    Main signal generation endpoint. 
    1. Fetches data. 
    2. Runs pre-trade checks (Risk Manager, News). 
    3. Calls ModelManager for the integrated signal.
    """
    # 1. Input Data Validation
    data = request.json
    if not data or 'prices' not in data or 'volumes' not in data:
        return jsonify({'signal_type': 'WAIT', 'reasoning': 'Missing price or volume data in request body.'}), 400
    
    prices = np.array(data.get('prices', []))
    volumes = np.array(data.get('volumes', []))
    
    if len(prices) < 100 or len(volumes) < 100:
        return jsonify({'signal_type': 'WAIT', 'reasoning': 'Insufficient data (requires ~100 bars) for integrated HMM/Context analysis.'}), 400

    # 2. Layer 5: Safety Gate Check (News)
    trade_allowed, news_reason = check_news_before_trade()
    if not trade_allowed:
        return jsonify({'signal_type': 'WAIT', 'reasoning': f'Trade Blocked by News Model: {news_reason}'}), 200

    # 3. Layer 5: Safety Gate Check (Rate Limiting, Max Signals, etc.)
    allowed, risk_reason = risk_manager.should_allow_signal(symbol, 'UNKNOWN')  # Check general limits
    if not allowed:
        return jsonify({'signal_type': 'WAIT', 'reasoning': f'Risk Manager Blocked: {risk_reason}'}), 200

    # 4. Layer 3/5: Generate Signal via ModelManager
    try:
        signal_result = model_manager.generate_signal(symbol, prices, volumes)
    except Exception as e:
        print(f"Error during signal generation for {symbol}: {e}")
        return jsonify({'signal_type': 'WAIT', 'reasoning': f'Internal Model Error: {str(e)}'}), 500

    # 5. Final Check & Record
    final_signal_type = signal_result.get('signal_type', 'WAIT')
    if final_signal_type != 'WAIT':
        risk_manager.record_signal(
            symbol,
            final_signal_type,
            signal_result.get('confidence', 0.0)
        )
        
    # 6. Return the final, validated result
    return jsonify(signal_result), 200


@app.route('/train/<symbol>', methods=['POST'])
def train_model_route(symbol):
    """Manually trigger model training for a symbol."""
    data = request.json
    prices = np.array(data.get('prices', []))
    
    if len(prices) < 100:
        return jsonify({'success': False, 'reasoning': 'Insufficient data for training.'}), 400
    
    # Run training asynchronously to not block the API request
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
    """Get the current training status of the model manager."""
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
