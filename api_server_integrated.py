# API SERVER INTEGRATED - Robust sender + verification endpoint
# Put this file at the repo root (replace existing api_server_integrated.py)

import os
import requests
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import threading
import json
from typing import Dict, Set, Optional
from datetime import datetime
import logging

load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

DATA_FETCHER_URL = os.getenv('DATA_FETCHER_URL', 'https://anso-vision-data-fetcher.onrender.com')
NEWS_MODEL_URL = os.getenv('NEWS_MODEL_URL', 'https://anso-vision-news-model.onrender.com')
COMMUNITY_TRADING_URL = os.getenv('COMMUNITY_TRADING_URL', 'https://ansorade-backend.onrender.com')
COMMUNITY_API_KEY = os.getenv('COMMUNITY_API_KEY', 'Mr.creative090')

logger.info("Posting signals to: %s", COMMUNITY_TRADING_URL)
logger.info("Using API key: %s", '*' * len(COMMUNITY_API_KEY or ''))

def canonicalize_signal(data: Dict) -> Dict:
    # Map field names to what Ansorade expects
    s = {
        'symbol': data.get('symbol'),
        'action': data.get('action') or data.get('signal_type'),
        'entry': data.get('entry'),
        'tp': data.get('tp'),
        'sl': data.get('sl'),
        'volume': data.get('volume', 0.01),
        'confidence': data.get('confidence', 0.0),
        'limit_orders': data.get('limit_orders', False),
        'reasoning': data.get('reasoning', ''),
        'timeframe': data.get('timeframe', '1h'),
        'signal_id': data.get('signal_id') or f"sig_{int(time.time())}"
    }
    return s

def post_to_community_trading(signal: Dict, max_retries: int = 3, backoff: float = 1.0) -> Optional[Dict]:
    url = COMMUNITY_TRADING_URL.rstrip('/') + '/api/signal'
    headers = {'Content-Type': 'application/json', 'X-API-Key': COMMUNITY_API_KEY}
    payload = canonicalize_signal(signal)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("POST attempt %d -> %s ; payload keys: %s", attempt, url, list(payload.keys()))
            r = requests.post(url, json=payload, headers=headers, timeout=8)
            logger.info("Response %s: %.300s", r.status_code, r.text)
            if r.status_code in (200, 201):
                try:
                    return r.json()
                except Exception:
                    return {'status_code': r.status_code, 'text': r.text}
            if r.status_code in (500, 502, 503, 504):
                time.sleep(backoff * attempt)
                continue
            # non-retriable (400/401/403/422 etc.)
            logger.error("Non-retriable response: %s %s", r.status_code, r.text)
            return None
        except requests.exceptions.RequestException as e:
            logger.warning("RequestException (attempt %d): %s", attempt, e)
            time.sleep(backoff * attempt)
    logger.error("Exceeded retries posting signal")
    return None

@app.route('/generate-signal', methods=['POST'])
def generate_signal():
    data = request.get_json() or {}
    try:
        signal = canonicalize_signal(data)
        result = post_to_community_trading(signal)
        if not result:
            return jsonify({'status': 'error', 'message': 'Failed to deliver signal'}), 502
        return jsonify({'status': 'ok', 'sent': result}), 200
    except Exception as e:
        logger.exception("generate_signal failed")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/send-test-signal', methods=['POST'])
def send_test_signal():
    test_signal = {
        'symbol': 'EURUSD',
        'action': 'BUY',
        'entry': 1.0850,
        'tp': 1.0900,
        'sl': 1.0800,
        'volume': 0.01,
        'confidence': 0.9,
        'limit_orders': False,
        'reasoning': 'integration test - send-test-signal',
        'timeframe': '1h',
        'signal_id': f"test_{int(time.time())}"
    }
    sent = post_to_community_trading(test_signal)
    if not sent:
        return jsonify({'status': 'error', 'message': 'Failed to POST to trading backend'}), 502

    # verify pending
    verify_url = COMMUNITY_TRADING_URL.rstrip('/') + '/api/signals/pending'
    headers = {'X-API-Key': COMMUNITY_API_KEY}
    try:
        r = requests.get(verify_url, headers=headers, timeout=6)
        logger.info("Verify pending response: %s", r.status_code)
        if r.status_code != 200:
            return jsonify({'status': 'warning', 'message': 'Posted but could not fetch pending signals', 'post_result': sent}), 202
        pending = r.json()
        found = any(
            (s.get('signal_id') == test_signal['signal_id']) or
            (s.get('symbol') == test_signal['symbol'] and s.get('action') == test_signal['action'] and abs(float(s.get('entry') or 0) - test_signal['entry']) < 0.01)
            for s in (pending or [])
        )
        return jsonify({'status': 'ok', 'posted': sent, 'found_in_pending': found, 'pending_count': len(pending)}), 200
    except Exception as e:
        logger.exception("Verification GET failed")
        return jsonify({'status': 'ok', 'posted': sent, 'verification_error': str(e)}), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
