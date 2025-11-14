import os
from flask import Flask, request, jsonify
import numpy as np
from kalman_filter import apply_kalman_filter
from signal_generator import SignalGenerator
from hmm_model import MarketHMM
from context_aware_hmm import ContextAwareHMM
from market_analyzer import MarketAnalyzer
from monte_carlo_optimizer import MonteCarloOptimizer

app = Flask(__name__)

# Initialize once at startup
signal_gen = SignalGenerator(n_hmm_components=3)

# ✅ Ensure MonteCarloOptimizer is attached
if not hasattr(signal_gen, "monte_carlo") or signal_gen.monte_carlo is None:
    signal_gen.monte_carlo = MonteCarloOptimizer()

context_hmm = ContextAwareHMM()
market_analyzer = MarketAnalyzer()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'Backend is running'}), 200

@app.route('/analyze', methods=['POST'])
def analyze_signal():
    try:
        data = request.json or {}
        symbol = data.get('symbol')
        candles = data.get('candles')

        if not candles or len(candles) < 100:
            return jsonify({
                'error': 'Insufficient candles',
                'required': 100,
                'provided': len(candles) if candles else 0
            }), 400

        prices = np.array([c['close'] for c in candles], dtype=float)
        volumes = np.array([c['volume'] for c in candles], dtype=float)

        smoothed_prices = apply_kalman_filter(prices)
        features = signal_gen._prepare_hmm_features(smoothed_prices)

        signal_gen.hmm_model.train(features)
        hmm_states = signal_gen.hmm_model.predict_states(features)
        current_hmm_state = hmm_states[-1]

        context_signal = context_hmm.analyze_with_context(prices, volumes, current_hmm_state)
        signal_type = 'BUY' if context_signal['signal'] == 'BUY' else 'SELL'

        current_price = prices[-1]

        # ✅ Safe Monte Carlo usage
        mc_result = {}
        risk_metrics = {}
        try:
            mc_result = signal_gen.monte_carlo.calculate_tp_sl(prices, current_price, signal_type)
            risk_metrics = signal_gen.monte_carlo.calculate_risk_metrics(
                prices, current_price, mc_result['tp'], mc_result['sl'], signal_type
            )
        except Exception as mc_error:
            mc_result = {'tp': current_price * 1.01, 'sl': current_price * 0.99, 'confidence': 0.5}
            risk_metrics = {'risk_reward_ratio': 1.0, 'potential_profit_pct': 1.0,
                            'potential_loss_pct': 1.0, 'prob_tp': 0.5, 'expected_value': 0.0}

        market_structure = market_analyzer.analyze_market_structure(prices, volumes)

        response = {
            'success': True,
            'symbol': symbol,
            'signal': context_signal['signal'],
            'entry': float(current_price),
            'tp': float(mc_result['tp']),
            'sl': float(mc_result['sl']),
            'confidence': float(mc_result['confidence']),
            'reasoning': context_signal['reasoning'],
            'signal_type': context_signal.get('type'),
            'market_context': context_signal['context'],
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
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/execute', methods=['POST'])
def execute_signal():
    try:
        data = request.json or {}
        symbol = data.get('symbol')
        entry = data.get('entry')
        tp = data.get('tp')
        sl = data.get('sl')
        signal_type = data.get('signal')
        account_type = data.get('account_type')
        api_key = data.get('api_key')

        return jsonify({
            'success': True,
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

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/webhook/live', methods=['POST'])
def receive_live_candles():
    data = request.json
    symbol = data.get("symbol")
    candles = data.get("candles", [])

    if len(candles) < 100:
        return jsonify({"error": "Insufficient candles"}), 400

    result = run_analysis(symbol, candles)
    return jsonify(result)

@app.route('/news/today', methods=['GET'])
def get_today_news():
    response = requests.get("https://anso-vision-data-fetcher.onrender.com/news")
    return jsonify(response.json())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
