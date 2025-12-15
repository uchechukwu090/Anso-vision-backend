"""
✅ CLEAN MT5 SIGNAL POSTING
BUY/SELL only - no 10 signal types
Whipsaw risk included for MT5 awareness
"""

def post_to_mt5_trading(symbol: str, signal: Dict):
    """
    ✅ CLEAN: Post BUY/SELL signals to MT5
    MT5 only understands: BUY, SELL, PLACE_ORDER
    """
    try:
        signal_type = signal.get('signal_type', 'WAIT')
        
        # Only send BUY/SELL, never WAIT
        if signal_type not in ['BUY', 'SELL']:
            print(f"⏭️ Skipping {signal_type} signal (MT5 needs BUY/SELL only)")
            return
        
        # ✅ SIMPLE PAYLOAD FOR MT5
        payload = {
            "symbol": symbol,
            "action": signal_type,  # BUY or SELL
            "entry": float(signal.get('entry', 0)),
            "tp": float(signal.get('tp', 0)),
            "sl": float(signal.get('sl', 0)),
            "volume": 0.01,  # Default position size
            "confidence": float(signal.get('confidence', 0)),
            "whipsaw_risk": signal.get('whipsaw_risk', 'LOW'),  # Risk awareness
            "order_type": "MARKET"  # MT5 market order
        }
        
        print(f"\n🚀 MT5 SIGNAL PAYLOAD:")
        print(f"   Symbol: {symbol}")
        print(f"   Action: {signal_type}")
        print(f"   Entry: {payload['entry']:.4f}")
        print(f"   TP: {payload['tp']:.4f} | SL: {payload['sl']:.4f}")
        print(f"   Confidence: {payload['confidence']:.1%}")
        print(f"   Whipsaw Risk: {payload['whipsaw_risk']}")
        
        # POST to MT5 backend
        response = requests.post(
            f"{COMMUNITY_TRADING_URL}/api/signal",
            json=payload,
            headers={
                "X-API-Key": COMMUNITY_API_KEY,
                "Content-Type": "application/json"
            },
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            signal_id = result.get('signal_id', 'unknown')
            print(f"✅ MT5 ACCEPTED: Signal ID {signal_id}")
            return True
        else:
            print(f"❌ MT5 ERROR: {response.status_code} - {response.text[:100]}")
            return False
    
    except Exception as e:
        print(f"❌ CANNOT POST TO MT5: {e}")
        return False


# Update api_server to call this instead:
# In candle_complete() after signal generation:
#
# if signal_result.get('signal_type') != 'WAIT':
#     post_to_mt5_trading(symbol, signal_result)
#
# In analyze() endpoint:
#
# if signal_result.get('signal_type') != 'WAIT':
#     post_to_mt5_trading(symbol, signal_result)
