def post_to_community_trading(symbol: str, signal: Dict):
    """✅ ENHANCED: Post signal to MT5 community trading platform with ALL FIELDS"""
    try:
        signal_type = signal.get('signal_type', 'WAIT')
        
        if signal_type == 'WAIT':
            logger.info(f"⏭️ Skipping WAIT signal for {symbol}")
            return
        
        # ✅ INCLUDE ALL FIELDS from signal
        payload = {
            "symbol": symbol,
            "action": signal_type,
            "entry": float(signal.get('entry', 0)),
            "tp": float(signal.get('tp', 0)),
            "sl": float(signal.get('sl', 0)),
            "volume": 0.01,
            "confidence": float(signal.get('confidence', 0)),
            "reasoning": signal.get('reasoning', 'No reasoning'),
            "limit_orders": signal.get('limit_orders', False),  # ✅ NEW
            "timeframe": "1h"
        }
        
        logger.info("\n" + "="*70)
        logger.info("🚀 POSTING SIGNAL TO MT5 BACKEND")
        logger.info("="*70)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Action: {signal_type}")
        logger.info(f"Entry: {payload['entry']:.4f}")
        logger.info(f"TP: {payload['tp']:.4f} | SL: {payload['sl']:.4f}")
        logger.info(f"Confidence: {payload['confidence']:.1%}")
        logger.info(f"Limit Orders: {payload['limit_orders']}")  # ✅ NEW
        logger.info(f"Target URL: {COMMUNITY_TRADING_URL}/api/signal")
        logger.info(f"API Key: {'*' * len(COMMUNITY_API_KEY)}")
        
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
            response_data = response.json()
            signal_id = response_data.get('signal_id', 'unknown')
            logger.info(f"✅ SIGNAL POSTED TO MT5 SUCCESSFULLY")
            logger.info(f"   Signal ID: {signal_id}")
            logger.info(f"   All fields saved including limit_orders")
            logger.info(f"   Response: {response.text[:150]}")
            logger.info("="*70 + "\n")
        elif response.status_code == 403:
            logger.error(f"❌ AUTHENTICATION FAILED: Invalid API key")
            logger.error(f"   Expected key: {COMMUNITY_API_KEY}")
        elif response.status_code == 404:
            logger.error(f"❌ ENDPOINT NOT FOUND: {COMMUNITY_TRADING_URL}/api/signal")
            logger.error(f"   Is MT5 backend running?")
        else:
            logger.error(f"⚠️ MT5 BACKEND ERROR: HTTP {response.status_code}")
            logger.error(f"   Response: {response.text[:200]}")
    
    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ CANNOT REACH MT5 BACKEND: {COMMUNITY_TRADING_URL}")
        logger.error(f"   Error: {e}")
    
    except requests.exceptions.Timeout:
        logger.error(f"❌ MT5 BACKEND TIMEOUT: Request took >5 seconds")
    
    except Exception as e:
        logger.error(f"❌ ERROR POSTING TO MT5: {type(e).__name__}: {e}", exc_info=True)
