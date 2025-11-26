"""
WebSocket Manager for Real-Time Market Data
Connects to multiple exchanges and maintains live candle buffers
"""
import json
import threading
import time
from collections import deque
from datetime import datetime
import websocket
import numpy as np
from typing import Dict, Callable, List

class CandleBuffer:
    """Thread-safe buffer for storing candles"""
    def __init__(self, symbol: str, max_size: int = 500):
        self.symbol = symbol
        self.candles = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.last_update = None
        
    def add_candle(self, candle: dict):
        """Add a new candle to the buffer"""
        with self.lock:
            # Ensure candle has required fields
            required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if all(field in candle for field in required_fields):
                self.candles.append(candle)
                self.last_update = datetime.now()
                return True
            return False
    
    def get_candles(self, n: int = None) -> List[dict]:
        """Get last N candles (or all if N is None)"""
        with self.lock:
            if n is None:
                return list(self.candles)
            return list(self.candles)[-n:]
    
    def get_prices(self, n: int = None) -> np.ndarray:
        """Get close prices as numpy array"""
        candles = self.get_candles(n)
        return np.array([c['close'] for c in candles])
    
    def get_volumes(self, n: int = None) -> np.ndarray:
        """Get volumes as numpy array"""
        candles = self.get_candles(n)
        return np.array([c['volume'] for c in candles])
    
    def is_ready(self, min_candles: int = 250) -> bool:
        """Check if buffer has enough data"""
        with self.lock:
            return len(self.candles) >= min_candles


class WebSocketManager:
    """
    Manages WebSocket connections to exchanges and maintains candle buffers
    """
    def __init__(self):
        self.buffers: Dict[str, CandleBuffer] = {}
        self.connections: Dict[str, websocket.WebSocketApp] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        self.running = False
        self.threads = []
        
    def add_symbol(self, symbol: str, exchange: str = "binance", timeframe: str = "1m"):
        """Add a symbol to monitor"""
        if symbol not in self.buffers:
            self.buffers[symbol] = CandleBuffer(symbol)
            self.callbacks[symbol] = []
            print(f"✅ Added buffer for {symbol}")
    
    def register_callback(self, symbol: str, callback: Callable):
        """Register callback to be called when new candle arrives"""
        if symbol in self.callbacks:
            self.callbacks[symbol].append(callback)
    
    def start_binance_stream(self, symbol: str, timeframe: str = "1m"):
        """Start Binance WebSocket stream"""
        # Convert symbol format (BTCUSD -> btcusdt)
        ws_symbol = symbol.lower().replace('usd', 'usdt')
        ws_url = f"wss://stream.binance.com:9443/ws/{ws_symbol}@kline_{timeframe}"
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'k' in data:
                    kline = data['k']
                    candle = {
                        'timestamp': kline['t'],
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v']),
                        'is_closed': kline['x']
                    }
                    
                    # Only add completed candles to buffer
                    if candle['is_closed']:
                        if self.buffers[symbol].add_candle(candle):
                            # Trigger callbacks
                            for callback in self.callbacks.get(symbol, []):
                                try:
                                    callback(symbol, candle)
                                except Exception as e:
                                    print(f"⚠️ Callback error for {symbol}: {e}")
            except Exception as e:
                print(f"❌ Error processing message for {symbol}: {e}")
        
        def on_error(ws, error):
            print(f"❌ WebSocket error for {symbol}: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print(f"⚠️ WebSocket closed for {symbol}")
            if self.running:
                print(f"🔄 Reconnecting {symbol}...")
                time.sleep(5)
                self.start_binance_stream(symbol, timeframe)
        
        def on_open(ws):
            print(f"✅ WebSocket connected for {symbol}")
        
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        self.connections[symbol] = ws
        
        # Run in separate thread
        thread = threading.Thread(target=ws.run_forever)
        thread.daemon = True
        thread.start()
        self.threads.append(thread)
    
    def start(self, symbols: List[str], exchange: str = "binance", timeframe: str = "1m"):
        """Start monitoring multiple symbols"""
        self.running = True
        for symbol in symbols:
            self.add_symbol(symbol, exchange, timeframe)
            if exchange == "binance":
                self.start_binance_stream(symbol, timeframe)
        
        print(f"🚀 WebSocket Manager started for {len(symbols)} symbols")
    
    def stop(self):
        """Stop all WebSocket connections"""
        self.running = False
        for symbol, ws in self.connections.items():
            try:
                ws.close()
            except:
                pass
        print("🛑 WebSocket Manager stopped")
    
    def get_buffer(self, symbol: str) -> CandleBuffer:
        """Get buffer for a symbol"""
        return self.buffers.get(symbol)
    
    def is_ready(self, symbol: str, min_candles: int = 250) -> bool:
        """Check if symbol has enough data"""
        buffer = self.get_buffer(symbol)
        return buffer and buffer.is_ready(min_candles)


# Simulated WebSocket for testing (when exchange is unavailable)
class SimulatedWebSocketManager(WebSocketManager):
    """Simulated WebSocket for testing without real exchange connection"""
    
    def start_simulated_stream(self, symbol: str):
        """Generate fake candles for testing"""
        def generate_candles():
            base_price = 50000 if 'BTC' in symbol else 1.1
            while self.running:
                # Generate realistic fake candle
                volatility = 0.001
                price_change = np.random.normal(0, volatility)
                base_price *= (1 + price_change)
                
                candle = {
                    'timestamp': int(time.time() * 1000),
                    'open': base_price * (1 + np.random.normal(0, volatility/2)),
                    'high': base_price * (1 + abs(np.random.normal(0, volatility))),
                    'low': base_price * (1 - abs(np.random.normal(0, volatility))),
                    'close': base_price,
                    'volume': abs(np.random.normal(1000, 200)),
                    'is_closed': True
                }
                
                if self.buffers[symbol].add_candle(candle):
                    for callback in self.callbacks.get(symbol, []):
                        try:
                            callback(symbol, candle)
                        except Exception as e:
                            print(f"⚠️ Callback error: {e}")
                
                time.sleep(1)  # 1 second per candle (for testing)
        
        thread = threading.Thread(target=generate_candles)
        thread.daemon = True
        thread.start()
        self.threads.append(thread)
    
    def start(self, symbols: List[str], **kwargs):
        """Start simulated streams"""
        self.running = True
        for symbol in symbols:
            self.add_symbol(symbol)
            self.start_simulated_stream(symbol)
        print(f"🧪 Simulated WebSocket started for {len(symbols)} symbols")


if __name__ == '__main__':
    # Test the WebSocket manager
    def on_new_candle(symbol, candle):
        print(f"📊 {symbol}: Close=${candle['close']:.2f}, Volume={candle['volume']:.0f}")
    
    # Use simulated for testing
    ws_manager = SimulatedWebSocketManager()
    ws_manager.add_symbol("BTCUSD")
    ws_manager.register_callback("BTCUSD", on_new_candle)
    ws_manager.start(["BTCUSD"])
    
    # Wait and check buffer
    time.sleep(10)
    buffer = ws_manager.get_buffer("BTCUSD")
    if buffer:
        print(f"\n✅ Buffer has {len(buffer.candles)} candles")
        prices = buffer.get_prices()
        print(f"📈 Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    
    ws_manager.stop()
