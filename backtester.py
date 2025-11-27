"""
REALISTIC Backtester with Slippage, Commissions, and Proper Validation
"""
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import json

from kalman_filter import apply_kalman_filter
from signal_generator import SignalGenerator
from model_manager import ModelManager


class RealisticBacktester:
    """
    Realistic backtesting with:
    - Slippage modeling
    - Commission/spread costs
    - Multiple market regimes
    - Proper risk metrics
    """
    
    def __init__(self, 
                 spread_pips: float = 2.0,
                 commission_pct: float = 0.001,
                 slippage_pips: float = 1.0):
        """
        Args:
            spread_pips: Typical spread in pips (2 for forex)
            commission_pct: Commission as percentage (0.001 = 0.1%)
            slippage_pips: Average slippage in pips
        """
        self.spread_pips = spread_pips
        self.commission_pct = commission_pct
        self.slippage_pips = slippage_pips
        
        self.model_manager = ModelManager(n_hmm_components=3, random_state=42)
        
    def calculate_costs(self, entry_price: float, signal_type: str) -> Tuple[float, float]:
        """
        Calculate realistic entry and exit prices with costs
        
        Returns:
            (actual_entry, cost_per_unit)
        """
        # Convert pips to price (assuming 4 decimal places for forex)
        pip_value = 0.0001
        spread = self.spread_pips * pip_value
        slippage = self.slippage_pips * pip_value
        
        if signal_type == 'BUY':
            # Buy at ask (entry_price + spread + slippage)
            actual_entry = entry_price + spread + slippage
        else:  # SELL
            # Sell at bid (entry_price - spread - slippage)
            actual_entry = entry_price - spread - slippage
        
        # Commission (both entry and exit)
        commission = entry_price * self.commission_pct * 2
        
        return actual_entry, commission
    
    def run_backtest(self, 
                    ticker: str,
                    start_date: str,
                    end_date: str,
                    initial_balance: float = 10000.0,
                    risk_per_trade_pct: float = 0.02) -> Dict:
        """
        Run comprehensive backtest
        
        Args:
            ticker: Symbol to backtest
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_balance: Starting capital
            risk_per_trade_pct: Risk percentage per trade (0.02 = 2%)
            
        Returns:
            dict: Complete backtest results
        """
        print(f"\n{'='*80}")
        print(f"📊 REALISTIC BACKTEST: {ticker}")
        print(f"{'='*80}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Balance: ${initial_balance:,.2f}")
        print(f"Risk per Trade: {risk_per_trade_pct*100:.1f}%")
        print(f"Spread: {self.spread_pips} pips | Commission: {self.commission_pct*100:.1f}% | Slippage: {self.slippage_pips} pips")
        print(f"{'='*80}\n")
        
        # Fetch data
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty or len(data) < 300:
                return {'error': f'Insufficient data ({len(data)} days)'}
            
            prices = data['Close'].values.flatten()
            volumes = data['Volume'].values.flatten() if 'Volume' in data else np.ones(len(prices))
            
        except Exception as e:
            return {'error': f'Data fetch failed: {str(e)}'}
        
        # Backtest parameters
        LOOKBACK_WINDOW = 250
        HOLD_PERIOD = 20
        MIN_RETRAIN_INTERVAL = 50
        
        # Initialize tracking
        balance = initial_balance
        trades = []
        equity_curve = [initial_balance]
        drawdowns = []
        
        last_train_idx = 0
        
        print("🔄 Running backtest...\n")
        
        for i in range(LOOKBACK_WINDOW, len(prices) - HOLD_PERIOD):
            # Training data
            train_prices = prices[i - LOOKBACK_WINDOW : i]
            
            # Retrain model periodically
            if i - last_train_idx >= MIN_RETRAIN_INTERVAL or last_train_idx == 0:
                try:
                    self.model_manager.train_model(ticker, train_prices, force=True)
                    last_train_idx = i
                except Exception as e:
                    continue
            
            # Generate signal
            try:
                signal = self.model_manager.generate_signal(
                    ticker, 
                    train_prices,
                    auto_train=False
                )
                
                if not signal or signal.get('entry') is None:
                    continue
                
                signal_type = signal.get('signal_type', 'UNKNOWN')
                if signal_type not in ['BUY', 'SELL']:
                    continue
                
                # Extract prices
                current_price = prices[i]
                entry_price = current_price  # Use actual market price
                tp_price = signal['tp']
                sl_price = signal['sl']
                
                # Calculate realistic costs
                actual_entry, commission = self.calculate_costs(entry_price, signal_type)
                
                # Calculate position size based on risk
                if signal_type == 'BUY':
                    risk_amount = actual_entry - sl_price
                else:
                    risk_amount = sl_price - actual_entry
                
                if risk_amount <= 0:
                    continue
                
                # Position size: risk only X% of capital
                position_size = (balance * risk_per_trade_pct) / risk_amount
                position_size = min(position_size, balance * 0.1 / actual_entry)  # Max 10% of balance
                
                # Execute trade simulation
                trade = {
                    'entry_idx': i,
                    'signal_type': signal_type,
                    'entry_price': actual_entry,
                    'raw_entry': entry_price,
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    'position_size': position_size,
                    'commission': commission * position_size,
                    'status': 'OPEN',
                    'confidence': signal.get('confidence', 0)
                }
                
                # Simulate holding period
                for j in range(i + 1, min(i + HOLD_PERIOD, len(prices))):
                    future_price = prices[j]
                    
                    if signal_type == 'BUY':
                        # Check TP/SL
                        if future_price >= tp_price:
                            trade['status'] = 'TP_HIT'
                            trade['exit_price'] = tp_price - self.slippage_pips * 0.0001
                            trade['exit_idx'] = j
                            profit = (trade['exit_price'] - actual_entry) * position_size
                            break
                        elif future_price <= sl_price:
                            trade['status'] = 'SL_HIT'
                            trade['exit_price'] = sl_price + self.slippage_pips * 0.0001
                            trade['exit_idx'] = j
                            profit = (trade['exit_price'] - actual_entry) * position_size
                            break
                    else:  # SELL
                        if future_price <= tp_price:
                            trade['status'] = 'TP_HIT'
                            trade['exit_price'] = tp_price + self.slippage_pips * 0.0001
                            trade['exit_idx'] = j
                            profit = (actual_entry - trade['exit_price']) * position_size
                            break
                        elif future_price >= sl_price:
                            trade['status'] = 'SL_HIT'
                            trade['exit_price'] = sl_price - self.slippage_pips * 0.0001
                            trade['exit_idx'] = j
                            profit = (actual_entry - trade['exit_price']) * position_size
                            break
                
                # If not closed, exit at end of period
                if trade['status'] == 'OPEN':
                    final_price = prices[min(i + HOLD_PERIOD - 1, len(prices) - 1)]
                    trade['exit_price'] = final_price
                    trade['status'] = 'TIME_EXIT'
                    
                    if signal_type == 'BUY':
                        profit = (final_price - actual_entry) * position_size
                    else:
                        profit = (actual_entry - final_price) * position_size
                
                # Subtract commission
                profit -= trade['commission']
                
                trade['profit'] = profit
                trade['return_pct'] = (profit / balance) * 100
                
                # Update balance
                balance += profit
                trades.append(trade)
                equity_curve.append(balance)
                
                # Calculate drawdown
                peak = max(equity_curve)
                drawdown = (peak - balance) / peak * 100 if peak > 0 else 0
                drawdowns.append(drawdown)
                
            except Exception as e:
                print(f"⚠️ Error at index {i}: {e}")
                continue
        
        # Calculate metrics
        if len(trades) == 0:
            return {'error': 'No trades executed'}
        
        results = self._calculate_metrics(trades, equity_curve, drawdowns, initial_balance)
        results['ticker'] = ticker
        results['period'] = f"{start_date} to {end_date}"
        results['initial_balance'] = initial_balance
        results['final_balance'] = balance
        
        self._print_results(results)
        
        return results
    
    def _calculate_metrics(self, trades: List[dict], equity_curve: List[float], 
                          drawdowns: List[float], initial_balance: float) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        winning_trades = [t for t in trades if t['profit'] > 0]
        losing_trades = [t for t in trades if t['profit'] <= 0]
        
        total_return = ((equity_curve[-1] - initial_balance) / initial_balance) * 100
        
        # Win rate
        win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0
        
        # Average profits/losses
        avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        total_profit = sum(t['profit'] for t in winning_trades)
        total_loss = abs(sum(t['profit'] for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        # Sharpe ratio
        returns = [t['return_pct'] for t in trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Max drawdown
        max_drawdown = max(drawdowns) if drawdowns else 0
        
        # Expectancy (average profit per trade)
        expectancy = np.mean([t['profit'] for t in trades])
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'expectancy': expectancy,
            'trades': trades
        }
    
    def _print_results(self, results: Dict):
        """Print formatted results"""
        print(f"\n{'='*80}")
        print(f"📈 BACKTEST RESULTS: {results['ticker']}")
        print(f"{'='*80}")
        print(f"Period: {results['period']}")
        print(f"Initial Balance: ${results['initial_balance']:,.2f}")
        print(f"Final Balance: ${results['final_balance']:,.2f}")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"\n--- Trade Statistics ---")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"Winning Trades: {results['winning_trades']}")
        print(f"Losing Trades: {results['losing_trades']}")
        print(f"\n--- Profit/Loss ---")
        print(f"Average Win: ${results['avg_win']:.2f}")
        print(f"Average Loss: ${results['avg_loss']:.2f}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Expectancy (avg per trade): ${results['expectancy']:.2f}")
        print(f"\n--- Risk Metrics ---")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    backtester = RealisticBacktester(
        spread_pips=2.0,
        commission_pct=0.001,
        slippage_pips=1.0
    )
    
    # Test on multiple symbols
    SYMBOLS = ["EURUSD=X", "BTC-USD"]
    
    for symbol in SYMBOLS:
        results = backtester.run_backtest(
            ticker=symbol,
            start_date="2023-01-01",
            end_date="2024-11-01",
            initial_balance=10000,
            risk_per_trade_pct=0.02
        )
