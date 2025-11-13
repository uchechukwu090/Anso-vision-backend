# backtester.py
"""
Backtester for Anso Vision trading signals.

Key insight: This system doesn't trade directly - it sends signals to capital.com broker.
Therefore, we backtest using historical price data only (no trade logs needed).
The system validates signal quality by simulating entry/exit based on signal levels.
"""

import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime

from kalman_filter import apply_kalman_filter
from hmm_model import MarketHMM
from wavelet_analysis import apply_wavelet_decomposition, reconstruct_wavelet_signal, denoise_signal_with_wavelets
from signal_generator import SignalGenerator


def time_series_split(data, n_splits, train_size, validation_size, embargo_size):
    """
    Generates time series splits for cross-validation with purging and embargoing.

    Args:
        data (pd.Series or np.array): The time series data.
        n_splits (int): Number of splits.
        train_size (int): Size of the training set for each split.
        validation_size (int): Size of the validation set for each split.
        embargo_size (int): Size of the embargo period between training and validation sets.

    Yields:
        tuple: (train_indices, validation_indices) for each split.
    """
    n_samples = len(data)
    if train_size + validation_size + embargo_size > n_samples:
        raise ValueError("Combined train, validation, and embargo sizes exceed total data length.")

    for i in range(n_splits):
        # Calculate the start and end indices for the current split
        # The splits will move forward in time
        start_index = i * (validation_size + embargo_size)
        
        train_end = start_index + train_size
        validation_start = train_end + embargo_size
        validation_end = validation_start + validation_size

        if validation_end > n_samples:
            break

        train_indices = list(range(start_index, train_end))
        validation_indices = list(range(validation_start, validation_end))

        yield train_indices, validation_indices


def run_backtest(ticker, start_date, end_date, n_hmm_components, covariance_type, random_state):
    """
    Runs a backtest of the signal generation logic using historical price data.
    
    Strategy:
    1. For each trading day, generate a BUY/SELL/WAIT signal
    2. Track if price reaches TP or SL within the next N periods
    3. Calculate win rate, avg profit/loss, and Sharpe ratio
    
    No trade logs needed - we simulate trade execution based on signal levels.
    """
    print(f"Fetching historical data for {ticker}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty or len(data) < 100:
            print(f"❌ Insufficient data for {ticker}. Got {len(data)} days, need at least 100.")
            return None, None, None, None, None
        
        historical_prices = data['Close'].values.flatten()
        
    except Exception as e:
        print(f"❌ Error fetching {ticker}: {str(e)}")
        return None, None, None, None, None

    # Implement a rolling window backtest
    LOOKBACK_WINDOW = 100  # Use 100 days for signal generation
    TRADE_HOLD_PERIOD = 20  # How many periods to hold the trade

    # Initialize metrics
    trades_executed = []
    signals_generated = 0
    winning_trades = 0
    losing_trades = 0
    total_profit = 0.0

    if len(historical_prices) < LOOKBACK_WINDOW + TRADE_HOLD_PERIOD:
        print(f"❌ Not enough data. Need {LOOKBACK_WINDOW + TRADE_HOLD_PERIOD}, have {len(historical_prices)}")
        return None, None, None, None, None

    for i in range(LOOKBACK_WINDOW, len(historical_prices) - TRADE_HOLD_PERIOD):
        # Define the current training window
        train_data = historical_prices[i - LOOKBACK_WINDOW : i]
        
        # Re-initialize SignalGenerator for each rolling window
        try:
            signal_generator = SignalGenerator(
                n_hmm_components=n_hmm_components, 
                covariance_type=covariance_type, 
                random_state=random_state
            )
            
            # Train HMM on features, not raw prices
            hmm_features = signal_generator._prepare_hmm_features(train_data)
            
            if len(hmm_features) < signal_generator.hmm_model.n_components:
                continue
            
            signal_generator.hmm_model.train(hmm_features)
            
            # Generate signal for the current data point
            signal = signal_generator.generate_signals(train_data)
            
            if not signal or signal['entry'] is None:
                continue
            
            signals_generated += 1
            current_price = historical_prices[i]
            entry_price = signal['entry']
            tp_price = signal['tp']
            sl_price = signal['sl']
            signal_type = signal.get('signal_type', 'UNKNOWN')
            
            # Simulate trade execution: check if price reaches TP or SL within hold period
            trade_result = {
                'entry_idx': i,
                'signal_type': signal_type,
                'entry_price': entry_price,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'status': 'OPEN'
            }
            
            # Check future prices within hold period
            for j in range(i + 1, min(i + TRADE_HOLD_PERIOD, len(historical_prices))):
                future_price = historical_prices[j]
                
                if signal_type == 'BUY':
                    if future_price >= tp_price:
                        trade_result['status'] = 'TP_HIT'
                        trade_result['exit_price'] = tp_price
                        trade_result['exit_idx'] = j
                        profit = tp_price - entry_price
                        winning_trades += 1
                        break
                    elif future_price <= sl_price:
                        trade_result['status'] = 'SL_HIT'
                        trade_result['exit_price'] = sl_price
                        trade_result['exit_idx'] = j
                        profit = sl_price - entry_price
                        losing_trades += 1
                        break
                
                elif signal_type == 'SELL':
                    if future_price <= tp_price:
                        trade_result['status'] = 'TP_HIT'
                        trade_result['exit_price'] = tp_price
                        trade_result['exit_idx'] = j
                        profit = entry_price - tp_price
                        winning_trades += 1
                        break
                    elif future_price >= sl_price:
                        trade_result['status'] = 'SL_HIT'
                        trade_result['exit_price'] = sl_price
                        trade_result['exit_idx'] = j
                        profit = entry_price - sl_price
                        losing_trades += 1
                        break
            
            # If trade didn't close, calculate P/L at end of hold period
            if trade_result['status'] == 'OPEN':
                final_price = historical_prices[min(i + TRADE_HOLD_PERIOD - 1, len(historical_prices) - 1)]
                if signal_type == 'BUY':
                    profit = final_price - entry_price
                    if profit > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
                else:
                    profit = entry_price - final_price
                    if profit > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
                
                trade_result['exit_price'] = final_price
                trade_result['status'] = 'CLOSED_EOH'
            
            trade_result['profit'] = profit
            trades_executed.append(trade_result)
            total_profit += profit
        
        except Exception as e:
            print(f"⚠️ Error at index {i}: {str(e)}")
            continue

    # Calculate metrics
    if signals_generated == 0:
        print(f"❌ No signals generated for {ticker}")
        return None, None, None, None, None
    
    win_rate = (winning_trades / (winning_trades + losing_trades) * 100) if (winning_trades + losing_trades) > 0 else 0
    avg_profit = total_profit / signals_generated if signals_generated > 0 else 0
    
    # Calculate Sharpe ratio from trade returns
    if len(trades_executed) > 1:
        trade_returns = [t['profit'] / t['entry_price'] for t in trades_executed]
        sharpe_ratio = (np.mean(trade_returns) / np.std(trade_returns)) * np.sqrt(252) if np.std(trade_returns) > 0 else 0
    else:
        sharpe_ratio = 0.0

    print(f"\n✅ Backtest Results for {ticker}:")
    print(f"   Signals Generated: {signals_generated}")
    print(f"   Trades Completed: {len(trades_executed)}")
    print(f"   Win Rate: {win_rate:.2f}%")
    print(f"   Avg Profit per Signal: ${avg_profit:.2f}")
    print(f"   Total Profit: ${total_profit:.2f}")
    print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
    
    return signals_generated, win_rate, avg_profit, total_profit, sharpe_ratio


if __name__ == "__main__":
    # Example usage:
    TICKERS = ["USDJPY=X", "EURUSD=X", "XAUUSD=X", "BTC-USD"]
    START_DATE = "2023-01-01"
    END_DATE = "2024-01-01"

    # Define lists of HMM parameters to experiment with
    N_HMM_COMPONENTS_OPTIONS = [2, 3, 4, 5]
    COVARIANCE_TYPE_OPTIONS = ['diag', 'full'] # 'spherical', 'tied' can also be added

    DEFAULT_RANDOM_STATE = 42 # Added for reproducibility

    print(f"\n--- Starting backtests for individual tickers ---")
    for ticker in TICKERS:
        for n_components in N_HMM_COMPONENTS_OPTIONS:
            for cov_type in COVARIANCE_TYPE_OPTIONS:
                print(f"\n--- Running backtest for {ticker} with n_components={n_components}, covariance_type='{cov_type}' ---")
                try:
                    signals_gen, win_rate, avg_profit, total_profit, sharpe = run_backtest(
                        ticker, START_DATE, END_DATE, n_components, cov_type, DEFAULT_RANDOM_STATE
                    )
                    if signals_gen is not None:
                        print(f"✅ Backtest completed successfully")
                    else:
                        print(f"⚠️  Backtest for {ticker} did not complete")
                except Exception as e:
                    error_message = f"Error running backtest for {ticker}: {e}"
                    print(error_message)
                    with open("backtest_errors.log", "a") as error_f:
                        error_f.write(error_message + "\n")