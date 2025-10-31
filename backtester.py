# backtester.py

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
    """Runs a backtest of the signal generation logic."""
    print(f"Fetching historical data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    historical_prices = data['Close'].values.flatten()

    if len(historical_prices) == 0:
        print(f"No historical data fetched for {ticker}. Skipping backtest for this ticker.")
        return None, None, None, None

    # Implement a rolling window backtest
    LOOKBACK_WINDOW = 252  # e.g., 1 year of daily data

    # Initialize portfolio
    initial_capital = 10000.0
    portfolio_value = initial_capital
    shares_held = 0
    trades = []

    # Ensure enough data for the initial lookback window
    if len(historical_prices) < LOOKBACK_WINDOW:
        print(f"Not enough historical data for {ticker} to perform rolling window backtest with a lookback window of {LOOKBACK_WINDOW}.")
        return None, None, None, None

    for i in range(LOOKBACK_WINDOW, len(historical_prices) - 1):
        # Define the current training window
        train_data = historical_prices[i - LOOKBACK_WINDOW : i]
        
        # Re-initialize SignalGenerator for each rolling window to ensure a fresh HMM model
        signal_generator = SignalGenerator(n_hmm_components=n_hmm_components, covariance_type=covariance_type, random_state=random_state)
        
        # Train HMM on the current training window
        signal_generator.hmm_model.train(train_data.reshape(-1, 1))

        # Get the current price for signal generation and trading
        current_price = historical_prices[i]
        
        # Generate signal for the current data point using the trained HMM
        # We pass the current price data up to the current point for signal generation
        signal = signal_generator.generate_signals(historical_prices[:i+1])

        # Simple trading logic
        if signal and signal['entry'] is not None and signal['tp'] is not None and signal['sl'] is not None:
            if shares_held == 0 and current_price <= signal['entry']:
                shares_to_buy = int(portfolio_value / current_price)
                if shares_to_buy > 0:
                    shares_held += shares_to_buy
                    portfolio_value -= shares_to_buy * current_price
                    trades.append({'type': 'buy', 'price': current_price, 'shares': shares_to_buy, 'date': data.index[i]})
            elif shares_held > 0:
                if current_price >= signal['tp'] or current_price <= signal['sl']:
                    portfolio_value += shares_held * current_price
                    trades.append({'type': 'sell', 'price': current_price, 'shares': shares_held, 'date': data.index[i]})
                    shares_held = 0

    final_portfolio_value = portfolio_value + (shares_held * historical_prices[-1] if shares_held > 0 else 0)
    total_return = ((final_portfolio_value - initial_capital) / initial_capital) * 100

    # TODO: Implement comprehensive backtesting metrics (e.g., Sharpe ratio, drawdown)
    # TODO: Add visualization of trades and equity curve
    return initial_capital, final_portfolio_value, total_return, trades

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
                    initial_capital, final_portfolio_value, total_return, trades = run_backtest(
                        ticker, START_DATE, END_DATE, n_components, cov_type, DEFAULT_RANDOM_STATE
                    )
                    if initial_capital is not None:
                        print(f"Backtest Results for {ticker} (n_components={n_components}, covariance_type='{cov_type}'):")
                        print(f"  Initial Capital: ${initial_capital:.2f}")
                        print(f"  Final Portfolio Value: ${final_portfolio_value:.2f}")
                        print(f"  Total Return: {total_return:.2f}%")
                        print(f"  Number of Trades: {len(trades)}")
                    else:
                        print(f"Backtest for {ticker} with n_components={n_components}, covariance_type='{cov_type}' did not run or returned no data.")
                except Exception as e:
                    error_message = f"Error running backtest for {ticker} with n_components={n_components}, covariance_type='{cov_type}': {e}"
                    print(error_message)
                    with open("backtest_errors.log", "a") as error_f:
                        error_f.write(error_message + "\n")