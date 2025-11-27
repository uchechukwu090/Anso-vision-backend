import numpy as np
from kalman_filter import apply_kalman_filter
from hmm_model import MarketHMM
from wavelet_analysis import apply_wavelet_decomposition, denoise_signal_with_wavelets
from monte_carlo_optimizer import MonteCarloOptimizer

class SignalGenerator:
    def __init__(self, n_hmm_components=3, wavelet_level=4, covariance_type='diag', random_state=None, n_monte_carlo_sims=10000):
        self.kalman_filter = None  # Kalman filter will be initialized with data
        self.hmm_model = MarketHMM(n_components=n_hmm_components, covariance_type=covariance_type, random_state=random_state)
        self.wavelet_level = wavelet_level
        self.monte_carlo = MonteCarloOptimizer(n_simulations=n_monte_carlo_sims, confidence_level=0.95)

    def _prepare_hmm_features(self, smoothed_data):
        """
        Prepares features for the HMM from smoothed data.
        This is a placeholder and should be expanded with more relevant features.
        For now, it uses simple price changes.

        Args:
            smoothed_data (np.array): 1D array of smoothed price data.

        Returns:
            np.array: 2D array of features for HMM.
        """
        # Calculate log returns
        log_returns = np.diff(np.log(smoothed_data), axis=0)

        # Calculate historical volatility (e.g., 10-period rolling standard deviation of log returns)
        # Pad with NaNs for the first few periods where volatility cannot be calculated
        volatility = np.zeros_like(log_returns)
        window = 10  # Example window size for volatility
        for i in range(window - 1, len(log_returns)):
            volatility[i] = np.std(log_returns[i - window + 1:i + 1])

        # Combine features. Ensure they have the same length.
        # We need to handle the length difference due to np.diff.
        # If smoothed_data has N points, log_returns and volatility will have N-1 points.
        # For HMM, we need (N-1, n_features)
        features = np.vstack([log_returns, volatility]).T
        
        # Handle potential NaN or inf values that might arise from calculations
        features[~np.isfinite(features)] = 0  # Replace non-finite values with 0 or a suitable replacement

        return features

    def generate_signals(self, raw_price_data):
        """
        Generates trading signals (entry, TP, SL) based on combined analysis.

        Args:
            raw_price_data (np.array): A 1D numpy array of raw price data.

        Returns:
            dict: A dictionary containing generated signals.
        """
        if not isinstance(raw_price_data, np.ndarray) or raw_price_data.ndim != 1:
            raise ValueError("Input raw_price_data must be a 1D numpy array.")
        
        if len(raw_price_data) < 100:
            return {
                "entry": None,
                "tp": None,
                "sl": None,
                "signal_type": "WAIT",
                "market_context": "Insufficient Data",
                "reasoning": f"Need at least 100 candles, got {len(raw_price_data)}"
            }

        # Use rolling window for HMM training (last 250 candles max)
        MAX_ROLLING_WINDOW = 250
        if len(raw_price_data) > MAX_ROLLING_WINDOW:
            raw_price_data = raw_price_data[-MAX_ROLLING_WINDOW:]

        # 1. Kalman Filter for smoothing
        # print("Applying Kalman Filter for smoothing...") # Suppress intermediate print
        smoothed_data = apply_kalman_filter(raw_price_data)

        # 2. Feature Engineering for HMM
        # print("Preparing HMM features...") # Suppress intermediate print
        hmm_features = self._prepare_hmm_features(smoothed_data)

        # Ensure enough data for HMM training/prediction
        if len(hmm_features) < self.hmm_model.n_components:
            # print("Not enough data for HMM. Skipping HMM and signal generation.") # Keep this for debugging if needed
            return {
                "entry": None,
                "tp": None,
                "sl": None,
                "signal_type": "WAIT",
                "market_context": "Insufficient Data",
                "reasoning": f"Need {self.hmm_model.n_components} features, got {len(hmm_features)}"
            }

        # 3. HMM State Prediction
        # print("Training and predicting HMM states...") # Suppress intermediate print
        # Train HMM if not already trained or needs retraining
        if not hasattr(self.hmm_model.model, 'transmat_') or len(hmm_features) > 200:
            self.hmm_model.train(hmm_features)
        predicted_states = self.hmm_model.predict_states(hmm_features)
        state_probabilities = self.hmm_model.get_state_probabilities(hmm_features)

        # 4. Wavelet Refinement (example: denoise state probabilities or other relevant data)
        # print("Applying Wavelet Analysis for refinement...") # Suppress intermediate print
        # For demonstration, let's denoise one of the state probability series
        # In a real scenario, you might apply wavelets to price series, volume, etc.
        if state_probabilities.shape[1] > 0:
            # Denoise the probability of the most likely state (or a specific state)
            most_likely_state_prob = state_probabilities[:, predicted_states[-1]]
            refined_prob = denoise_signal_with_wavelets(most_likely_state_prob, level=self.wavelet_level)
        else:
            refined_prob = np.array([])

        # 5. Signal Generation Logic
        current_market_context = predicted_states[-1] if len(predicted_states) > 0 else None
        
        entry_point = None
        take_profit = None
        stop_loss = None

        # Define thresholds for state probabilities to trigger signals
        # These thresholds would ideally be optimized through backtesting
        BULLISH_STATE_THRESHOLD = 0.7
        BEARISH_STATE_THRESHOLD = 0.7

        if current_market_context is not None and state_probabilities.shape[1] > 0:
            # Assuming state 0 is bullish, state 1 is bearish, and others are neutral/sideways
            # This mapping needs to be determined based on HMM training and interpretation
            # For now, let's assume the state with the highest mean return is bullish, lowest is bearish
            
            # A more robust approach would involve analyzing the HMM's means and covariances
            # to truly identify bullish/bearish/neutral states.
            # For this example, let's simplify and assume state 0 is bullish, state 1 is bearish.
            # In a real scenario, you'd map these based on observed market behavior in each state.

            # Example: Identify bullish and bearish states based on their characteristics (e.g., mean return)
            # This is a placeholder and would require deeper analysis of HMM output
            # For now, let's assume we have identified them.
            
            # Placeholder for identifying bullish/bearish states dynamically
            # In a real application, you'd analyze the HMM's `means_` attribute
            # to determine which state corresponds to an uptrend or downtrend.
            # For instance, if self.hmm_model.means_[state_idx, 0] (mean of log returns) is positive, it's bullish.
            
            # Dynamically identify bullish and bearish states based on the mean of log returns
            # Assuming the first feature in hmm_features is log returns
            if hasattr(self.hmm_model.model, 'means_') and self.hmm_model.model.means_.shape[1] > 0:
                mean_log_returns = self.hmm_model.model.means_[:, 0] # Assuming first feature is log returns
                bullish_state_idx = np.argmax(mean_log_returns) # State with highest mean log return
                bearish_state_idx = np.argmin(mean_log_returns) # State with lowest mean log return
            else:
                # Fallback if means_ are not available or not as expected
                bullish_state_idx = 0
                bearish_state_idx = 1

            # Get the probability of being in the bullish/bearish state
            prob_bullish = state_probabilities[-1, bullish_state_idx] if state_probabilities.shape[1] > bullish_state_idx else 0
            prob_bearish = state_probabilities[-1, bearish_state_idx] if state_probabilities.shape[1] > bearish_state_idx else 0

            last_price = raw_price_data[-1]

            if prob_bullish > BULLISH_STATE_THRESHOLD:
                # Bullish signal: Use Monte Carlo for dynamic TP/SL
                entry_point = last_price * 1.001  # Buy slightly above current price
                
                # Calculate Monte Carlo TP/SL
                mc_result = self.monte_carlo.calculate_tp_sl(
                    raw_price_data, last_price, signal_type='BUY', time_horizon=20
                )
                take_profit = mc_result['tp']
                stop_loss = mc_result['sl']
                
                # Calculate risk metrics
                risk_metrics = self.monte_carlo.calculate_risk_metrics(
                    raw_price_data, last_price, take_profit, stop_loss, 'BUY'
                )
                
                return {
                    "entry": entry_point,
                    "tp": take_profit,
                    "sl": stop_loss,
                    "market_context": current_market_context,
                    "signal_type": "BUY",
                    "confidence": float(prob_bullish),
                    "reasoning": f"Bullish HMM state detected (prob: {prob_bullish:.2%})",
                    "refined_probability_example": refined_prob[-1] if len(refined_prob) > 0 else None,
                    "monte_carlo": mc_result,
                    "risk_metrics": risk_metrics
                }
                
            elif prob_bearish > BEARISH_STATE_THRESHOLD:
                # Bearish signal: Use Monte Carlo for dynamic TP/SL
                entry_point = last_price * 0.999  # Sell slightly below current price
                
                # Calculate Monte Carlo TP/SL
                mc_result = self.monte_carlo.calculate_tp_sl(
                    raw_price_data, last_price, signal_type='SELL', time_horizon=20
                )
                take_profit = mc_result['tp']
                stop_loss = mc_result['sl']
                
                # Calculate risk metrics
                risk_metrics = self.monte_carlo.calculate_risk_metrics(
                    raw_price_data, last_price, take_profit, stop_loss, 'SELL'
                )
                
                return {
                    "entry": entry_point,
                    "tp": take_profit,
                    "sl": stop_loss,
                    "market_context": current_market_context,
                    "signal_type": "SELL",
                    "confidence": float(prob_bearish),
                    "reasoning": f"Bearish HMM state detected (prob: {prob_bearish:.2%})",
                    "refined_probability_example": refined_prob[-1] if len(refined_prob) > 0 else None,
                    "monte_carlo": mc_result,
                    "risk_metrics": risk_metrics
                }
            else:
                # Neutral or uncertain state, no signal
                return {
                    "entry": None,
                    "tp": None,
                    "sl": None,
                    "signal_type": "WAIT",
                    "market_context": current_market_context,
                    "reasoning": f"Neutral state detected - confidence too low (Bullish: {prob_bullish:.2f}, Bearish: {prob_bearish:.2f})"
                }
        else:
            # Not enough data or HMM not trained, no signal
            return {
                "entry": None,
                "tp": None,
                "sl": None,
                "signal_type": "WAIT",
                "market_context": "Insufficient Data",
                "reasoning": "HMM state probabilities unavailable"
            }

        return {
            "entry": entry_point,
            "tp": take_profit,
            "sl": stop_loss,
            "market_context": current_market_context,
            "refined_probability_example": refined_prob[-1] if len(refined_prob) > 0 else None
        }

if __name__ == '__main__':
    # Example usage with synthetic data
    np.random.seed(42)
    # Generate some synthetic price data with trends and noise
    time = np.arange(0, 100, 0.1)
    true_prices = 100 + 5 * np.sin(time / 10) + 2 * np.cos(time / 5)
    # Introduce some trend changes
    true_prices[200:500] += np.linspace(0, 10, 300)
    true_prices[700:] -= np.linspace(0, 5, 300)
    raw_data = true_prices + np.random.normal(0, 0.5, len(time)) # Add noise

    signal_gen = SignalGenerator(n_hmm_components=3, wavelet_level=2, random_state=42)
    signals = signal_gen.generate_signals(raw_data)

    print("\nGenerated Signals:", signals)