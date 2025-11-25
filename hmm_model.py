import numpy as np
from hmmlearn import hmm

class MarketHMM:
    def __init__(self, n_components=3, n_iter=100, covariance_type='diag', random_state=None, smoothing_window=12, min_state_duration=3, state_switch_threshold=0.75):
        """
        Initializes the Hidden Markov Model for market context identification.

        Args:
            n_components (int): The number of hidden states in the HMM.
            n_iter (int): Number of iterations for the EM algorithm.
            covariance_type (str): The type of covariance matrix to use ('spherical', 'diag', 'full', 'tied').
            random_state (int, optional): Seed for the random number generator for reproducibility. Defaults to None.
            smoothing_window (int): Number of candles to smooth over to prevent rapid state changes. Default: 12.
            min_state_duration (int): Minimum candles a state must persist before switching. Default: 3.
            state_switch_threshold (float): Minimum probability required to switch states. Default: 0.75.
        """
        self.model = hmm.GaussianHMM(n_components=n_components, covariance_type=covariance_type, n_iter=n_iter, random_state=random_state)
        self.n_components = n_components
        self.smoothing_window = smoothing_window
        self.min_state_duration = min_state_duration
        self.state_switch_threshold = state_switch_threshold
        self.previous_state = None
        self.state_confidence = 0.0
        self.state_history = []  # Track last N states for stability analysis
        self.state_duration_counter = 0  # Track how long current state has been active

    def train(self, data):
        """
        Trains the HMM model on the provided data.

        Args:
            data (np.array): A 2D numpy array of observations, where each row is a sample
                             and each column is a feature (e.g., [price_change, volume_change]).
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("Input data for training must be a 2D numpy array.")
        self.model.fit(data)
        print(f"HMM trained with {self.n_components} components.")

    def predict_states(self, data):
        """
        Predicts the hidden states for the given data with smoothing to prevent rapid oscillation.

        Args:
            data (np.array): A 2D numpy array of observations.

        Returns:
            np.array: A 1D numpy array of predicted hidden states (smoothed).
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("Input data for prediction must be a 2D numpy array.")
        
        # Get raw predictions
        raw_states = self.model.predict(data)
        
        # Get state probabilities for confidence checking
        state_probs = self.model.predict_proba(data)
        
        # Apply smoothing to prevent rapid state changes
        smoothed_states = self._smooth_states(raw_states, state_probs)
        
        return smoothed_states

    def _smooth_states(self, states, state_probabilities=None):
        """
        Smooth state predictions using majority voting over a sliding window
        with minimum state duration and confidence threshold requirements.
        This prevents the HMM from flipping states on every candle.
        
        Args:
            states (np.array): Raw predicted states from the model
            state_probabilities (np.array): State probabilities for confidence checking
            
        Returns:
            np.array: Smoothed states
        """
        if len(states) < self.smoothing_window:
            return states
        
        smoothed = np.copy(states)
        
        # Apply majority voting in a sliding window with confidence check
        for i in range(len(states)):
            # Define window boundaries
            start = max(0, i - self.smoothing_window // 2)
            end = min(len(states), i + self.smoothing_window // 2 + 1)
            
            # Get most common state in the window
            window_states = states[start:end]
            most_common_state = np.bincount(window_states.astype(int)).argmax()
            
            # Check if we should enforce minimum state duration
            if i > 0 and smoothed[i-1] != most_common_state:
                # State wants to change - check if it meets duration requirement
                recent_states = smoothed[max(0, i - self.min_state_duration):i]
                if len(recent_states) > 0 and len(recent_states) == self.min_state_duration:
                    # Check if previous state held for minimum duration
                    prev_state = smoothed[i-1]
                    if np.sum(recent_states == prev_state) >= self.min_state_duration:
                        # Previous state held long enough, check confidence for switch
                        if state_probabilities is not None and i < len(state_probabilities):
                            new_state_prob = state_probabilities[i, most_common_state]
                            if new_state_prob < self.state_switch_threshold:
                                # Not confident enough to switch, keep previous state
                                smoothed[i] = prev_state
                                continue
            
            smoothed[i] = most_common_state
        
        return smoothed

    def get_state_probabilities(self, data):
        """
        Calculates the probabilities of each state for each observation.

        Args:
            data (np.array): A 2D numpy array of observations.

        Returns:
            np.array: A 2D numpy array where each row corresponds to an observation
                      and each column to the probability of being in a particular state.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("Input data for state probabilities must be a 2D numpy array.")
        return self.model.predict_proba(data)

    def get_state_stability(self, states):
        """
        Calculate how stable/confident the current state is.
        If the last N candles all show the same state, confidence is high.
        
        Args:
            states (np.array): Predicted states
            
        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        if len(states) < self.smoothing_window:
            return 0.5  # Default moderate confidence
        
        recent_states = states[-self.smoothing_window:]
        # Count how many of the last N states match the current state
        current_state = states[-1]
        matches = np.sum(recent_states == current_state)
        confidence = matches / len(recent_states)
        
        return float(confidence)

if __name__ == '__main__':
    # Example usage
    # Generate some synthetic data for demonstration
    np.random.seed(42)
    # Let's assume 2 features: price change and volume change
    # State 0: low volatility, slight upward trend
    data_state0 = np.random.normal(loc=[0.01, 0.1], scale=[0.02, 0.05], size=(100, 2))
    # State 1: high volatility, downward trend
    data_state1 = np.random.normal(loc=[-0.02, 0.2], scale=[0.05, 0.1], size=(100, 2))
    # State 2: moderate volatility, sideways market
    data_state2 = np.random.normal(loc=[0.00, 0.15], scale=[0.03, 0.08], size=(100, 2))

    # Combine data to simulate a sequence of market states
    synthetic_data = np.vstack([data_state0, data_state1, data_state2, data_state0])

    # Initialize and train HMM
    hmm_model = MarketHMM(n_components=3, n_iter=100, random_state=42, smoothing_window=12)
    hmm_model.train(synthetic_data)

    # Predict states
    predicted_states = hmm_model.predict_states(synthetic_data)
    print("\nPredicted states (first 20):", predicted_states[:20])
    print("Predicted states (last 20):", predicted_states[-20:])

    # Get state probabilities
    state_probabilities = hmm_model.get_state_probabilities(synthetic_data)
    print("\nState probabilities for first sample:", np.exp(state_probabilities[0]))
    print("State probabilities for last sample:", np.exp(state_probabilities[-1]))
    
    # Get state stability
    stability = hmm_model.get_state_stability(predicted_states)
    print(f"\nState stability: {stability:.2%}")
