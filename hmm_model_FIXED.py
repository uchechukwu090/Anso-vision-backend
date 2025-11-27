import numpy as np
from hmmlearn import hmm

class MarketHMM:
    def __init__(self, n_components=3, n_iter=100, covariance_type='diag', random_state=None, 
                 smoothing_window=12, min_state_duration=3, state_switch_threshold=0.75):
        """Hidden Markov Model for market context - FIXED VERSION"""
        # FIX: Set init_params to empty string to avoid warnings
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            init_params='',  # ← CRITICAL FIX: Don't auto-initialize
            params='stmc'     # But allow training to update all params
        )
        self.n_components = n_components
        self.smoothing_window = smoothing_window
        self.min_state_duration = min_state_duration
        self.state_switch_threshold = state_switch_threshold
        self.previous_state = None
        self.state_confidence = 0.0
        self.state_history = []
        self.state_duration_counter = 0
        self.is_trained = False

    def train(self, data):
        """Train HMM model"""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("Input data for training must be a 2D numpy array.")
        
        # Initialize parameters manually to avoid warnings
        self.model.startprob_ = np.ones(self.n_components) / self.n_components
        self.model.transmat_ = np.ones((self.n_components, self.n_components)) / self.n_components
        
        # Initialize means and covariances based on data
        n_features = data.shape[1]
        self.model.means_ = np.random.randn(self.n_components, n_features)
        
        if self.model.covariance_type == 'diag':
            self.model.covars_ = np.ones((self.n_components, n_features))
        elif self.model.covariance_type == 'full':
            self.model.covars_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
        elif self.model.covariance_type == 'spherical':
            self.model.covars_ = np.ones(self.n_components)
        elif self.model.covariance_type == 'tied':
            self.model.covars_ = np.eye(n_features)
        
        # Now train
        self.model.fit(data)
        self.is_trained = True
        print(f"✅ HMM trained with {self.n_components} components (no warnings!)")

    def predict_states(self, data):
        """Predict hidden states with smoothing"""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("Input data for prediction must be a 2D numpy array.")
        
        raw_states = self.model.predict(data)
        state_probs = self.model.predict_proba(data)
        smoothed_states = self._smooth_states(raw_states, state_probs)
        
        return smoothed_states

    def _smooth_states(self, states, state_probabilities=None):
        """Smooth state predictions"""
        if len(states) < self.smoothing_window:
            return states
        
        smoothed = np.copy(states)
        
        for i in range(len(states)):
            start = max(0, i - self.smoothing_window // 2)
            end = min(len(states), i + self.smoothing_window // 2 + 1)
            
            window_states = states[start:end]
            most_common_state = np.bincount(window_states.astype(int)).argmax()
            
            if i > 0 and smoothed[i-1] != most_common_state:
                recent_states = smoothed[max(0, i - self.min_state_duration):i]
                if len(recent_states) > 0 and len(recent_states) == self.min_state_duration:
                    prev_state = smoothed[i-1]
                    if np.sum(recent_states == prev_state) >= self.min_state_duration:
                        if state_probabilities is not None and i < len(state_probabilities):
                            new_state_prob = state_probabilities[i, most_common_state]
                            if new_state_prob < self.state_switch_threshold:
                                smoothed[i] = prev_state
                                continue
            
            smoothed[i] = most_common_state
        
        return smoothed

    def get_state_probabilities(self, data):
        """Get state probabilities"""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("Input data for state probabilities must be a 2D numpy array.")
        return self.model.predict_proba(data)

    def get_state_stability(self, states):
        """Calculate state stability/confidence"""
        if len(states) < self.smoothing_window:
            return 0.5
        
        recent_states = states[-self.smoothing_window:]
        current_state = states[-1]
        matches = np.sum(recent_states == current_state)
        confidence = matches / len(recent_states)
        
        return float(confidence)
