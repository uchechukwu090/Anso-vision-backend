import numpy as np
from hmmlearn import hmm

class MarketHMM:
    def __init__(self, n_components=3, n_iter=100, covariance_type='diag', random_state=None):
        """
        Initializes the Hidden Markov Model for market context identification.

        Args:
            n_components (int): The number of hidden states in the HMM.
            n_iter (int): Number of iterations for the EM algorithm.
            covariance_type (str): The type of covariance matrix to use ('spherical', 'diag', 'full', 'tied').
            random_state (int, optional): Seed for the random number generator for reproducibility. Defaults to None.
        """
        self.model = hmm.GaussianHMM(n_components=n_components, covariance_type=covariance_type, n_iter=n_iter, random_state=random_state)
        self.n_components = n_components

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
        Predicts the hidden states for the given data.

        Args:
            data (np.array): A 2D numpy array of observations.

        Returns:
            np.array: A 1D numpy array of predicted hidden states.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("Input data for prediction must be a 2D numpy array.")
        return self.model.predict(data)

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
    hmm_model = MarketHMM(n_components=3, n_iter=100, random_state=42)
    hmm_model.train(synthetic_data)

    # Predict states
    predicted_states = hmm_model.predict_states(synthetic_data)
    print("\nPredicted states (first 20):", predicted_states[:20])
    print("Predicted states (last 20):", predicted_states[-20:])

    # Get state probabilities
    state_probabilities = hmm_model.get_state_probabilities(synthetic_data)
    print("\nState probabilities for first sample:", np.exp(state_probabilities[0]))
    print("State probabilities for last sample:", np.exp(state_probabilities[-1]))