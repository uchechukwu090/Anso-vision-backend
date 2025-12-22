import numpy as np
from filterpy.kalman import KalmanFilter

def apply_kalman_filter(data):
    """
    Applies a Kalman filter to a 1D data array for smoothing.
    Returns smoothed data only (backward compatible).

    Args:
        data (np.array): A 1D numpy array of observations.

    Returns:
        np.array: The smoothed data.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 1:
        raise ValueError("Input data must be a 1D numpy array.")

    # Initialize Kalman Filter
    # State vector: [position]
    # Measurement vector: [position]
    kf = KalmanFilter(dim_x=1, dim_z=1)

    # State transition matrix (A)
    # Assumes constant velocity, but since dim_x=1, it's just a constant state
    kf.F = np.array([[1.]])

    # Measurement function (H)
    # Maps state to measurement (we directly observe position)
    kf.H = np.array([[1.]])

    # Covariance matrix (P)
    # Initial uncertainty in the state
    kf.P *= 1000.

    # Measurement noise covariance (R)
    # How much noise do we expect in our measurements
    kf.R = np.array([[5.]])

    # Process noise covariance (Q)
    # How much uncertainty in our state transition (model uncertainty)
    kf.Q = np.array([[0.1]])

    # Initialize state (x) with the first data point
    kf.x = np.array([[data[0]]])

    smoothed_data = []
    for z in data:
        kf.predict()
        kf.update(z)
        smoothed_data.append(kf.x[0, 0])

    return np.array(smoothed_data)

def apply_kalman_filter_enhanced(data):
    """
    ✅ ENHANCED: Returns Kalman-filtered components for feature engineering:
    - Smoothed price (level)
    - Velocity (first derivative / trend)
    - Acceleration (second derivative / rate of change of trend)
    
    Args:
        data (np.array): A 1D numpy array of observations.
    
    Returns:
        dict: Contains 'smoothed_price', 'velocity', 'acceleration'
    """
    if not isinstance(data, np.ndarray) or data.ndim != 1:
        raise ValueError("Input data must be a 1D numpy array.")
    
    # Apply Kalman filter
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = np.array([[1., 1.], [0., 1.]])  # State: [position, velocity]
    kf.H = np.array([[1., 0.]])  # Measure position only
    kf.P *= 1000.
    kf.R = np.array([[5.]])
    kf.Q = np.array([[0.1, 0.], [0., 0.01]])
    kf.x = np.array([[data[0]], [0.]])  # Initial state
    
    smoothed_prices = []
    velocities = []
    
    for z in data:
        kf.predict()
        kf.update(z)
        smoothed_prices.append(kf.x[0, 0])
        velocities.append(kf.x[1, 0])
    
    smoothed_prices = np.array(smoothed_prices)
    velocities = np.array(velocities)
    
    # Calculate acceleration (rate of change of velocity)
    acceleration = np.zeros_like(velocities)
    for i in range(1, len(velocities)):
        acceleration[i] = velocities[i] - velocities[i-1]
    
    return {
        'smoothed_price': smoothed_prices,
        'velocity': velocities,
        'acceleration': acceleration
    }

if __name__ == '__main__':
    # Example usage
    # Generate some noisy data
    np.random.seed(0)
    true_signal = np.linspace(0, 10, 100)
    noisy_data = true_signal + np.random.normal(0, 1, 100)

    # Apply Kalman filter
    smoothed_signal = apply_kalman_filter(noisy_data)

    print("Original noisy data (first 5 points):", noisy_data[:5])
    print("Smoothed data (first 5 points):", smoothed_signal[:5])

    # You can plot these to visualize the smoothing effect
    # import matplotlib.pyplot as plt
    # plt.plot(true_signal, label='True Signal')
    # plt.plot(noisy_data, label='Noisy Data')
    # plt.plot(smoothed_signal, label='Smoothed Signal')
    # plt.legend()
    # plt.show()