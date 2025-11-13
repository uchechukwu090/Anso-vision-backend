import numpy as np
import pywt

def apply_wavelet_decomposition(signal, wavelet='db1', level=1):
    """
    Applies wavelet decomposition to a 1D signal.

    Args:
        signal (np.array): A 1D numpy array representing the signal.
        wavelet (str): The name of the wavelet to use (e.g., 'db1', 'haar').
        level (int): The level of decomposition.

    Returns:
        list: A list of approximation and detail coefficients.
    """
    if not isinstance(signal, np.ndarray) or signal.ndim != 1:
        raise ValueError("Input signal must be a 1D numpy array.")
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return coeffs

def reconstruct_wavelet_signal(coeffs, wavelet='db1'):
    """
    Reconstructs a signal from its wavelet coefficients.

    Args:
        coeffs (list): A list of approximation and detail coefficients.
        wavelet (str): The name of the wavelet used for decomposition.

    Returns:
        np.array: The reconstructed 1D signal.
    """
    return pywt.waverec(coeffs, wavelet)

def denoise_signal_with_wavelets(signal, wavelet='db1', level=1, mode='soft', sigma=None):
    """
    Denoises a signal using wavelet decomposition and thresholding.

    Args:
        signal (np.array): A 1D numpy array representing the signal.
        wavelet (str): The name of the wavelet to use.
        level (int): The level of decomposition.
        mode (str): Thresholding mode ('soft' or 'hard').
        sigma (float): Standard deviation of the noise. If None, it's estimated.

    Returns:
        np.array: The denoised signal.
    """
    if np.all(signal == signal[0]):  # Check if the signal is constant
        return signal  # Return original signal if it's constant to avoid issues

    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = sigma if sigma is not None else np.median(np.abs(coeffs[-1])) / 0.6745
    # Ensure sigma is not zero to prevent potential division by zero in thresholding
    if sigma < np.finfo(float).eps:
        sigma = np.finfo(float).eps
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    # Apply thresholding
    denoised_coeffs = []
    for i, c in enumerate(coeffs):
        if i == 0:  # Approximation coefficients are not thresholded
            denoised_coeffs.append(c)
        else:
            denoised_coeffs.append(pywt.threshold(c, threshold, mode=mode))

    return pywt.waverec(denoised_coeffs, wavelet)

if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    # Generate a sample signal with noise
    t = np.linspace(0, 1, 500, endpoint=False)
    clean_signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)
    noisy_signal = clean_signal + 2 * np.random.randn(len(t))

    # Apply wavelet decomposition
    coeffs = apply_wavelet_decomposition(noisy_signal, wavelet='db1', level=4)
    print(f"Number of coefficient arrays: {len(coeffs)}")
    print(f"Length of approximation coefficients: {len(coeffs[0])}")

    # Reconstruct signal
    reconstructed_signal = reconstruct_wavelet_signal(coeffs, wavelet='db1')
    print(f"Length of original signal: {len(noisy_signal)}")
    print(f"Length of reconstructed signal: {len(reconstructed_signal)}")

    # Denoise signal
    denoised_signal = denoise_signal_with_wavelets(noisy_signal, wavelet='db1', level=4)
    print(f"Length of denoised signal: {len(denoised_signal)}")

    # You can plot these to visualize the effects
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 6))
    # plt.plot(noisy_signal, label='Noisy Signal', alpha=0.7)
    # plt.plot(clean_signal, label='Clean Signal', linestyle='--')
    # plt.plot(denoised_signal, label='Denoised Signal')
    # plt.legend()
    # plt.title('Wavelet Denoising Example')
    # plt.show()