import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, svd, null_space


# Generate realizations
def generate_signal(N, c):
    # Statistics
    var_M = c
    var_L = N * (1 - c) / 2

    # Generate M and L
    M = np.random.normal(0, np.sqrt(var_M))
    L = np.random.normal(0, np.sqrt(var_L))

    # Generate K uniformly from {1, ..., N/2}
    K = int(np.random.uniform(1, N // 2 + 1))

    # Create the signal
    signal = np.full(N, M)
    signal[K - 1] = M + L
    signal[K - 1 + N // 2] = M + L

    return signal

def noise(signals, var):
    noise = np.random.normal(0, np.sqrt(var), size=(len(signals), 1))
    return signals + noise

def denoise(signals, noisy_signals, R_phi, noise_sigma):
    R_n = noise_sigma ** 2 * np.eye(noisy_signals.shape[0])
    W = R_phi @ np.linalg.inv(R_phi + R_n)

    denoised_signals = W @ noisy_signals

    # Compute MSE for each denoised signal
    mse = np.mean((signals - denoised_signals) ** 2, axis=1)
    mse_mean = np.mean(mse)
    print(f'Average MSE: {mse_mean}')

    return R_n, W, denoised_signals

def plot_signals(signals, noisy_signals, denoised_signals):
    indices_to_plot = np.random.choice(signals.shape[0]-1, size=5, replace=False)  # Select 5 random realizations

    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(indices_to_plot):
        clean_signal = signals[idx]
        noisy_signal = noisy_signals[idx]
        denoised_signal = denoised_signals[idx]

        plt.subplot(5, 1, i + 1)
        plt.plot(clean_signal, label='Clean Signal', color='green', linestyle='-', alpha=0.7)
        plt.plot(noisy_signal, label='Noisy Signal', color='red', linestyle='--', alpha=0.7)
        plt.plot(denoised_signal, label='Denoised Signal', color='blue', linestyle='-.', alpha=0.7)
        plt.title(f'Signal Example {i + 1}')
        plt.xlabel('Index')
        plt.ylabel('Amplitude')
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
        '''---------------------------------PART A---------------------------------'''
    # Parameters
    N = 64
    c = 0.6
    num_realizations = 100

    signals = np.array([generate_signal(N, c) for _ in range(num_realizations)])
    mean_signal = np.mean(signals, axis=0)
    R_phi = np.corrcoef(signals)

    # Plot the empirical mean signal
    plt.figure(figsize=(12, 6))
    plt.plot(mean_signal, label='Empirical Mean Signal')
    plt.title('Empirical Mean Signal')
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    plt.show()

    # Plot the autocorrelation matrix
    plt.imshow(R_phi, cmap='viridis', aspect='auto')
    plt.title('Empirical Autocorrelation Matrix')
    plt.colorbar(label='Correlation')
    plt.xlabel('Index')
    plt.ylabel('Index')

    plt.tight_layout()
    plt.show()

    '''---------------------------------PART B---------------------------------'''
    noise_sigma = np.sqrt(5)
    noisy_signals = noise(signals, noise_sigma ** 2)
    _, _, denoised_signals = denoise(signals, noisy_signals, R_phi, noise_sigma)
    plot_signals(signals, noisy_signals, denoised_signals)

    '''---------------------------------PART C---------------------------------'''
    first_row = np.array([-5 / 2, 4 / 3, -1 / 12] + [0] * (N - 5) + [4 / 3, -1 / 12])
    H = toeplitz(first_row, first_row)
    degraded_signals = np.array([H @ signal for signal in signals])
    print(signals[0] - degraded_signals[0])
    noisy_degraded_signals = noise(degraded_signals, noise_sigma**2)
    _, _, denoised_degraded_signals = denoise(signals, noisy_signals, R_phi, noise_sigma)
    plot_signals(signals, noisy_degraded_signals, denoised_degraded_signals)

    '''
    test_signal = signals[4]
    degraded_test_signal = H @ test_signal

    for i in range(2, 10):
        val = 1/12 * test_signal[i-2] + 4/3 * test_signal[i-1] - 5/2 * test_signal[i] + 4/3 * test_signal[i+1] - 1/12 * test_signal[i+2]
        print(degraded_test_signal[i] - val)
    '''

    '''---------------------------------PART E---------------------------------'''
    U, s, Vh = svd(H)
    Sigma_inv = np.diag(1/s)
    H_pseudo_inv = Vh.T @ Sigma_inv @ U.T
    '''
    print(np.linalg.matrix_rank(H_pseudo_inv))
    null_space_H_pseudo_inv = null_space(H_pseudo_inv)

    phi1 = np.random.randn(N)
    # φ2 = φ1 + vector in the null space of H†
    phi2 = phi1 + 256 * null_space_H_pseudo_inv[:, 0]

    print("Norm of φ1 - φ2:", np.linalg.norm(phi1 - phi2))
    print("H†φ1:", H_pseudo_inv @ phi1)
    print("H†φ2:", H_pseudo_inv @ phi2)
    '''
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(H_pseudo_inv @ H, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('H†H')

    plt.subplot(1, 2, 2)
    plt.imshow(H @ H_pseudo_inv, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('HH†')

    plt.tight_layout()
    plt.show()
