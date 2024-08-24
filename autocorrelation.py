import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, svd, null_space


def autocor(phi):
    #phi (n,N)

    cor = np.einsum('ni,nj->nij', phi, phi)
    print(np.mean(cor,axis=0).shape)
    return np.mean(cor,axis=0)
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
    noise = np.random.normal(0, np.sqrt(var), size=(len(signals), len(signals[0])))
    return signals + noise

def denoise(signals, noisy_signals, R_phi, noise_sigma,H):
    R_n = noise_sigma ** 2 * np.eye(R_phi.shape[0])
    W = R_phi@H.T @ np.linalg.inv(H@R_phi@H.T + R_n)

    denoised_signals = noisy_signals @ W

    # Compute MSE for each denoised signal
    mse = np.mean((signals - denoised_signals) ** 2, axis=0)
    print("Noise sigma: ", noise_sigma)
    print(f'Average MSE per entry: {np.mean(mse)}')
    print(f'Average MSE per signal: {sum(mse)}\n')

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
    ''''---------------------------------PART A---------------------------------'''
    # Parameters
    N = 64
    c = 0.6
    num_realizations = 100

    signals = np.array([generate_signal(N, c) for _ in range(num_realizations)])
    mean_signal = np.mean(signals, axis=0)
    R_phi = autocor(signals)

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
    _, W, denoised_signals = denoise(signals, noisy_signals, R_phi, noise_sigma,np.eye(R_phi.shape[0]))
    plot_signals(signals, noisy_signals, denoised_signals)

    # Plot the Wiener filter matrix
    plt.imshow(W, cmap='viridis', aspect='auto')
    plt.title('Wiener filter matrix')
    plt.colorbar(label='Correlation')

    '''
    M = np.eye(64,64)+np.diag(np.ones(32),32)+np.diag(np.ones(32),-32)
    plt.imshow(M, cmap='viridis', aspect='auto')
    plt.title('Theoretical Autocorrelation Matrix')
    plt.colorbar(label='Correlation')
    '''

    '''---------------------------------PART C---------------------------------'''
    first_row = np.array([-5 / 2, 4 / 3, -1 / 12] + [0] * (N - 5) + [-1 / 12, 4 / 3])
    H = toeplitz(first_row, first_row)

    degraded_signals = np.array([H @ signal for signal in signals])
    noisy_degraded_signals = noise(degraded_signals, noise_sigma**2)
    _, W_new, denoised_degraded_signals = denoise(signals, noisy_degraded_signals, R_phi, noise_sigma,H)
    plot_signals(signals, noisy_degraded_signals, denoised_degraded_signals)

    # Plot the Wiener filter matrix
    plt.imshow(W_new, cmap='viridis', aspect='auto')
    plt.title('Wiener filter matrix')
    plt.colorbar(label='Correlation')
    plt.show()
    '''---------------------------------PART E---------------------------------'''
    U, s, Vh = svd(H)
    Sigma_inv = np.diag(1/s)
    H_pseudo_inv = Vh.T @ Sigma_inv @ U.T

    null_space_H_pseudo_inv = null_space(H_pseudo_inv)

    phi1 = np.ones(N)
    phi2 = 250 * phi1

    plt.imshow(H @ H_pseudo_inv, cmap='viridis', aspect='auto')
    plt.title('HH†')
    plt.show()

    plt.imshow(H_pseudo_inv @ H, cmap='viridis', aspect='auto')
    plt.title('H†H')

    print("Norm of φ1 - φ2:", np.linalg.norm(phi1 - phi2))
    print("Norm of H†φ1 - H†φ2:", np.linalg.norm(H_pseudo_inv @ phi1 - H_pseudo_inv @ phi2))

    to_plot = {
        "φ1": phi1,
        "φ2": phi2,
        "φ1-φ2": phi1 - phi2,
        "H†φ1": H_pseudo_inv @ phi1,
        "H†φ2": H_pseudo_inv @ phi2,
        "H†φ1 - H†φ2": H_pseudo_inv @ phi1 - H_pseudo_inv @ phi2
        }

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for i, (label, signal) in enumerate(to_plot.items()):
        row = i // 3
        col = i % 3
        axs[row, col].plot(signal)
        axs[row, col].set_title(label)
        axs[row, col].grid(True)
    plt.tight_layout()
    plt.show()
