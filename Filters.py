import numpy as np
import matplotlib.pyplot as plt


def convolution(h, x, mode='full'):
    M = len(h)
    N = len(x)
    full_length = N + M - 1
    y = np.zeros(full_length)

    for n in range(full_length):
        s = 0.0
        for k in range(N):

            h_index = n - k

            if 0 <= h_index < M:
                s += x[k] * h[h_index]

        y[n] = s

    if mode == 'full':
        return y

    elif mode == 'same':
        start = (M - 1) // 2
        end = start + N
        return y[start:end]

    elif mode == 'valid':
        start = M - 1
        end = N
        return y[start:end]
    
    else:
        raise ValueError("mode musi być: 'full', 'same' lub 'valid'")


def hamming_window(M):
    n = np.arange(M)
    return 0.53836 - 0.46164 * np.cos(2 * np.pi * n / M)


def rectangular_window(M):
    return np.ones(M)


def lowpass_filter_coefficients(M, K):
    if M % 2 == 0:
        raise ValueError("M musi być nieparzyste!")

    h = np.zeros(M)
    mid = (M - 1) / 2

    for n in range(M):
        if n == mid:
            h[n] = 2.0 / K
        else:
            numerator = np.sin(2 * np.pi * (n - mid) / K)
            denominator = np.pi * (n - mid)
            h[n] = numerator / denominator

    return h


def apply_window(h, window_type='rectangular'):
    M = len(h)
    
    if window_type == 'rectangular':
        window = rectangular_window(M)
    elif window_type == 'hamming':
        window = hamming_window(M)
    else:
        raise ValueError(f"Nieznany typ okna: {window_type}")
    
    return h * window


def bandpass_filter_coefficients(h, filter_type='bandpass'):
    h_modified = np.copy(h)
    M = len(h)

    if filter_type == 'bandpass':
        # Filtr środkowoprzepustowy
        for n in range(M):
            h_modified[n] *= 2 * np.sin(np.pi * n / 2)
    elif filter_type != 'lowpass':
        raise ValueError(f"Nieznany typ filtru: {filter_type}")

    return h_modified


def filter_signal(signal, h, compensate_delay=True):
    if compensate_delay:
        return convolution(h, signal, mode='same')
    else:
        return convolution(h, signal, mode='full')


def design_filter(M, K, window_type='rectangular', filter_type='lowpass'):
    h = lowpass_filter_coefficients(M, K)

    h = apply_window(h, window_type)

    if filter_type in ['bandpass', 'highpass']:
        h = bandpass_filter_coefficients(h, filter_type)

    return h


def show_filter_comparison(original, filtered, h):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    axes[0].plot(original, 'b-', label='Sygnał oryginalny')
    axes[0].set_ylabel('Amplituda')
    axes[0].set_title('Sygnał wejściowy')
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(filtered, 'r-', label='Sygnał przefiltrowany')
    axes[1].set_ylabel('Amplituda')
    axes[1].set_title('Sygnał wyjściowy (po filtracji)')
    axes[1].grid(True)
    axes[1].legend()

    axes[2].stem(h, linefmt='g-', markerfmt='go')
    axes[2].set_ylabel('Amplituda')
    axes[2].set_xlabel('n')
    axes[2].set_title('Odpowiedź impulsowa filtru h(n)')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

def correlation_direct(h, x):

    M = len(h)
    N = len(x)

    result_len = M + N - 1
    result = np.zeros(result_len)

    for n in range(-(N - 1), M):

        s = 0.0

        for k in range(M):

            x_index = k - n

            if 0 <= x_index < N:
                s += h[k] * x[x_index]

        result[n + (N - 1)] = s

    return result


def correlation_using_convolution(h, x):
    M = len(h)
    N = len(x)
    
    x_reversed = x[::-1]
    
    result = convolution(h, x_reversed, mode='full')
    
    return result


def cross_correlation(signal1, signal2, method='direct'):
    if method == 'direct':
        return correlation_direct(signal1, signal2)
    elif method == 'convolution':
        return correlation_using_convolution(signal1, signal2)
    else:
        raise ValueError(f"Nieznana metoda: {method}")


def default_radar_search_window(num_samples, sampling_rate):
    window = int(round(0.04 * sampling_rate))
    window = max(10, window)
    window = min(window, max(1, num_samples // 20))
    return window


def find_delay(correlation_result, reference_length, search_window=None):
    corr = np.abs(np.asarray(correlation_result, dtype=float))
    zero_lag = reference_length - 1
    n = len(corr)

    if search_window is None:
        w = max(zero_lag, n - 1 - zero_lag)
    else:
        w = int(search_window)

    lo = max(0, zero_lag - w)
    hi = min(n, zero_lag + w + 1)

    if hi - lo < 3:
        return 0, zero_lag, zero_lag

    peaks = []
    for i in range(lo + 1, hi - 1):
        if corr[i] >= corr[i - 1] and corr[i] > corr[i + 1]:
            peaks.append(i)

    if peaks:
        peak_idx = min(peaks, key=lambda i: abs(i - zero_lag))
    else:
        segment = corr[lo:hi]
        max_val = float(np.max(segment))
        tol = 1e-9 * max(max_val, 1.0)
        candidates = np.flatnonzero(segment >= max_val - tol) + lo
        peak_idx = int(candidates[int(np.argmin(np.abs(candidates - zero_lag)))])

    delay_samples = abs(peak_idx - zero_lag)
    return delay_samples, peak_idx, zero_lag


def radar_distance_measurement(
        probe_signal,
        reflected_signal,
        sampling_rate,
        signal_speed,
        method='direct',
        search_window=None,
        echo_mode=False):
    if echo_mode:
        corr = cross_correlation(reflected_signal, probe_signal, method=method)
        ref_len = len(reflected_signal)
    else:
        corr = cross_correlation(probe_signal, reflected_signal, method=method)
        ref_len = len(probe_signal)

    delay_samples, peak_idx, zero_lag = find_delay(
        corr, ref_len, search_window=search_window
    )

    delay_time = delay_samples / sampling_rate
    distance = (signal_speed * delay_time) / 2

    return distance, delay_time, delay_samples, corr, peak_idx, zero_lag


def show_correlation_analysis(signal1, signal2, correlation, title="Korelacja sygnałów",
                              search_window=None, peak_idx=None, zero_lag=None):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    axes[0].plot(signal1, 'b-', label='Sygnał 1 (sondujący)')
    axes[0].set_ylabel('Amplituda')
    axes[0].set_title('Sygnał sondujący')
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(signal2, 'g-', label='Sygnał 2 (zwrotny)')
    axes[1].set_ylabel('Amplituda')
    axes[1].set_title('Sygnał zwrotny')
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(correlation, 'r-', linewidth=1.5)
    ref_len = len(signal1)
    if zero_lag is None:
        zero_lag = ref_len - 1
    if peak_idx is None:
        delay_samples, peak_idx, zero_lag = find_delay(
            correlation, ref_len, search_window=search_window
        )
    else:
        delay_samples = peak_idx - zero_lag

    axes[2].axvline(x=zero_lag, color='gray', linestyle=':', label='Lag 0')
    axes[2].axvline(x=peak_idx, color='k', linestyle='--',
                    label=f'Maksimum przy środku (opóźnienie {delay_samples} próbek)')
    axes[2].set_ylabel('Korelacja')
    axes[2].set_xlabel('Opóźnienie (próbki)')
    axes[2].set_title(title)
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.show()