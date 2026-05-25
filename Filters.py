import numpy as np
import matplotlib.pyplot as plt


import numpy as np


def convolution(h, x, mode='full', compensate_delay=True):
    """
    Ręczna implementacja splotu dyskretnego.

    Parametry:
    - h : odpowiedź impulsowa / filtr
    - x : sygnał wejściowy
    - mode:
        'full'  -> pełny splot
        'same'  -> wynik o długości sygnału x
        'valid' -> tylko pełne nakładanie

    Zwraca:
    - tablica numpy ze splotem
    """
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
    """
    Okno Hamminga wg wzoru (5) z zadania.
    w(n) = 0.53836 - 0.46164 * cos(2πn/M)
    """
    n = np.arange(M)
    return 0.53836 - 0.46164 * np.cos(2 * np.pi * n / M)


def rectangular_window(M):
    """Okno prostokątne (domyślne)"""
    return np.ones(M)


def lowpass_filter_coefficients(M, K):
    """
    Współczynniki filtru dolnoprzepustowego wg wzoru (4) z zadania.
    h(n) = 2/K dla n=(M-1)/2
    h(n) = sin(2π(n-(M-1)/2)/K) / (π(n-(M-1)/2)) w pozostałych przypadkach
    """
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
    """
    Zastosuj funkcję okna do współczynników filtru.
    """
    M = len(h)
    
    if window_type == 'rectangular':
        window = rectangular_window(M)
    elif window_type == 'hamming':
        window = hamming_window(M)
    else:
        raise ValueError(f"Nieznany typ okna: {window_type}")
    
    return h * window


def bandpass_filter_coefficients(h, filter_type='bandpass'):
    """
    Transformuj filtr dolnoprzepustowy na filtr bandpass/highpass.
    
    Dla bandpass: mnóż przez 2*sin(πn/2) wg str. 7 zadania
    Dla highpass: mnóż przez (-1)^n
    """
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
    """
    Filtruj sygnał wg wzoru (3) z zadania - splot.
    
    Parametry:
    - signal: sygnał wejściowy
    - h: współczynniki filtru
    - compensate_delay: czy kompensować przesunięcie (True = wyrównanie do oryginalnego)
    
    ✅ Domyślnie sygnał filtrowany zaczyna się w tym samym punkcie co oryginalny
    """
    if compensate_delay:
        # Zwróć splot o tej samej długości co sygnał wejściowy (mode='same')
        return convolution(h, signal, mode='same')
    else:
        # Zwróć pełny splot
        return convolution(h, signal, mode='full')


def design_filter(M, K, window_type='rectangular', filter_type='lowpass'):
    """
    Zaprojektuj filtr FIR wg metody okna.
    
    Parametry:
    - M: rząd filtru (musi być nieparzysty)
    - K: parametr częstości odcięcia (f_o = f_p / K)
    - window_type: typ okna ('rectangular', 'hamming', 'hanning', 'blackman')
    - filter_type: typ filtru ('lowpass', 'bandpass', 'highpass')
    """
    h = lowpass_filter_coefficients(M, K)

    h = apply_window(h, window_type)

    if filter_type in ['bandpass', 'highpass']:
        h = bandpass_filter_coefficients(h, filter_type)

    return h


def show_filter_comparison(original, filtered, h):
    """Wyświetl porównanie sygnału oryginalnego i przefiltrowanego"""
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


# ============================================================================
# CZĘŚĆ DODATKOWA: KORELACJA SYGNAŁÓW (wymagane wg zadania str. 10-13)
# ============================================================================

def correlation_direct(h, x):
    """
    Korelacja wzajemna - implementacja bezpośrednia.

    Wynik ma długość M+N-1.
    Indeksy są przeindeksowane do zakresu 0..M+N-2.
    """

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
    """
    Korelacja wzajemna - implementacja z użyciem splotu wg wzoru (2).
    Korelacja: R_hx = splot(h, odwrócone_x)
    
    Zwraca sygnał o długości M+N-1, zindeksowany od 0.
    """
    M = len(h)
    N = len(x)
    
    x_reversed = x[::-1]
    
    result = convolution(h, x_reversed, mode='full')
    
    return result


def cross_correlation(signal1, signal2, method='direct'):
    """
    Oblicz korelację wzajemną dwóch sygnałów.
    
    Parametry:
    - signal1, signal2: sygnały wejściowe
    - method: 'direct' lub 'convolution'
    
    Zwraca: korelacja wzajemna (indeksowana od 0)
    """
    if method == 'direct':
        return correlation_direct(signal1, signal2)
    elif method == 'convolution':
        return correlation_using_convolution(signal1, signal2)
    else:
        raise ValueError(f"Nieznana metoda: {method}")


def find_delay(correlation_result):
    global_max_idx = np.argmax(np.abs(correlation_result))
        
    return global_max_idx


def radar_distance_measurement(
        probe_signal,
        reflected_signal,
        sampling_rate,
        signal_speed,
        method='direct'
):
    """
    Pomiar odległości metodą korelacyjną.
    """

    corr = cross_correlation(
        probe_signal,
        reflected_signal,
        method=method
    )

    delay_samples = find_delay(corr)

    delay_time = delay_samples / sampling_rate

    distance = (signal_speed * delay_time) / 2

    return distance, delay_time, delay_samples, corr


def show_correlation_analysis(signal1, signal2, correlation, title="Korelacja sygnałów"):
    """Wyświetl analizę korelacji"""
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
    axes[2].axvline(x=np.argmax(np.abs(correlation)), color='k', linestyle='--', label='Maksimum')
    axes[2].set_ylabel('Korelacja')
    axes[2].set_xlabel('Opóźnienie (próbki)')
    axes[2].set_title(title)
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.show()