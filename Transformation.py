import numpy as np
import time


def dft_definition(x):
    x = np.asarray(x, dtype=np.complex128)
    N = len(x)
    if N == 0:
        return np.array([], dtype=np.complex128)
    n = np.arange(N)
    m = n.reshape((N, 1))
    W = np.exp(-2j * np.pi * m * n / N)
    return (W @ x) / N


def _bit_reverse_permutation(x):
    N = len(x)
    if N <= 1:
        return np.asarray(x, dtype=np.complex128)
    if np.log2(N) % 1 != 0:
        raise ValueError("DIT FFT wymaga długości będącej potęgą 2.")
    n_bits = int(np.log2(N))
    out = np.empty(N, dtype=np.complex128)
    for i in range(N):
        rev = 0
        v = i
        for _ in range(n_bits):
            rev = (rev << 1) | (v & 1)
            v >>= 1
        out[rev] = x[i]
    return out


def dit_fft(x):
    x = np.asarray(x, dtype=np.complex128)
    N = len(x)
    if N == 0:
        return np.array([], dtype=np.complex128)
    if np.log2(N) % 1 != 0:
        raise ValueError("DIT FFT wymaga długości będącej potęgą 2.")
    X = _bit_reverse_permutation(x)
    log2N = int(np.log2(N))
    for s in range(1, log2N + 1):
        block = 2 ** s
        half = block // 2
        twiddle = np.exp(-2j * np.pi * np.arange(half) / block)
        for k in range(0, N, block):
            even = X[k:k + half].copy()
            odd_tw = X[k + half:k + block] * twiddle
            X[k:k + half] = even + odd_tw
            X[k + half:k + block] = even - odd_tw
    return X / N


def dct2_definition(x):
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if N == 0:
        return np.array([], dtype=np.float64)
    n = np.arange(N)
    m = np.arange(N).reshape((N, 1))
    cos_mat = np.cos(np.pi * (2 * n + 1) * m / (2.0 * N))
    c = np.where(m.flatten() == 0, np.sqrt(1.0 / N), np.sqrt(2.0 / N))
    return c * (cos_mat @ x)


def fct2(x):
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if N == 0:
        return np.array([], dtype=np.float64)
    if np.log2(N) % 1 != 0:
        raise ValueError("FCT II wymaga długości będącej potęgą 2.")
    half = N // 2
    y = np.empty(N, dtype=np.float64)
    for n in range(half):
        y[n] = x[2 * n]
        y[N - 1 - n] = x[2 * n + 1]
    Y = dit_fft(y) * N
    m = np.arange(N)
    c = np.where(m == 0, np.sqrt(1.0 / N), np.sqrt(2.0 / N))
    twiddle = np.exp(-1j * np.pi * m / (2.0 * N))
    return c * np.real(twiddle * Y)


METHOD_NAMES = {
    'dft': 'DFT (z definicji)',
    'dit_fft': 'DIT FFT',
    'dct2': 'DCT II (z definicji)',
    'fct2': 'FCT II',
}

COMPARISON_PAIRS = {
    'dft': 'dit_fft',
    'dit_fft': 'dft',
    'dct2': 'fct2',
    'fct2': 'dct2',
}


def run_transform(signal_values, method):
    x = np.asarray(signal_values)
    N = len(x)
    if N < 2 or N > 1024:
        raise ValueError("Liczba próbek musi wynosić od 2 do 1024.")
    if np.log2(N) % 1 != 0:
        raise ValueError("Liczba próbek musi być potęgą 2.")
    if method == 'dft':
        start = time.perf_counter()
        result = dft_definition(x)
        elapsed = time.perf_counter() - start
    elif method == 'dit_fft':
        start = time.perf_counter()
        result = dit_fft(x)
        elapsed = time.perf_counter() - start
    elif method == 'dct2':
        if np.iscomplexobj(x):
            raise ValueError("DCT II wymaga sygnału rzeczywistego.")
        start = time.perf_counter()
        result = dct2_definition(x.real)
        elapsed = time.perf_counter() - start
    elif method == 'fct2':
        if np.iscomplexobj(x):
            raise ValueError("FCT II wymaga sygnału rzeczywistego.")
        start = time.perf_counter()
        result = fct2(x.real)
        elapsed = time.perf_counter() - start
    else:
        raise ValueError(f"Nieznana metoda: {method}")
    return result, elapsed, N