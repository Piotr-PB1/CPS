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
            # BUTTERFLY OPERATION - podstawowa jednostka FFT
            X[k:k + half] = even + odd_tw
            X[k + half:k + block] = even - odd_tw
    return X / N


def idft_definition(X):
    X = np.asarray(X, dtype=np.complex128)
    N = len(X)
    if N == 0:
        return np.array([], dtype=np.complex128)
    n = np.arange(N)
    m = n.reshape((N, 1))
    W = np.exp(2j * np.pi * m * n / N)
    return (W @ X)


def idit_fft(X):
    X = np.asarray(X, dtype=np.complex128)
    N = len(X)
    if N == 0:
        return np.array([], dtype=np.complex128)
    if np.log2(N) % 1 != 0:
        raise ValueError("IFFT wymaga długości będącej potęgą 2.")
    X = _bit_reverse_permutation(X)
    log2N = int(np.log2(N))
    for s in range(1, log2N + 1):
        block = 2 ** s
        half = block // 2
        twiddle = np.exp(2j * np.pi * np.arange(half) / block)
        for k in range(0, N, block):
            even = X[k:k + half].copy()
            odd_tw = X[k + half:k + block] * twiddle
            X[k:k + half] = even + odd_tw
            X[k + half:k + block] = even - odd_tw
    return X


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


def idct2_definition(X):
    X = np.asarray(X, dtype=np.float64)
    N = len(X)
    if N == 0:
        return np.array([], dtype=np.float64)
    
    n = np.arange(N)
    m = np.arange(N).reshape((N, 1))
    c_n = np.where(n == 0, np.sqrt(1.0 / N), np.sqrt(2.0 / N))
    cos_mat = np.cos(np.pi * n * (2 * m + 1) / (2.0 * N))
    return (c_n * X) @ cos_mat


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


def ifct2(X):
    X = np.asarray(X, dtype=np.float64)
    N = len(X)
    if N == 0:
        return np.array([], dtype=np.float64)
    if np.log2(N) % 1 != 0:
        raise ValueError("IFCT II wymaga długości będącej potęgą 2.")
    
    m = np.arange(N)
    c = np.where(m == 0, np.sqrt(1.0 / N), np.sqrt(2.0 / N))
    twiddle = np.exp(1j * np.pi * m / (2.0 * N))
    Y = twiddle * X / c
    
    Y = idit_fft(Y) / N
    
    half = N // 2
    x = np.empty(N, dtype=np.float64)
    for n in range(half):
        x[2 * n] = np.real(Y[n])
        x[2 * n + 1] = np.real(Y[N - 1 - n])
    return x


METHOD_NAMES = {
    'dft': 'DFT ',
    'idft': 'IDFT - odwrotna',
    'dit_fft': 'DIT FFT',
    'idit_fft': 'IFFT - odwrotna FFT',
    'dct2': 'DCT II',
    'idct2': 'IDCT II (odwrotna)',
    'fct2': 'FCT II',
    'ifct2': 'IFCT II (odwrotna)',
}

COMPARISON_PAIRS = {
    'dft': 'idft',
    'idft': 'dft',
    'dit_fft': 'idit_fft',
    'idit_fft': 'dit_fft',
    'dct2': 'idct2',
    'idct2': 'dct2',
    'fct2': 'ifct2',
    'ifct2': 'fct2',
}


def run_transform(signal_values, method):
    x = np.asarray(signal_values)
    N = len(x)
    if method in ['dit_fft', 'idit_fft', 'ifct2', 'fct2']:
        if N < 2 or N > 1024:
            raise ValueError("Liczba próbek musi wynosić od 2 do 1024.")
        if np.log2(N) % 1 != 0:
            raise ValueError("Liczba próbek musi być potęgą 2.")
    
    if method == 'dft':
        start = time.perf_counter()
        result = dft_definition(x)
        elapsed = time.perf_counter() - start
    elif method == 'idft':
        start = time.perf_counter()
        result = idft_definition(x)
        elapsed = time.perf_counter() - start
    elif method == 'dit_fft':
        start = time.perf_counter()
        result = dit_fft(x)
        elapsed = time.perf_counter() - start
    elif method == 'idit_fft':
        start = time.perf_counter()
        result = idit_fft(x)
        elapsed = time.perf_counter() - start
    elif method == 'dct2':
        if np.iscomplexobj(x):
            raise ValueError("DCT II wymaga sygnału rzeczywistego.")
        start = time.perf_counter()
        result = dct2_definition(x.real)
        elapsed = time.perf_counter() - start
    elif method == 'idct2':
        if np.iscomplexobj(x):
            raise ValueError("IDCT II wymaga sygnału rzeczywistego.")
        start = time.perf_counter()
        result = idct2_definition(x.real)
        elapsed = time.perf_counter() - start
    elif method == 'fct2':
        if np.iscomplexobj(x):
            raise ValueError("FCT II wymaga sygnału rzeczywistego.")
        start = time.perf_counter()
        result = fct2(x.real)
        elapsed = time.perf_counter() - start
    elif method == 'ifct2':
        if np.iscomplexobj(x):
            raise ValueError("IFCT II wymaga sygnału rzeczywistego.")
        start = time.perf_counter()
        result = ifct2(x.real)
        elapsed = time.perf_counter() - start
    else:
        raise ValueError(f"Nieznana metoda: {method}")
    return result, elapsed, N