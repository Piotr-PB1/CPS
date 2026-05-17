import numpy as np
import matplotlib.pyplot as plt

def uniform_quantization_with_clipping(signal, num_bits):
    """
    Kwantyzacja równomierna z obcięciem.
    
    Args:
        signal: Wejściowa tablica numpy.
        num_bits: Liczba bitów kwantyzacji (np. 8).
        min_val: Dolna granica obcięcia.
        max_val: Górna granica obcięcia.
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    
    # 1. Obcięcie (Clipping/Saturation)
    clipped_signal = np.clip(signal, min_val, max_val)
    
    # 2. Obliczenie liczby poziomów i kroku kwantyzacji
    num_levels = 2**num_bits
    step = (max_val - min_val) / (num_levels - 1)
    
    # 3. Kwantyzacja (skalowanie, zaokrąglanie, przeskalowanie z powrotem)
    quantized_signal = np.round((clipped_signal - min_val) / step) * step + min_val
    
    return quantized_signal

# --- Przykład użycia ---
# Tworzymy sygnał (np. sinusoida + szum)
T = np.linspace(0, 1, 200)
signal = 1.0 * np.sin(2 * np.pi * T)

# Parametry
num_bits = 2  # 8 poziomów ($2^3$)

# Kwantyzacja
quantized = uniform_quantization_with_clipping(signal, num_bits)

# Wizualizacja
plt.figure(figsize=(10, 6))
plt.plot(T, signal, label='Sygnał oryginalny', alpha=0.5)
plt.plot(T, quantized, label='Sygnał kwantowany (z obcięciem)', marker='o', linestyle='--')
plt.axhline(y=np.max(signal), color='r', linestyle='-', label='Zakres obcięcia')
plt.axhline(y=np.min(signal), color='r', linestyle='-')
plt.legend()
plt.title(f'Kwantyzacja równomierna {num_bits} bitów')
plt.grid(True)
plt.show()
