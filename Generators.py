import copy
import numpy as np
import Signal as sg


def _find_signal_class(index):
    n = index + 1
    for name in (f"S{n}", f"S{n:02d}"):
        cls = getattr(sg, name, None)
        if cls is not None:
            return cls
    raise ValueError(f"Brak klasy sygnału dla indeksu {index}")


def create_signal(signal_index, A=0, T=0, t1=0, d=0, kw=0, ts=0, ns=0, n1=0, p=0, sampling=1000):
    cls = _find_signal_class(signal_index)
    sig_name = cls.__name__

    if sig_name in ("S1", "S2"):
        obj = cls(A, t1, d, sampling)
    elif sig_name in ("S3", "S4", "S5"):
        obj = cls(A, T, t1, d, sampling)
    elif sig_name in ("S6", "S7", "S8"):
        obj = cls(A, T, t1, d, kw, sampling)
    elif sig_name == "S9":
        obj = cls(A, t1, d, ts, sampling)
    elif sig_name == "S10":
        obj = cls(A, n1, ns, sampling)
    elif sig_name == "S11":
        obj = cls(A, t1, d, p, sampling)
    else:
        obj = cls(A=A, T=T, t1=t1, d=d, kw=kw, ts=ts, ns=ns, n1=n1, p=p, sampling=sampling)

    obj.generate_signal()
    return obj


def quantize_signal(signal, bits):
    bits = int(bits)
    if bits < 1 or bits > 32:
        raise ValueError("Liczba bitów musi być między 1 a 32")

    signal._ensure_signal()
    quantized_signal = copy.deepcopy(signal)

    min_val = np.min(signal.signal)
    max_val = np.max(signal.signal)

    levels = 2 ** bits

    quantized_signal.signal = np.round(
        (signal.signal - min_val) / (max_val - min_val) * (levels - 1)
    ) * (max_val - min_val) / (levels - 1) + min_val

    quantized_signal.quantization_signal = signal.signal
    quantized_signal.info_text = getattr(signal, 'info_text', '') + f" [Kwantyzacja: {bits} bitów]"

    return quantized_signal


def extrapolate_signal(s, method_type, range_sinc=1):
    s._ensure_signal()
    extrapolated_signal = copy.deepcopy(s)

    if method_type == "zero":
        extrapolated_signal.signal = np.concatenate([s.signal, np.zeros(len(s.signal))])
    elif method_type == "sinc":
        range_sinc = int(range_sinc)
        extended_signal = s.signal.copy()

        for i_ext in range(len(s.signal)):
            sinc_sum = 0
            for k in range(-range_sinc, range_sinc + 1):
                if 0 <= i_ext + k < len(s.signal):
                    if k != 0:
                        sinc_sum += s.signal[i_ext + k] * np.sinc(k)
            extended_signal[i_ext] += sinc_sum * 0.1

        extrapolated_signal.signal = np.concatenate([extended_signal, np.zeros(len(s.signal))])

    if not hasattr(extrapolated_signal, 'info_text'):
        extrapolated_signal.info_text = ""

    if method_type == "sinc":
        extrapolated_signal.info_text += f" [Ekstrapolacja: {method_type} ({range_sinc})]"
    else:
        extrapolated_signal.info_text += f" [Ekstrapolacja: {method_type}]"

    return extrapolated_signal
