import tkinter as tk
import Signal as sg
from tkinter import messagebox, filedialog, ttk
import matplotlib.pyplot as plt
import numpy as np
import copy
from Filters import (convolution, design_filter, filter_signal, show_filter_comparison,
                     cross_correlation, radar_delay_measurement, show_correlation_analysis,
                     default_radar_search_window)
from Generators import create_signal, quantize_signal, extrapolate_signal
import Transformation as tr

SYGNALY = [
    "szum o rozkładzie jednostajnym",
    "szum gaussowski",
    "sygnał sinusoidalny",
    "sygnał sinusoidalny wyprostowany jednopołówkowo",
    "sygnał sinusoidalny wyprostowany dwupołówkowo",
    "sygnał prostokątny",
    "sygnał prostokątny symetryczny",
    "sygnał trójkątny",
    "skok jednostkowy",
    "impuls jednostkowy",
    "szum impulsowy"
]

PARAMS_NEEDED = [
    (True, False, True, True, False, False, False, False, False),  # S1
    (True, False, True, True, False, False, False, False, False),  # S2
    (True, True,  True, True, False, False, False, False, False),  # S3
    (True, True,  True, True, False, False, False, False, False),  # S4
    (True, True,  True, True, False, False, False, False, False),  # S5
    (True, True,  True, True, True, False, False, False, False),   # S6
    (True, True,  True, True, True, False, False, False, False),   # S7
    (True, True,  True, True, True, False, False, False, False),   # S8
    (True, False, True, True, False, True, False, False, False),   # S9
    (True, False, False, False, False, False, True, True, False),  # S10
    (True, False, True, True, False, False, False, False, True),   # S11
]

PARAM_LABELS = ['A', 'T', 't1', 'd', 'kw', 'ts', 'ns', 'n1', 'p']

list_of_signals = []
signal_frames = []


def do_math(sig1_idx, sig2_idx, operation):
    if sig1_idx < 0 or sig2_idx < 0:
        messagebox.showerror("Błąd", "Wybierz oba sygnały!")
        return
    if not operation:
        messagebox.showerror("Błąd", "Wybierz operację!")
        return

    signal1 = list_of_signals[sig1_idx]
    signal2 = list_of_signals[sig2_idx]

    if not sg._same_sampling(signal1, signal2):
        messagebox.showerror(
            "Błąd",
            "Sygnały muszą mieć takie samo próbkowanie (tę samą częstotliwość próbkowania)."
        )
        return

    op_labels = {
        'dodawanie': ('+', 'Dodawanie'),
        'odejmowanie': ('-', 'Odejmowanie'),
        'mnozenie': ('*', 'Mnożenie'),
        'dzielenie': ('/', 'Dzielenie'),
    }
    if operation not in op_labels:
        messagebox.showerror("Błąd", f"Nieznana operacja: {operation}")
        return

    try:
        t_common, result, fs = sg.apply_arithmetic_operation(signal1, signal2, operation)
        sym, label = op_labels[operation]
        text = f"[{label}: {signal1.info_text or 'Sig1'} {sym} {signal2.info_text or 'Sig2'}]"

        result_obj = sg.Signal.from_array(t_common, result, sampling=fs)
        result_obj.t1 = float(t_common[0])
        result_obj.d = float(t_common[-1] - t_common[0]) if len(t_common) > 1 else 0.0
        result_obj.info_text = text
        list_of_signals.append(result_obj)
        update_signals_display()
        messagebox.showinfo(
            "Sukces",
            f"{label} wykonane.\n"
            f"Zakres wyniku: t ∈ [{t_common[0]:.4g}, {t_common[-1]:.4g}] s\n"
            f"Liczba próbek: {len(result)}"
        )
    except ValueError as e:
        if str(e) == "SAME_SAMPLING":
            messagebox.showerror(
                "Błąd",
                "Sygnały muszą mieć takie samo próbkowanie (tę samą częstotliwość próbkowania)."
            )
        else:
            messagebox.showerror("Błąd", str(e))
    except Exception as e:
        messagebox.showerror("Błąd", f"Błąd operacji: {str(e)}")


def get_signal_type_index(sig):
    class_name = sig.__class__.__name__
    mapping = {
        'S1': 0, 'S2': 1, 'S3': 2, 'S4': 3, 'S5': 4,
        'S6': 5, 'S7': 6, 'S8': 7, 'S9': 8, 'S10': 9, 'S11': 10
    }
    if class_name in mapping:
        return mapping[class_name]
    if hasattr(sig, 'signal_type') and sig.signal_type is not None:
        return sig.signal_type
    return -1


def build_info_from_signal(sig):
    idx = get_signal_type_index(sig)
    if idx < 0 or idx >= len(SYGNALY):
        sig_name = "Nieznany"
    else:
        sig_name = SYGNALY[idx]

    parts = [sig_name]
    needed = PARAMS_NEEDED[idx] if idx >= 0 else [False]*9

    attrs = {
        'A': sig.A, 'T': sig.T, 't1': sig.t1, 'd': sig.d,
        'kw': sig.kw, 'ts': sig.ts, 'ns': sig.ns, 'n1': sig.n1, 'p': sig.p
    }
    for i, label in enumerate(PARAM_LABELS):
        if needed[i]:
            val = attrs[label]
            if isinstance(val, float):
                parts.append(f"{label}={val:.3g}")
            else:
                parts.append(f"{label}={val}")
    parts.append(f"sampling={sig.sampling:.0f}")
    return " ".join(parts)


def signal_params_window(signal, parent):
    window = tk.Toplevel(parent)
    window.title("Parametry sygnału")
    window.geometry("450x250")

    params = [
        f"Średnia wartość sygnału: {signal.mean_value():.2f}",
        f"Średnia wartość bezwzględna sygnału: {signal.mean_abs_value():.2f}",
        f"Wariancja sygnału: {signal.variance():.2f}",
        f"Wartość skuteczna (RMS) sygnału: {signal.rms_value():.2f}",
        f"Moc sygnału: {signal.power():.2f}"
    ]

    for param in params:
        tk.Label(window, text=param, justify=tk.LEFT).pack(fill=tk.X, padx=10, pady=5)


def save_signal_bin(idx):
    try:
        filename = filedialog.asksaveasfilename(defaultextension=".bin",
                                                filetypes=[("Binary", "*.bin"), ("All", "*.*")])
        if filename:
            list_of_signals[idx].save_to_bin(filename)
            messagebox.showinfo("Sukces", f"Sygnał zapisany do {filename}")
    except Exception as e:
        messagebox.showerror("Błąd", f"Błąd zapisu: {str(e)}")


def save_signal_txt(idx):
    try:
        filename = filedialog.asksaveasfilename(defaultextension=".txt",
                                                filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if filename:
            list_of_signals[idx].save_to_txt(filename)
            messagebox.showinfo("Sukces", f"Sygnał zapisany do {filename}")
    except Exception as e:
        messagebox.showerror("Błąd", f"Błąd zapisu: {str(e)}")


def load_signal():
    try:
        filename = filedialog.askopenfilename(filetypes=[("Binary", "*.bin"), ("Text", "*.txt"), ("All", "*.*")])
        if filename:
            if filename.endswith('.bin'):
                sig = sg.Signal.load_from_bin(filename)
            else:
                sig = sg.Signal.load_from_txt(filename)
            sig.info_text = build_info_from_signal(sig)
            list_of_signals.append(sig)
            update_signals_display()
            messagebox.showinfo("Sukces", f"Sygnał wczytany z {filename}")
    except Exception as e:
        messagebox.showerror("Błąd", f"Błąd odczytu: {str(e)}")


def quantize_signal_dialog(idx):
    def apply_quantization():
        try:
            bits = int(bits_entry.get())
            quantized = quantize_signal(list_of_signals[idx], bits)
            quantized.info_text = build_info_from_signal(quantized) + f" [Kwantyzacja: {bits}b]"
            list_of_signals.append(quantized)
            update_signals_display()
            messagebox.showinfo("Sukces", f"Sygnał skwantyzowany na {bits} bitów")
            quant_window.destroy()
        except ValueError as e:
            messagebox.showerror("Błąd", f"Błąd: {str(e)}")

    quant_window = tk.Toplevel()
    quant_window.title("Kwantyzacja sygnału")
    tk.Label(quant_window, text="Liczba bitów (1-32):").pack(padx=10, pady=5)
    bits_entry = tk.Entry(quant_window, width=10)
    bits_entry.insert(0, "8")
    bits_entry.pack(padx=10, pady=5)
    tk.Button(quant_window, text="Zastosuj", command=apply_quantization, bg='lightgreen').pack(padx=10, pady=10)


def extrapolate_signal_dialog(idx):
    def apply_extrapolation():
        try:
            method = method_var.get()
            range_sinc = int(range_entry.get()) if method == "sinc" else 1
            extrapolated = extrapolate_signal(list_of_signals[idx], method, range_sinc)
            extrapolated.info_text = build_info_from_signal(extrapolated)
            list_of_signals.append(extrapolated)
            update_signals_display()
            messagebox.showinfo("Sukces", f"Sygnał ekstrapolowany metodą: {method}")
            extrap_window.destroy()
        except Exception as e:
            messagebox.showerror("Błąd", f"Błąd: {str(e)}")

    extrap_window = tk.Toplevel()
    extrap_window.title("Ekstrapolacja sygnału")
    tk.Label(extrap_window, text="Metoda:").pack(padx=10, pady=5)
    method_var = tk.StringVar(value="zero")
    tk.Radiobutton(extrap_window, text="Zero padding", variable=method_var, value="zero").pack()
    tk.Radiobutton(extrap_window, text="Sinc interpolation", variable=method_var, value="sinc").pack()
    tk.Label(extrap_window, text="Zasięg sinc (jeśli wybrany):").pack(padx=10, pady=5)
    range_entry = tk.Entry(extrap_window, width=10)
    range_entry.insert(0, "1")
    range_entry.pack(padx=10, pady=5)
    tk.Button(extrap_window, text="Zastosuj", command=apply_extrapolation, bg='lightgreen').pack(padx=10, pady=10)


def perform_convolution(sig1_idx, sig2_idx):
    try:
        if sig1_idx < 0 or sig2_idx < 0:
            messagebox.showerror("Błąd", "Wybierz oba sygnały!")
            return

        sig1 = list_of_signals[sig1_idx]
        sig2 = list_of_signals[sig2_idx]

        sig1._ensure_signal()
        sig2._ensure_signal()

        result = convolution(sig1.signal, sig2.signal, mode='full')

        if abs(sig1.sampling - sig2.sampling) > 1e-9:
            messagebox.showerror("Błąd", "Sygnały muszą mieć tę samą częstotliwość próbkowania!")
            return

        t1 = sig1.t[0] if sig1.t is not None else sig1.t1
        t2 = sig2.t[0] if sig2.t is not None else sig2.t1
        t_result = t1 + t2 + np.arange(len(result), dtype=float) / sig1.sampling

        result_obj = sg.Signal.from_array(
            t_result,
            result,
            sampling=sig1.sampling
        )
        result_obj.info_text = f"[Splot: {sig1.info_text or 'Sig1'} * {sig2.info_text or 'Sig2'}]"

        list_of_signals.append(result_obj)
        update_signals_display()

        messagebox.showinfo("Sukces", f"Splot obliczony!\nDługość wyniku: {len(result)}")

    except Exception as e:
        messagebox.showerror("Błąd", f"Błąd podczas obliczania splotu: {str(e)}")


def design_and_apply_filter():
    try:
        sig_idx = filter_signal_combo.current()
        if sig_idx < 0:
            messagebox.showerror("Błąd", "Wybierz sygnał do filtracji!")
            return

        M = int(filter_order_entry.get())
        window_type = filter_window_var.get()
        filter_type = filter_type_var.get()

        if M % 2 == 0:
            messagebox.showerror("Błąd", "Rząd filtru M musi być nieparzysty!")
            return

        sig = list_of_signals[sig_idx]
        sig._ensure_signal()
        fs = sig.sampling
        nyquist_freq = fs / 2

        if filter_type == 'lowpass':
            fc = float(filter_cutoff_entry.get())
            if fc <= 0 or fc >= nyquist_freq:
                messagebox.showerror(
                    "Błąd",
                    f"Częstotliwość cięcia musi być z zakresu (0, {nyquist_freq:.1f}) Hz!\n"
                    f"Dla tego sygnału fs={fs} Hz"
                )
                return
            K = fs / fc
            h = design_filter(M, K=K, window_type=window_type, filter_type='lowpass')
            filter_desc = f"dolnoprzepustowy fc={fc}Hz"
        elif filter_type == 'bandpass':
            f_center = float(filter_center_entry.get())
            bandwidth = float(filter_bandwidth_entry.get())
            if f_center <= 0 or f_center >= nyquist_freq:
                messagebox.showerror(
                    "Błąd",
                    f"Częstotliwość środkowa musi być z zakresu (0, {nyquist_freq:.1f}) Hz!"
                )
                return
            if bandwidth <= 0 or bandwidth >= 2 * min(f_center, nyquist_freq - f_center):
                messagebox.showerror(
                    "Błąd",
                    f"Szerokość pasma musi być dodatnia i nie większa niż {2*min(f_center, nyquist_freq-f_center):.1f} Hz"
                )
                return
            h = design_filter(M, filter_type='bandpass', window_type=window_type,
                              f_center=f_center, fs=fs, bandwidth=bandwidth)
            filter_desc = f"środkowoprzepustowy f0={f_center}Hz, BW={bandwidth}Hz"
        else:
            messagebox.showerror("Błąd", f"Nieobsługiwany typ filtru: {filter_type}")
            return

        filtered = filter_signal(sig.signal, h, compensate_delay=True)

        result_obj = sg.Signal.from_array(
            sig.t if sig.t is not None else np.arange(len(filtered)),
            filtered,
            sampling=fs
        )
        result_obj.info_text = f"[{filter_desc} M={M} {window_type}]"
        list_of_signals.append(result_obj)
        update_signals_display()

        show_filter_comparison(sig.signal, filtered, h)

        messagebox.showinfo(
            "Sukces",
            f"Filtr zastosowany!\n{filter_desc}\nDługość wyniku: {len(filtered)}"
        )

    except ValueError as e:
        messagebox.showerror("Błąd", f"Błędna wartość: {str(e)}")
    except Exception as e:
        messagebox.showerror("Błąd", f"Błąd podczas filtracji: {str(e)}")


def perform_transform():
    try:
        idx = transform_signal_combo.current()
        if idx < 0:
            messagebox.showerror("Błąd", "Wybierz sygnał do transformacji!")
            return

        sig = list_of_signals[idx]
        sig._ensure_signal()
        if sig.signal is None or len(sig.signal) == 0:
            messagebox.showerror("Błąd", "Sygnał nie zawiera próbek.")
            return

        method = transform_method_var.get()
        mode = transform_mode_var.get()
        if method not in tr.METHOD_NAMES:
            messagebox.showerror("Błąd", "Wybierz metodę transformacji!")
            return
        if mode not in ('W1', 'W2'):
            messagebox.showerror("Błąd", "Wybierz tryb wykresu (W1 lub W2)!")
            return

        result, elapsed, N = tr.run_transform(sig.signal, method)
        fs = float(sig.sampling)
        
        if method in ('dct2', 'fct2', 'idct2', 'ifct2'):
            freq_axis = np.arange(N) * fs / (2.0 * N)
            X = result
            shift_note = ""
        else:
            freqs = np.fft.fftfreq(N, d=1.0 / fs)
            if transform_shift_var.get():
                X = np.fft.fftshift(result)
                freq_axis = np.fft.fftshift(freqs)
                shift_note = ", shifted"
            else:
                X = result
                freq_axis = freqs
                shift_note = ""

        result_obj = sg.Signal.from_array(freq_axis, X, t1=float(freq_axis[0]) if N > 0 else 0.0,
                                          sampling=sig.sampling, discrete_signal=True)
        result_obj.info_text = f"[{tr.METHOD_NAMES[method]}, N={N}, {mode}{shift_note}, czas={elapsed:.6f}s]"
        list_of_signals.append(result_obj)
        update_signals_display()

        show_complex_plot(X, freq_axis, result_obj.info_text, mode)
        messagebox.showinfo("Sukces", f"Transformacja wykonana w {elapsed:.6f} s")
    except Exception as e:
        messagebox.showerror("Błąd", f"Błąd podczas transformacji: {str(e)}")


def compute_correlation():
    try:
        sig1_idx = corr_signal1_combo.current()
        sig2_idx = corr_signal2_combo.current()

        if sig1_idx < 0 or sig2_idx < 0:
            messagebox.showerror("Błąd", "Wybierz oba sygnały!")
            return

        method = corr_method_var.get()

        sig1 = list_of_signals[sig1_idx]
        sig2 = list_of_signals[sig2_idx]

        sig1._ensure_signal()
        sig2._ensure_signal()

        result = cross_correlation(sig1.signal, sig2.signal, method=method)

        fs = sig1.sampling if abs(sig1.sampling - sig2.sampling) < 1e-9 else 1.0
        zero_lag_idx = len(sig2.signal) - 1
        t_result = (sig1.t1 - sig2.t1) + (np.arange(len(result)) - zero_lag_idx) / fs

        result_obj = sg.Signal.from_array(
            t_result,
            result,
            sampling=fs
        )
        result_obj.info_text = f" [Korelacja: {sig1.info_text or 'Sig1'} x {sig2.info_text or 'Sig2'}]"

        list_of_signals.append(result_obj)
        update_signals_display()

        show_correlation_analysis(
            sig1.signal,
            sig2.signal,
            result,
            title="Analiza korelacji radarowej"
        )

        messagebox.showinfo("Sukces", f"Korelacja obliczona!\nDługość wyniku: {len(result)}")

    except Exception as e:
        messagebox.showerror("Błąd", f"Błąd podczas obliczania korelacji: {str(e)}")


def simulate_radar():
    try:
        probe_idx = radar_probe_combo.current()
        reflected_idx = radar_reflected_combo.current()
        known_distance = float(radar_distance_entry.get())

        use_delay = reflected_idx < 0

        if probe_idx < 0:
            messagebox.showerror("Błąd", "Wybierz sygnał sondujący!")
            return

        if known_distance <= 0:
            messagebox.showerror("Błąd", "Odległość musi być dodatnia!")
            return

        probe_sig = list_of_signals[probe_idx]
        probe_sig._ensure_signal()

        radar_search_window = None
        echo_mode = False
        reflected_label = ""
        t1_delay_time = 0.0
        t1_delay_samples = 0
        t1_diff = 0.0

        if use_delay:
            delay_seconds = float(radar_delay_entry.get())
            if delay_seconds <= 0:
                messagebox.showerror("Błąd", "Opóźnienie musi być dodatnie!")
                return

            actual_sampling = float(probe_sig.sampling)
            probe_data = probe_sig.signal.astype(float)
            delay_samples = int(round(delay_seconds * actual_sampling))
            reflected_data = np.zeros_like(probe_data)

            if delay_samples < len(probe_data):
                reflected_data[delay_samples:] = probe_data[:len(probe_data) - delay_samples]
            else:
                messagebox.showwarning("Ostrzeżenie",
                    f"Opóźnienie ({delay_seconds:.4f} s) jest większe niż czas trwania sygnału!\n"
                    "Wynik może być niejednoznaczny.")
                reflected_data = np.concatenate(
                    [np.zeros(delay_samples), probe_data])[:len(probe_data)]

            reflected_label = (f"WYGENEROWANY z opóźnieniem {delay_seconds:.4f} s "
                               f"({delay_samples} próbek)")
            echo_mode = True

        else:
            reflected_sig = list_of_signals[reflected_idx]
            reflected_sig._ensure_signal()

            probe_t1 = sg.signal_start_time(probe_sig)
            refl_t1 = sg.signal_start_time(reflected_sig)

            try:
                probe_data, reflected_data, actual_sampling, t1_diff, t1_delay_samples = (
                    sg.place_reflected_on_probe_timeline(probe_sig, reflected_sig)
                )
            except ValueError as e:
                if str(e) == "SAME_SAMPLING":
                    messagebox.showerror(
                        "Błąd",
                        "Sygnały muszą mieć takie samo próbkowanie "
                        "(tę samą częstotliwość próbkowania)."
                    )
                else:
                    messagebox.showerror("Błąd", str(e))
                return

            if t1_diff < 0:
                messagebox.showwarning(
                    "Uwaga",
                    f"Sygnał zwrotny zaczyna się wcześniej niż sondujący (Δt1 = {t1_diff:.4f} s).\n"
                    "Przyjęto przesunięcie t1 = 0 próbek."
                )
                t1_diff = 0.0
                t1_delay_samples = 0

            echo_mode = True
            radar_search_window = None
            t1_delay_time = 0.0

            reflected_label = (
                f"{list_of_signals[reflected_idx].info_text or 'Sygnał zwrotny'}\n"
                f"  (t1_sonda={probe_t1:.4g} s, t1_zwrotny={refl_t1:.4g} s, "
                f"Δt1={max(0.0, t1_diff):.4g} s → {t1_delay_samples} próbek na osi sondy)"
            )

        detected_delay_time, detected_shift, corr, peak_idx, zero_lag = (
            radar_delay_measurement(
                probe_data,
                reflected_data,
                actual_sampling,
                method='auto',
                search_window=radar_search_window,
                echo_mode=echo_mode,
            )
        )

        corr_delay_time = detected_delay_time
        corr_lag_samples = detected_shift
        corr_delay_samples = abs(detected_shift)

        total_delay_time = corr_delay_time
        if total_delay_time <= 0:
            messagebox.showerror(
                "Błąd",
                "Nie można wyznaczyć prędkości dla zerowego opóźnienia korelacji."
            )
            return

        measured_speed = (2 * known_distance) / total_delay_time

        show_correlation_analysis(
            probe_data,
            reflected_data,
            corr,
            title="Korelacja sygnałów radarowych (Sensor radarowy)",
            search_window=radar_search_window,
            peak_idx=peak_idx,
            zero_lag=zero_lag,
        )

        t1_info = ""
        if not use_delay and t1_delay_samples > 0:
            t1_info = (
                f"Przesunięcie z t1 na osi sondy: {max(0.0, t1_diff):.6f} s "
                f"({t1_delay_samples} próbek) — uwzględnione w korelacji.\n\n"
            )

        result_text = (
            f"Wyniki pomiaru SENSORA RADAROWEGO:\n\n"
            f"Sygnał sondujący: {list_of_signals[probe_idx].info_text}\n"
            f"Sygnał zwrotny:   {reflected_label}\n\n"
            f"{t1_info}"
            f"Maksimum funkcji korelacji:\n"
            f"  indeks maksimum: {peak_idx}, indeks lag 0: {zero_lag}\n"
            f"  lag maksimum: {corr_lag_samples:+d} próbek\n"
            f"  opóźnienie do obliczenia prędkości: {corr_delay_time:.6f} s "
            f"({corr_delay_samples} próbek)\n\n"
            f"Podana odległość: {known_distance:.4f} km ({known_distance * 1000:.2f} m)\n"
            f"Wyznaczona prędkość: {measured_speed:.4f} km/s\n"
            f"  (2 × d / τ_korelacji)\n\n"
            f"Parametry:\n"
            f"  Częstość próbkowania: {actual_sampling:.0f} Hz\n"
            f"  Odległość wejściowa:  {known_distance} km\n"
            f"  Długość sygnału:      {len(probe_data)} próbek\n"
            f"  Długość korelacji:    {len(corr)} próbek"
        )

        messagebox.showinfo("Wynik pomiaru SENSORA RADAROWEGO", result_text)

    except ValueError as e:
        messagebox.showerror("Błąd", f"Błędna wartość: {str(e)}")
    except Exception as e:
        messagebox.showerror("Błąd", f"Błąd podczas pomiaru: {str(e)}")


def update_field_visibility():
    signal_type = signal_type_combo.current()
    if signal_type < 0:
        return
    fields = [
        (signal_A_entry, signal_A_label),
        (signal_T_entry, signal_T_label),
        (signal_t1_entry, signal_t1_label),
        (signal_d_entry, signal_d_label),
        (signal_kw_entry, signal_kw_label),
        (signal_ts_entry, signal_ts_label),
        (signal_ns_entry, signal_ns_label),
        (signal_n1_entry, signal_n1_label),
        (signal_p_entry, signal_p_label),
    ]
    for i, (entry, label) in enumerate(fields):
        if i < len(PARAMS_NEEDED[signal_type]) and PARAMS_NEEDED[signal_type][i]:
            entry.config(state=tk.NORMAL)
            label.config(fg='black')
        else:
            entry.config(state=tk.DISABLED)
            label.config(fg='gray')


def generate_signal():
    try:
        signal_type = signal_type_combo.current()
        if signal_type < 0:
            messagebox.showerror("Błąd", "Wybierz typ sygnału!")
            return

        params = {
            'A': float(signal_A_entry.get()) if signal_A_entry.get() and signal_A_entry.cget('state') == tk.NORMAL else 0,
            'T': float(signal_T_entry.get()) if signal_T_entry.get() and signal_T_entry.cget('state') == tk.NORMAL else 0,
            't1': float(signal_t1_entry.get()) if signal_t1_entry.get() and signal_t1_entry.cget('state') == tk.NORMAL else 0,
            'd': float(signal_d_entry.get()) if signal_d_entry.get() and signal_d_entry.cget('state') == tk.NORMAL else 0,
            'kw': float(signal_kw_entry.get()) if signal_kw_entry.get() and signal_kw_entry.cget('state') == tk.NORMAL else 0,
            'ts': float(signal_ts_entry.get()) if signal_ts_entry.get() and signal_ts_entry.cget('state') == tk.NORMAL else 0,
            'ns': int(signal_ns_entry.get()) if signal_ns_entry.get() and signal_ns_entry.cget('state') == tk.NORMAL else 0,
            'n1': int(signal_n1_entry.get()) if signal_n1_entry.get() and signal_n1_entry.cget('state') == tk.NORMAL else 0,
            'p': float(signal_p_entry.get()) if signal_p_entry.get() and signal_p_entry.cget('state') == tk.NORMAL else 0,
            'sampling': float(signal_sampling_entry.get()) if signal_sampling_entry.get() else 1000
        }

        sig = create_signal(signal_type, **params)
        sig.info_text = build_info_from_signal(sig)
        list_of_signals.append(sig)
        update_signals_display()
        messagebox.showinfo("Sukces", f"Sygnał '{SYGNALY[signal_type]}' wygenerowany!")

    except Exception as e:
        messagebox.showerror("Błąd", f"Błąd podczas generacji: {str(e)}")


def update_signals_display():
    print(f"DEBUG: update_signals_display called; list_of_signals={len(list_of_signals)}")
    for frame in signal_frames:
        try:
            frame.destroy()
        except Exception:
            pass
    signal_frames.clear()

    for i, sig in enumerate(list_of_signals):
        frame = tk.Frame(signals_frame, bg='lightgray', relief=tk.RIDGE, bd=1, height=50)
        frame.pack(fill=tk.X, padx=5, pady=2)
        frame.pack_propagate(False)

        left_frame = tk.Frame(frame, bg='lightgray')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=2)
        right_frame = tk.Frame(frame, bg='lightgray')
        right_frame.pack(side=tk.RIGHT, padx=5, pady=2)

        info_text = f"{i+1}. {sig.info_text or type(sig).__str__}"
        label = tk.Label(left_frame, text=info_text, bg='lightgray', justify=tk.LEFT, wraplength=400)
        label.pack(side=tk.LEFT, padx=5)

        tk.Button(right_frame, text="Wykres", width=7, font=('Arial', 8),
                 command=lambda idx=i: show_signal_plot(idx)).pack(side=tk.LEFT, padx=1)
        tk.Button(right_frame, text="Bin", width=5, font=('Arial', 8),
                 command=lambda idx=i: save_signal_bin(idx)).pack(side=tk.LEFT, padx=1)
        tk.Button(right_frame, text="Txt", width=5, font=('Arial', 8),
                 command=lambda idx=i: save_signal_txt(idx)).pack(side=tk.LEFT, padx=1)
        tk.Button(right_frame, text="Kwantyzacja", width=10, font=('Arial', 8),
                 command=lambda idx=i: quantize_signal_dialog(idx)).pack(side=tk.LEFT, padx=1)
        tk.Button(right_frame, text="Ekstrapolacja", width=10, font=('Arial', 8),
                 command=lambda idx=i: extrapolate_signal_dialog(idx)).pack(side=tk.LEFT, padx=1)
        tk.Button(right_frame, text="Parametry", width=7, font=('Arial', 8),
                 command=lambda idx=i: signal_params_window(list_of_signals[idx], right_frame)).pack(side=tk.LEFT, padx=1)
        tk.Button(right_frame, text="Usuń", width=5, font=('Arial', 8), fg='red',
                 command=lambda idx=i: delete_signal(idx)).pack(side=tk.LEFT, padx=1)
        signal_frames.append(frame)

    signal_list = [f"{i+1}. {s.info_text or type(s).__str__}" for i, s in enumerate(list_of_signals)]
    combos = [
        'filter_signal_combo', 'corr_signal1_combo', 'corr_signal2_combo',
        'radar_probe_combo', 'radar_reflected_combo', 'splot_signal1_combo', 'splot_signal2_combo',
        'operation_signal1_combo', 'operation_signal2_combo', 'transform_signal_combo'
    ]
    for name in combos:
        try:
            combo = globals().get(name)
            if combo is not None:
                combo['values'] = signal_list
        except Exception as e:
            print(f"DEBUG: failed to update combo {name}: {e}")

    # Ensure canvas scrollregion and layout are updated so new frames become visible
    try:
        if signals_frame is None:
            print("DEBUG: signals_frame is None")
            return
        signals_frame.update_idletasks()
        parent_canvas = getattr(signals_frame, 'master', None)
        if parent_canvas is not None and hasattr(parent_canvas, 'configure'):
            bbox = parent_canvas.bbox("all")
            print(f"DEBUG: canvas.bbox(all)={bbox}")
            parent_canvas.configure(scrollregion=bbox)
            try:
                parent_canvas.yview_moveto(0)
            except Exception:
                pass
        else:
            print("DEBUG: parent_canvas not found or not a canvas")
    except Exception as e:
        print(f"DEBUG: update_signals_display exception: {e}")


def show_complex_plot(X, freq_axis, title_str, mode):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(title_str, fontsize=12, fontweight='bold')

    if mode == 'W1':
        ax1.plot(freq_axis, np.real(X), 'b-', linewidth=1)
        ax1.set_ylabel('Re(X)')
        ax1.grid(True)
        ax1.legend(['Re(X)'])

        ax2.plot(freq_axis, np.imag(X), 'r-', linewidth=1)
        ax2.set_ylabel('Im(X)')
        ax2.grid(True)
        ax2.legend(['Im(X)'])
    else:
        ax1.plot(freq_axis, np.abs(X), 'b-', linewidth=1)
        ax1.set_ylabel('|X|')
        ax1.grid(True)
        ax1.legend(['|X|'])

        ax2.plot(freq_axis, np.angle(X), 'r-', linewidth=1)
        ax2.set_ylabel('arg(X)')
        ax2.grid(True)
        ax2.legend(['arg(X)'])

    ax2.set_xlabel('Częstotliwość [Hz]')
    plt.tight_layout()
    plt.show()


def show_signal_plot(idx):
    if 0 <= idx < len(list_of_signals):
        sig = list_of_signals[idx]
        sig._ensure_signal()
        if sig.signal is None:
            messagebox.showerror("Błąd", "Sygnał nie zawiera próbek.")
            return

        # Robust complex detection: dtype kind 'c' or any non-zero imaginary part
        arr = np.asarray(sig.signal)
        is_complex = False
        try:
            if hasattr(arr, 'dtype') and getattr(arr.dtype, 'kind', None) == 'c':
                is_complex = True
            else:
                # check imaginary component magnitude
                if np.any(np.abs(np.imag(arr)) > 1e-12):
                    is_complex = True
        except Exception:
            is_complex = np.iscomplexobj(arr)

        if is_complex:
            if sig.t is None:
                freq_axis = np.arange(len(arr), dtype=float)
            else:
                freq_axis = np.asarray(sig.t, dtype=float)
            show_complex_plot(arr, freq_axis, sig.info_text or type(sig).__str__, 'W1')
            return

        plt.figure(figsize=(10, 4))
        if sig.t is not None:
            plt.plot(sig.t, sig.signal, 'b-', linewidth=1)
            plt.xlabel('Czas')
        else:
            plt.plot(sig.signal, 'b-', linewidth=1)
            plt.xlabel('Próbka')
        plt.ylabel('Amplituda')
        plt.title(sig.info_text or type(sig).__str__)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def delete_signal(idx):
    list_of_signals.pop(idx)
    update_signals_display()


def run():
    global root, signal_type_combo, signal_A_entry, signal_T_entry, signal_t1_entry, signal_d_entry
    global signal_kw_entry, signal_ts_entry, signal_ns_entry, signal_n1_entry, signal_p_entry
    global signal_sampling_entry, signals_frame, filter_signal_combo, filter_type_var, filter_window_var
    global filter_order_entry, filter_cutoff_entry, corr_signal1_combo, corr_signal2_combo, corr_method_var
    global radar_probe_combo, radar_reflected_combo, radar_sampling_entry, radar_distance_entry, radar_delay_entry
    global splot_signal1_combo, splot_signal2_combo, transform_signal_combo, signal_A_label, signal_T_label, signal_t1_label
    global signal_d_label, signal_kw_label, signal_ts_label, signal_ns_label, signal_n1_label, signal_p_label
    global operation_signal1_combo, operation_signal2_combo
    global filter_center_entry, filter_bandwidth_entry, transform_method_var, transform_mode_var
    global transform_shift_var

    root = tk.Tk()
    root.title("Cyfrowe Przetwarzanie Sygnałów - Zadanie 3")
    root.geometry("1000x800")

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # ========== TAB 0: GENERACJA SYGNAŁÓW ==========
    tab_generate = ttk.Frame(notebook)
    notebook.add(tab_generate, text="Generacja sygnałów")

    params_frame = tk.Frame(tab_generate)
    params_frame.pack(fill=tk.X, padx=5, pady=5)

    tk.Label(params_frame, text="Typ sygnału:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    signal_type_combo = ttk.Combobox(params_frame, values=SYGNALY, state='readonly', width=40)
    signal_type_combo.grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
    signal_type_combo.bind('<<ComboboxSelected>>', lambda e: update_field_visibility())

    entries_info = [
        ("Amplituda (A):", 'A'),
        ("Okres (T):", 'T'),
        ("Czas początkowy (t1):", 't1'),
        ("Czas trwania (d):", 'd'),
        ("Współczynnik wypełnienia (kw):", 'kw'),
        ("Czas skoku (ts):", 'ts'),
        ("Indeks impulsu (ns):", 'ns'),
        ("Indeks próbki (n1):", 'n1'),
        ("Prawdopodobieństwo (p):", 'p'),
    ]

    row = 1
    entry_map = {}
    for label_text, param_name in entries_info:
        label = tk.Label(params_frame, text=label_text)
        label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        entry = tk.Entry(params_frame, width=15)
        entry.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=5)
        entry_map[param_name] = (entry, label)
        row += 1

    signal_A_entry, signal_A_label = entry_map['A']
    signal_T_entry, signal_T_label = entry_map['T']
    signal_t1_entry, signal_t1_label = entry_map['t1']
    signal_d_entry, signal_d_label = entry_map['d']
    signal_kw_entry, signal_kw_label = entry_map['kw']
    signal_ts_entry, signal_ts_label = entry_map['ts']
    signal_ns_entry, signal_ns_label = entry_map['ns']
    signal_n1_entry, signal_n1_label = entry_map['n1']
    signal_p_entry, signal_p_label = entry_map['p']

    tk.Label(params_frame, text="Częstość próbkowania [Hz]:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
    signal_sampling_entry = tk.Entry(params_frame, width=15)
    signal_sampling_entry.insert(0, "1000")
    signal_sampling_entry.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=5)
    params_frame.columnconfigure(1, weight=1)

    buttons_frame = tk.Frame(tab_generate)
    buttons_frame.pack(fill=tk.X, padx=5, pady=5)

    tk.Button(buttons_frame, text="Generuj sygnał", command=generate_signal,
             bg='lightgreen', height=2, width=20).pack(side=tk.LEFT, padx=5)
    tk.Button(buttons_frame, text="Wczytaj z pliku", command=load_signal,
             bg='lightblue', height=2, width=20).pack(side=tk.LEFT, padx=5)

    tk.Label(tab_generate, text="Wygenerowane sygnały:").pack(fill=tk.X, padx=5, pady=(10, 0))
    container = tk.Frame(tab_generate)
    container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    canvas = tk.Canvas(container)
    scrollbar = ttk.Scrollbar(container, orient='vertical', command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    def _on_frame_config(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    scrollable_frame.bind("<Configure>", _on_frame_config)
    window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')

    def _on_canvas_config(event):
        try:
            canvas.itemconfig(window_id, width=event.width)
        except Exception:
            pass

    canvas.bind('<Configure>', _on_canvas_config)
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    signals_frame = scrollable_frame
    tab_generate.rowconfigure(2, weight=1)
    tab_generate.columnconfigure(1, weight=1)

    # ========== TAB 1.25 operacja =======
    tab_operation = ttk.Frame(notebook)
    notebook.add(tab_operation, text="Operacje na sygnałach")
    tk.Label(tab_operation, text="Sygnał 1:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    operation_signal1_combo = ttk.Combobox(tab_operation, state='readonly', width=40)
    operation_signal1_combo.grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
    tk.Label(tab_operation, text="Sygnał 2:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    operation_signal2_combo = ttk.Combobox(tab_operation, state='readonly', width=40)
    operation_signal2_combo.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
    tk.Label(tab_operation, text="Operacja:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
    operation_var = ttk.Combobox(tab_operation, state='readonly', values=['dodawanie', 'odejmowanie', 'mnozenie', 'dzielenie'], width=20)
    operation_var.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

    tk.Button(tab_operation, text="Wykonaj operację", command=lambda: do_math(
             operation_signal1_combo.current(), operation_signal2_combo.current(), operation_var.get()),
             bg='lightyellow', height=2).grid(row=3, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=10)

    # ========== TAB 1.5: SPLOT ==========
    tab_splot = ttk.Frame(notebook)
    notebook.add(tab_splot, text="Splot")
    tk.Label(tab_splot, text="Sygnał 1:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    splot_signal1_combo = ttk.Combobox(tab_splot, state='readonly', width=40)
    splot_signal1_combo.grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
    tk.Label(tab_splot, text="Sygnał 2:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    splot_signal2_combo = ttk.Combobox(tab_splot, state='readonly', width=40)
    splot_signal2_combo.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
    tk.Button(tab_splot, text="Wykonaj splot", command=lambda: perform_convolution(
             splot_signal1_combo.current(), splot_signal2_combo.current()),
             bg='lightyellow', height=2).grid(row=2, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=10)

    # ========== TAB 2: TRANSFORMACJE (F1/T1) ==========
    tab_transform = ttk.Frame(notebook)
    notebook.add(tab_transform, text="Transformacje (F1/T1)")

    tk.Label(tab_transform, text="Sygnał do transformacji:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    transform_signal_combo = ttk.Combobox(tab_transform, state='readonly', width=40)
    transform_signal_combo.grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)

    tk.Label(tab_transform, text="Metoda transformacji:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    transform_method_var = tk.StringVar(value='dft')
    method_combo = ttk.Combobox(tab_transform, textvariable=transform_method_var,
                                 values=['dft', 'idft', 'dit_fft', 'idit_fft', 'dct2', 'idct2', 'fct2', 'ifct2'], state='readonly')
    method_combo.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)

    tk.Label(tab_transform, text="Tryb wyświetlenia (W1/W2):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
    transform_mode_var = tk.StringVar(value='W1')
    mode_combo = ttk.Combobox(tab_transform, textvariable=transform_mode_var,
                               values=['W1', 'W2'], state='readonly')
    mode_combo.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)

    tk.Label(tab_transform, text="Przesunięcie (DFT/FFT shift):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
    transform_shift_var = tk.BooleanVar(value=True)
    shift_check = tk.Checkbutton(tab_transform, variable=transform_shift_var)
    shift_check.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)

    tk.Button(tab_transform, text="Wykonaj transformację", command=perform_transform,
             bg='lightblue', height=2).grid(row=4, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=10)

    # ========== TAB 3: FILTRACJA (poprawiona) ==========
    tab_filter = ttk.Frame(notebook)
    notebook.add(tab_filter, text="Filtracja (O1/F1)")

    tk.Label(tab_filter, text="Sygnał do filtracji:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    filter_signal_combo = ttk.Combobox(tab_filter, state='readonly', width=40)
    filter_signal_combo.grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)

    tk.Label(tab_filter, text="Typ filtru:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    filter_type_var = tk.StringVar(value='bandpass')
    filter_type_combo = ttk.Combobox(tab_filter, textvariable=filter_type_var,
                                      values=['bandpass', 'lowpass'], state='readonly')
    filter_type_combo.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)

    tk.Label(tab_filter, text="Okno (O1=Hamming):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
    filter_window_var = tk.StringVar(value='hamming')
    window_combo = ttk.Combobox(tab_filter, textvariable=filter_window_var,
                                 values=['rectangular', 'hamming'], state='readonly')
    window_combo.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)

    tk.Label(tab_filter, text="Rząd filtru (M - nieparzyste):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
    filter_order_entry = tk.Entry(tab_filter, width=15)
    filter_order_entry.insert(0, "25")
    filter_order_entry.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)

    tk.Label(tab_filter, text="Częstotliwość odcięcia [Hz] (lowpass):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
    filter_cutoff_entry = tk.Entry(tab_filter, width=15)
    filter_cutoff_entry.insert(0, "100")
    filter_cutoff_entry.grid(row=4, column=1, sticky=tk.EW, padx=5, pady=5)

    tk.Label(tab_filter, text="Częstotliwość środkowa [Hz] (bandpass):").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
    filter_center_entry = tk.Entry(tab_filter, width=15)
    filter_center_entry.insert(0, "200")
    filter_center_entry.grid(row=5, column=1, sticky=tk.EW, padx=5, pady=5)

    tk.Label(tab_filter, text="Szerokość pasma [Hz] (bandpass):").grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
    filter_bandwidth_entry = tk.Entry(tab_filter, width=15)
    filter_bandwidth_entry.insert(0, "50")
    filter_bandwidth_entry.grid(row=6, column=1, sticky=tk.EW, padx=5, pady=5)

    tk.Button(tab_filter, text="Zastosuj filtr", command=design_and_apply_filter,
             bg='lightblue', height=2).grid(row=7, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=10)

    # ========== TAB 3: KORELACJA ==========
    tab_corr = ttk.Frame(notebook)
    notebook.add(tab_corr, text="Korelacja")
    tk.Label(tab_corr, text="Sygnał 1:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    corr_signal1_combo = ttk.Combobox(tab_corr, state='readonly', width=40)
    corr_signal1_combo.grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
    tk.Label(tab_corr, text="Sygnał 2:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    corr_signal2_combo = ttk.Combobox(tab_corr, state='readonly', width=40)
    corr_signal2_combo.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
    tk.Label(tab_corr, text="Metoda:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
    corr_method_var = tk.StringVar(value='direct')
    method_combo = ttk.Combobox(tab_corr, textvariable=corr_method_var,
                                 values=['direct', 'convolution'], state='readonly')
    method_combo.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
    tk.Button(tab_corr, text="Oblicz korelację", command=compute_correlation,
             bg='lightyellow', height=2).grid(row=3, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=10)

    # ========== TAB 4: SENSOR RADAROWY ==========
    tab_radar = ttk.Frame(notebook)
    notebook.add(tab_radar, text="Sensor radarowy")
    radar_params_frame = tk.Frame(tab_radar)
    radar_params_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    tk.Label(radar_params_frame, text="Sygnał sondujący (probe):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    radar_probe_combo = ttk.Combobox(radar_params_frame, state='readonly', width=40)
    radar_probe_combo.grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)

    tk.Label(radar_params_frame, text="Sygnał zwrotny (reflected):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    radar_reflected_combo = ttk.Combobox(radar_params_frame, state='readonly', width=40)
    radar_reflected_combo.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
    tk.Label(radar_params_frame,
             text="Dwa sygnały: zwrotny jest przesuwany o Δt1=(t1_zwrotny−t1_sonda) na osi sondy, "
                  "jak echo przy jednym sygnale. Korelacja daje τ, a dla podanej odległości d liczymy v=2d/τ.",
             fg='gray', font=('Arial', 8)).grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=5, pady=0)

    tk.Label(radar_params_frame, text="Odległość [km]:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
    radar_distance_entry = tk.Entry(radar_params_frame, width=15)
    radar_distance_entry.insert(0, "7500")
    radar_distance_entry.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)

    tk.Label(radar_params_frame, text="Opóźnienie [s] (jeśli brak sygnału zwrotnego):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
    radar_delay_entry = tk.Entry(radar_params_frame, width=15)
    radar_delay_entry.insert(0, "0.05")
    radar_delay_entry.grid(row=4, column=1, sticky=tk.EW, padx=5, pady=5)

    radar_sampling_entry = tk.Entry(radar_params_frame, width=1)
    radar_sampling_entry.insert(0, "1000")

    tk.Button(radar_params_frame, text="Wyznacz prędkość SENSOREM RADAROWYM", command=simulate_radar,
             bg='lightcoral', height=2, font=('Arial', 10, 'bold')).grid(row=5, column=0, columnspan=2,
             sticky=tk.EW, padx=5, pady=10)

    root.mainloop()


if __name__ == '__main__':
    run()
