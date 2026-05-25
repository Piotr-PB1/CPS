import tkinter as tk
import Signal as sg
from tkinter import messagebox, filedialog, ttk
import matplotlib.pyplot as plt
import numpy as np
import copy
from Filters import (convolution, design_filter, filter_signal, show_filter_comparison,
                     cross_correlation, radar_distance_measurement, show_correlation_analysis, find_delay)
from Generators import create_signal, quantize_signal, extrapolate_signal

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

# ===================================================
# POMOCNICZE DO BUDOWANIA NAZWY SYGNAŁU Z PARAMETRÓW
# ===================================================

def get_signal_type_index(sig):
    """Zwraca indeks typu sygnału (0..10) na podstawie nazwy klasy lub atrybutu signal_type."""
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
    """Tworzy opis sygnału: typ i parametry (na podstawie atrybutów obiektu)."""
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

# ============================================================================
# OPERACJE NA SYGNAŁACH - Zapis/Wczytywanie/Kwantyzacja/Ekstrapolacja
# ============================================================================

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

    for i, text in enumerate(params):
        label = tk.Label(window, text=text, font=("Arial", 11), anchor="w")
        label.grid(row=i, column=0, padx=10, pady=8, sticky="w")

    close_button = tk.Button(window, text="Zamknij", command=window.destroy)
    close_button.grid(row=len(params), column=0, pady=15)

def save_signal_bin(idx):
    """Zapisz sygnał do pliku binarnego"""
    try:
        if idx < 0 or idx >= len(list_of_signals):
            return
        file = filedialog.asksaveasfilename(
            defaultextension=".bin",
            filetypes=[("Signal files", "*.bin"), ("All files", "*.*")]
        )
        if file:
            list_of_signals[idx].save_to_bin(file)
            messagebox.showinfo("Sukces", f"Sygnał zapisany do:\n{file}")
    except Exception as e:
        messagebox.showerror("Błąd", f"Błąd zapisu: {str(e)}")

def save_signal_txt(idx):
    """Zapisz sygnał do pliku tekstowego"""
    try:
        if idx < 0 or idx >= len(list_of_signals):
            return
        file = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file:
            list_of_signals[idx].save_to_txt(file)
            messagebox.showinfo("Sukces", f"Sygnał zapisany do:\n{file}")
    except Exception as e:
        messagebox.showerror("Błąd", f"Błąd zapisu: {str(e)}")

def load_signal():
    """Wczytaj sygnał z pliku i nadaj mu opis z parametrami."""
    try:
        file = filedialog.askopenfilename(
            filetypes=[("Signal files", "*.bin"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file:
            if file.endswith('.bin'):
                sig = sg.Signal.load_from_bin(file)
            else:
                sig = sg.Signal.load_from_txt(file)
            
            sig.info_text = build_info_from_signal(sig)
            list_of_signals.append(sig)
            update_signals_display()
            messagebox.showinfo("Sukces", f"Sygnał wczytany z:\n{file}")
    except Exception as e:
        messagebox.showerror("Błąd", f"Błąd wczytywania: {str(e)}")

def quantize_signal_dialog(idx):
    """Dialog dla kwantyzacji"""
    try:
        if idx < 0 or idx >= len(list_of_signals):
            return
        dialog = tk.Toplevel(root)
        dialog.title("Kwantyzacja")
        dialog.geometry("300x150")
        tk.Label(dialog, text="Liczba bitów (1-32):").pack(pady=5)
        bits_entry = tk.Entry(dialog, width=15)
        bits_entry.insert(0, "8")
        bits_entry.pack(pady=5)
        
        def apply_quantize():
            try:
                bits = int(bits_entry.get())
                sig = list_of_signals[idx]
                sig_quant = quantize_signal(sig, bits)
                sig_quant.info_text = build_info_from_signal(sig_quant) + " [skwantowany]"
                list_of_signals.append(sig_quant)
                update_signals_display()
                dialog.destroy()
                messagebox.showinfo("Sukces", f"Sygnał kwantyzowany na {bits} bitów!")
            except ValueError as e:
                messagebox.showerror("Błąd", f"Błędna wartość: {str(e)}")
        tk.Button(dialog, text="Zastosuj kwantyzację", command=apply_quantize, 
                 bg='lightgreen', width=30).pack(pady=10)
    except Exception as e:
        messagebox.showerror("Błąd", f"Błąd kwantyzacji: {str(e)}")

def extrapolate_signal_dialog(idx):
    """Dialog dla ekstrapolacji"""
    try:
        if idx < 0 or idx >= len(list_of_signals):
            return
        dialog = tk.Toplevel(root)
        dialog.title("Ekstrapolacja")
        dialog.geometry("300x200")
        tk.Label(dialog, text="Metoda ekstrapolacji:").pack(pady=5)
        method_var = tk.StringVar(value='zero')
        tk.Radiobutton(dialog, text="Zero padding", variable=method_var, 
                      value='zero').pack(anchor=tk.W, padx=20)
        tk.Radiobutton(dialog, text="Sinc", variable=method_var, 
                      value='sinc').pack(anchor=tk.W, padx=20)
        tk.Label(dialog, text="Zasięg sinc (dla metody sinc):").pack(pady=5)
        range_entry = tk.Entry(dialog, width=15)
        range_entry.insert(0, "1")
        range_entry.pack(pady=5)
        
        def apply_extrapolate():
            try:
                method = method_var.get()
                range_sinc = int(range_entry.get()) if method == 'sinc' else 1
                sig = list_of_signals[idx]
                sig_ext = extrapolate_signal(sig, method, range_sinc)
                sig_ext.info_text = build_info_from_signal(sig_ext) + f" [ekstrapolowany {method}]"
                list_of_signals.append(sig_ext)
                update_signals_display()
                dialog.destroy()
                messagebox.showinfo("Sukces", f"Sygnał ekstrapolowany metodą '{method}'!")
            except ValueError as e:
                messagebox.showerror("Błąd", f"Błędna wartość: {str(e)}")
        tk.Button(dialog, text="Zastosuj ekstrapolację", command=apply_extrapolate, 
                 bg='lightblue', width=30).pack(pady=10)
    except Exception as e:
        messagebox.showerror("Błąd", f"Błąd ekstrapolacji: {str(e)}")

# ===========================================================================
# SPLOT
# ==========================================================================

def perform_convolution(s1, s2):
    """Wykonaj splot dwóch wybranych sygnałów"""
    try:
        sig1_idx = s1
        sig2_idx = s2

        if sig1_idx < 0 or sig2_idx < 0:
            messagebox.showerror("Błąd", "Wybierz oba sygnały do splotu!")
            return

        sig1 = list_of_signals[sig1_idx]
        sig2 = list_of_signals[sig2_idx]
        
        sig1._ensure_signal()
        sig2._ensure_signal()

        result = convolution(sig1.signal, sig2.signal)

        result_obj = sg.Signal.from_array(
            np.arange(len(result)),
            result,
            sampling=1.0
        )
        result_obj.info_text = f" [Splot: {sig1.info_text or 'Sig1'} * {sig2.info_text or 'Sig2'}]"
        
        list_of_signals.append(result_obj)
        update_signals_display()

        messagebox.showinfo("Sukces", f"Splot wykonany!\nDługość wyniku: {len(result)}")

    except Exception as e:
        messagebox.showerror("Błąd", f"Błąd podczas splotu: {str(e)}")

# ============================================================================
# FILTRACJA
# ============================================================================

def design_and_apply_filter():
    """Zaprojektuj i zastosuj filtr do wybranego sygnału"""
    try:
        signal_idx = filter_signal_combo.current()
        filter_type = filter_type_var.get()
        M = int(filter_order_entry.get())
        K = int(filter_cutoff_entry.get())
        window_type = filter_window_var.get()

        if signal_idx < 0:
            messagebox.showerror("Błąd", "Wybierz sygnał do filtracji!")
            return

        if M % 2 == 0 or M < 1:
            messagebox.showerror("Błąd", "Rząd filtru M musi być liczbą nieparzystą!")
            return

        if K < 1:
            messagebox.showerror("Błąd", "Parametr K musi być większy od 0!")
            return

        signal = list_of_signals[signal_idx]
        signal._ensure_signal()
        signal_data = signal.signal

        h = design_filter(M, K, window_type=window_type, filter_type=filter_type)
        filtered_signal = filter_signal(signal_data, h, compensate_delay=True)

        t_filtered = signal.t if (signal.t is not None and len(signal.t) == len(filtered_signal)) else (signal.t1 + np.arange(len(filtered_signal)) / signal.sampling)
        filtered_obj = sg.Signal.from_array(
            t_filtered,
            filtered_signal,
            t1=signal.t1,
            sampling=signal.sampling,
            discrete_signal=signal.discrete_signal
        )

        filter_names = {
            'bandpass': 'środkowoprzepustowy (F1)'
        }
        window_names = {
            'rectangular': 'Prostokątne',
            'hamming': 'Hamming (O1)'
        }
        
        filter_name = filter_names.get(filter_type, filter_type)
        window_name = window_names.get(window_type, window_type)
        
        filtered_obj.info_text = f" [Filtr {filter_name}: M={M}, K={K}, okno={window_name}]"

        list_of_signals.append(filtered_obj)
        update_signals_display()

        messagebox.showinfo("Sukces", f"Sygnał przefiltrowany dodany do listy!\n"
                          f"Typ filtru: {filter_name}\n"
                          f"Rząd filtru: M={M}\n"
                          f"Częstość odcięcia: fp/{K}\n"
                          f"Okno: {window_name}")

        show_filter_comparison(signal_data, filtered_signal, h)

    except ValueError as e:
        messagebox.showerror("Błąd", f"Błędne wartości: {str(e)}")
    except Exception as e:
        messagebox.showerror("Błąd", f"Błąd podczas filtracji: {str(e)}")

# ============================================================================
# KORELACJA I SENSOR RADAROWY
# ============================================================================

def compute_correlation():
    """Oblicz korelację dwóch wybranych sygnałów"""
    try:
        sig1_idx = corr_signal1_combo.current()
        sig2_idx = corr_signal2_combo.current()
        method = corr_method_var.get()

        if sig1_idx < 0 or sig2_idx < 0:
            messagebox.showerror("Błąd", "Wybierz oba sygnały do korelacji!")
            return

        sig1 = list_of_signals[sig1_idx]
        sig2 = list_of_signals[sig2_idx]
        
        sig1._ensure_signal()
        sig2._ensure_signal()

        result = cross_correlation(sig1.signal, sig2.signal, method=method)

        result_obj = sg.Signal.from_array(
            np.arange(len(result)),
            result,
            sampling=1.0
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
        sampling_rate = float(radar_sampling_entry.get())
        signal_speed = float(radar_speed_entry.get())
        delay_seconds = float(radar_delay_entry.get())

        if probe_idx < 0:
            messagebox.showerror("Błąd", "Wybierz sygnał sondujący!")
            return
        if delay_seconds < 0:
            messagebox.showerror("Błąd", "Opóźnienie nie może być ujemne!")
            return

        probe_sig = list_of_signals[probe_idx]
        probe_sig._ensure_signal()
        probe_data = probe_sig.signal
        
        delay_samples = int(round(delay_seconds * sampling_rate))

        reflected_data = np.zeros_like(probe_data)
        if delay_samples < len(probe_data):
            reflected_data[delay_samples:] = probe_data[:len(probe_data) - delay_samples]
        else:
            messagebox.showwarning("Ostrzeżenie", "Opóźnienie jest większe niż czas trwania sygnału! Wynik może być niejednoznaczny.")
            reflected_data = np.concatenate([np.zeros(delay_samples), probe_data])[:len(probe_data)]

        corr = cross_correlation(probe_data, reflected_data, method='direct')
        
        zero_index = len(probe_data) - 1
        
        positive_part = corr[zero_index:]
        detected_shift = np.argmax(np.abs(positive_part))
        
        detected_delay_time = detected_shift / sampling_rate
        distance = (signal_speed * detected_delay_time) / 2

        show_correlation_analysis(probe_data, reflected_data, corr, title="Korelacja sygnałów radarowych")

        result_text = f"""Wyniki pomiaru radarowego:

Zadane opóźnienie: {delay_seconds} s
Wykryte opóźnienie: {detected_delay_time:.6f} s (próbki: {detected_shift})
Odległość: {distance:.2f} j.u.

Parametry:
- Częstość próbkowania: {sampling_rate} Hz
- Prędkość sygnału: {signal_speed} j.u./s"""
        messagebox.showinfo("Wynik pomiaru", result_text)

    except ValueError as e:
        messagebox.showerror("Błąd", f"Błędna wartość: {str(e)}")
    except Exception as e:
        messagebox.showerror("Błąd", f"Błąd podczas pomiaru: {str(e)}")

# ============================================================================
# GENEROWANIE SYGNAŁÓW
# ============================================================================

def update_field_visibility():
    """Aktualizuj widoczność pól w zależności od typu sygnału"""
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
    """Wygeneruj nowy sygnał"""
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
    """Odśwież listę sygnałów"""
    for frame in signal_frames:
        frame.destroy()
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
    for combo in [filter_signal_combo, corr_signal1_combo, corr_signal2_combo, 
                  radar_probe_combo, radar_reflected_combo, splot_signal1_combo, splot_signal2_combo]:
        combo['values'] = signal_list

def show_signal_plot(idx):
    """Pokaż wykres sygnału"""
    if 0 <= idx < len(list_of_signals):
        sig = list_of_signals[idx]
        sig._ensure_signal()
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
    """Usuń sygnał"""
    if 0 <= idx < len(list_of_signals):
        list_of_signals.pop(idx)
        update_signals_display()

# ============================================================================
# GŁÓWNE GUI
# ============================================================================

def run():
    global root, notebook
    global signal_type_combo, signal_A_entry, signal_T_entry, signal_t1_entry
    global signal_d_entry, signal_kw_entry, signal_ts_entry, signal_ns_entry
    global signal_n1_entry, signal_p_entry, signal_sampling_entry, signals_frame
    global signal_A_label, signal_T_label, signal_t1_label, signal_d_label
    global signal_kw_label, signal_ts_label, signal_ns_label, signal_n1_label, signal_p_label
    global filter_signal_combo, filter_type_var, filter_order_entry, filter_cutoff_entry
    global filter_window_var
    global corr_signal1_combo, corr_signal2_combo, corr_method_var
    global radar_probe_combo, radar_reflected_combo, radar_sampling_entry, radar_speed_entry, radar_delay_entry
    global splot_signal1_combo, splot_signal2_combo

    root = tk.Tk()
    root.title("Cyfrowe Przetwarzanie Sygnałów - Zadanie 3")
    root.geometry("1200x800")

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # ========== TAB 1: GENEROWANIE SYGNAŁÓW ==========
    tab_generate = ttk.Frame(notebook)
    notebook.add(tab_generate, text="Generowanie sygnałów")

    params_frame = tk.LabelFrame(tab_generate, text="Parametry sygnału", padx=10, pady=10)
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
        ("Liczba próbek (ns):", 'ns'),
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
    signals_frame = tk.Frame(tab_generate)
    signals_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    tab_generate.rowconfigure(2, weight=1)
    tab_generate.columnconfigure(1, weight=1)

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

    # ========== TAB 2: FILTRACJA ==========
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
    tk.Label(tab_filter, text="Parametr cięcia (K - fp/K):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
    filter_cutoff_entry = tk.Entry(tab_filter, width=15)
    filter_cutoff_entry.insert(0, "8")
    filter_cutoff_entry.grid(row=4, column=1, sticky=tk.EW, padx=5, pady=5)
    tk.Button(tab_filter, text="Zastosuj filtr", command=design_and_apply_filter,
             bg='lightblue', height=2).grid(row=5, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=10)

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
    tk.Label(tab_radar, text="Sygnał sondujący:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    radar_probe_combo = ttk.Combobox(tab_radar, state='readonly', width=40)
    radar_probe_combo.grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
    tk.Label(tab_radar, text="Sygnał zwrotny:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    radar_reflected_combo = ttk.Combobox(tab_radar, state='readonly', width=40)
    radar_reflected_combo.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
    tk.Label(tab_radar, text="Częstość próbkowania [Hz]:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
    radar_sampling_entry = tk.Entry(tab_radar, width=15)
    radar_sampling_entry.insert(0, "1000")
    radar_sampling_entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
    tk.Label(tab_radar, text="Prędkość sygnału [j.u./s]:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
    radar_speed_entry = tk.Entry(tab_radar, width=15)
    radar_speed_entry.insert(0, "300000")
    radar_speed_entry.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
    tk.Label(tab_radar, text="Opóźnienie sygnału zwrotnego [s]:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
    radar_delay_entry = tk.Entry(tab_radar, width=15)
    radar_delay_entry.insert(0, "0.5")
    radar_delay_entry.grid(row=4, column=1, sticky=tk.EW, padx=5, pady=5)
    tk.Button(tab_radar, text="Pomiar odległości", command=simulate_radar,
             bg='lightcoral', height=2).grid(row=5, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=10)

    info_label = tk.Label(root, text="Warianty: O1 (Hamming) + F1 (Bandpass) | Kompensacja przesunięcia czasowego ✓ | Zapis/Wczytywanie ✓ | Kwantyzacja ✓ | Ekstrapolacja ✓",
                         bg='lightblue', pady=5, wraplength=1200)
    info_label.pack(fill=tk.X)

    root.mainloop()

if __name__ == '__main__':
    run()