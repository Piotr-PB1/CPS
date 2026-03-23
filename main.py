import tkinter as tk
import Signal as sg
from tkinter import messagebox, filedialog, ttk
import matplotlib.pyplot as plt
import numpy as np

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
    # A, T, t1, d, kw, ts, ns, n1, p
    (True, False, True, True, False, False, False, False, False),
    (True, False, True, True, False, False, False, False, False),
    (True, True,  True, True, False, False, False, False, False), 
    (True, True,  True, True, False, False, False, False, False), 
    (True, True,  True, True, False, False, False, False, False), 
    (True, True,  True, True, True, False, False, False, False), 
    (True, True,  True, True, True, False, False, False, False),
    (True, True,  True, True, True, False, False, False, False), 
    (True, False, True, True, False, True, False, False, False), 
    (True, False, False, False, False, False, True, True, False), 
    (True, False, True, True, False, False, False, False, True), 
]

list_of_signals = []  # lista stworzonych sygnałów
signal_frames = []  # przechowywanie ramek sygnałów

def update_signals_display():
    global signal_frames
    
    for frame in signal_frames:
        frame.destroy()
    signal_frames = []
    
    # Utwórz nowe ramki dla każdego sygnału
    for idx, signal in enumerate(list_of_signals):
        frame = tk.Frame(signals_list_frame, relief=tk.RAISED, borderwidth=1, bg="#f0f0f0")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        signal_type = type(signal).__name__
        if signal_type.startswith("S11"):
            label_text = f"{idx + 1}. {signal_type} - A={signal.A}, t1={signal.t1}, d={signal.d}, fs={signal.sampling}Hz"
        elif signal.discrete_signal:
            label_text = f"{idx + 1}. {signal_type} - A={signal.A}, n1={signal.n1}, ns={signal.ns}, fs={signal.sampling}Hz"
        else:
            label_text = f"{idx + 1}. {signal_type} - A={signal.A}, t1={signal.t1}, d={signal.d}, fs={signal.sampling}Hz"
        
        label = tk.Label(frame, text=label_text, bg="#f0f0f0", justify=tk.LEFT)
        label.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.X, expand=True)
        
        # Przycisk zapisz do BIN
        btn_bin = tk.Button(frame, text="Zapisz BIN", width=12,
                           command=lambda s=signal, i=idx: save_signal_bin(s, i))
        btn_bin.pack(side=tk.LEFT, padx=2, pady=5)
        
        # Przycisk zapisz do TXT
        btn_txt = tk.Button(frame, text="Zapisz TXT", width=12,
                           command=lambda s=signal, i=idx: save_signal_txt(s, i))
        btn_txt.pack(side=tk.LEFT, padx=2, pady=5)
        
        # Przycisk pokaż wykres czasowy
        btn_chart = tk.Button(frame, text="Wykres", width=10,
                             command=lambda s=signal, i=idx: show_signal_chart(s, i))
        btn_chart.pack(side=tk.LEFT, padx=2, pady=5)
        
        # Przycisk pokaż histogram
        btn_hist = tk.Button(frame, text="Histogram", width=10,
                            command=lambda s=signal, i=idx: show_signal_histogram(s, i))
        btn_hist.pack(side=tk.LEFT, padx=2, pady=5)
                
        #prametry sygnału
        btn_params = tk.Button(frame, text="Parametry", width=10, 
                               command=lambda s=signal: signal_params_to_labels(s))
        btn_params.pack(side=tk.LEFT, padx=2, pady=5)
        
        # Przycisk usuń
        btn_delete = tk.Button(frame, text="Usuń", width=8, bg="#ff6b6b", fg="white",
                              command=lambda i=idx: delete_signal(i))
        btn_delete.pack(side=tk.LEFT, padx=2, pady=5)

        combo_1_list_signal_1_math['values'] = [
            f"{idx+1}. {type(s).__name__}" for idx, s in enumerate(list_of_signals)
        ]

        combo_2_list_signal_2_math['values'] = [
            f"{idx+1}. {type(s).__name__}" for idx, s in enumerate(list_of_signals)
        ]
        
        signal_frames.append(frame)
    
    # Odśwież scroll region
    signals_list_frame.update_idletasks()
    signals_list_canvas.config(scrollregion=signals_list_canvas.bbox("all"))

def load_file():

    filename = filedialog.askopenfilename(
        title="Wybierz plik sygnału",
        filetypes=[("Binary files", "*.bin"), ("Text files", "*.txt")]
    )

    if not filename:
        return

    try:
        ext = filename.lower().split('.')[-1]

        if ext == "bin":
            signal = sg.Signal.load_from_bin(filename)

        elif ext == "txt":
            signal = sg.Signal.load_from_txt(filename)

        else:
            messagebox.showerror("Błąd", "Nieobsługiwany format pliku")
            return

        list_of_signals.append(signal)
        update_signals_display()
    except Exception as e:
        messagebox.showerror("Błąd", f"Nie udało się wczytać pliku: {e}")


def save_signal_bin(signal, idx):
    filename = filedialog.asksaveasfilename(
        defaultextension=".bin",
        filetypes=[("Binary files", "*.bin")],
        initialfile=f"signal_{idx+1}.bin"
    )
    if filename:
        try:
            signal.save_to_bin(filename)
            messagebox.showinfo("Sukces", f"Sygnał zapisany do {filename}")
        except Exception as e:
            messagebox.showerror("Błąd", f"Błąd przy zapisie: {str(e)}")

def save_signal_txt(signal, idx):
    filename = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt")],
        initialfile=f"signal_{idx+1}.txt"
    )
    if filename:
        try:
            signal.save_to_txt(filename)
            messagebox.showinfo("Sukces", f"Sygnał zapisany do {filename}")
        except Exception as e:
            messagebox.showerror("Błąd", f"Błąd przy zapisie: {str(e)}")

def show_signal_chart(signal, idx):
    if not hasattr(signal, 'signal') or signal.signal is None:
        signal.generate_signal()
    plt.figure(figsize=(10, 5))
    plt.style.use('seaborn-v0_8-whitegrid')
    if signal.discrete_signal:
        plt.stem(signal.t, signal.signal, basefmt=" ", markerfmt='o', linefmt='C0-')
    else:
        plt.plot(signal.t, signal.signal, color='C0', linewidth=1.8, label="Sygnał")
    plt.title(f"Sygnał #{idx+1} ({type(signal).__name__})", fontsize=13, weight='bold')
    plt.xlabel("Czas [s]" if not signal.discrete_signal else "Numer próbki")
    plt.ylabel("Amplituda")
    plt.legend()
    plt.tight_layout()
    plt.show()

def show_signal_histogram(signal, _):

    if not hasattr(signal, 'signal') or signal.signal is None:
        signal.generate_signal()
    
    plt.figure(figsize=(12, 6))
    plt.hist(signal.signal, bins=int(combo.get()), edgecolor='black', alpha=0.7, color='green')

    plt.title(signal.__str__(), fontsize=12)
    plt.xlabel("Wartość")
    plt.ylabel("Liczba próbek")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def delete_signal(idx):
    if 0 <= idx < len(list_of_signals):
        list_of_signals.pop(idx)
        update_signals_display()

def new_signal(variant, A=0, T=0, t1=0, d=0, kw=0, ts=0, ns=0, n1=-1, p=0.5, sampling=1000):
    print(f"new_signal({variant}, A={A}, T={T}, t1={t1}, d={d}, kw={kw}, ts={ts}, ns={ns}, n1={n1}, p={p} sampling={sampling})")
    if variant == 1:
        signal = sg.S1(A, t1, d, sampling)
        list_of_signals.append(signal)
    elif variant == 2:
        signal = sg.S2(A, t1, d, sampling)
        list_of_signals.append(signal)
    elif variant == 3:
        signal = sg.S3(A, T, t1, d, sampling)
        list_of_signals.append(signal)
    elif variant == 4:
        signal = sg.S4(A, T, t1, d, sampling)
        list_of_signals.append(signal)
    elif variant == 5:
        signal = sg.S5(A, T, t1, d, sampling)
        list_of_signals.append(signal)
    elif variant == 6:
        signal = sg.S6(A, T, t1, d, kw, sampling)
        list_of_signals.append(signal)
    elif variant == 7:
        signal = sg.S7(A, T, t1, d, kw, sampling)
        list_of_signals.append(signal)
    elif variant == 8:
        signal = sg.S8(A, T, t1, d, kw, sampling)
        list_of_signals.append(signal)
    elif variant == 9:
        signal = sg.S9(A, t1, d, ts, sampling)
        list_of_signals.append(signal)
    elif variant == 10:
        signal = sg.S10(A, n1, ns, sampling)
        list_of_signals.append(signal)
    elif variant == 11:
        signal = sg.S11(A, t1, d, p, sampling)
        list_of_signals.append(signal)
    
    update_signals_display()

def enter():
    if not lb.curselection():
        messagebox.showerror("Błąd", "Wybierz typ sygnału")
        return

    index = lb.curselection()[0]
    variant = index + 1

    try:
        A = float(entry_A.get() if entry_A.get() else 0)
        T = float(entry_T.get() if entry_T.get() else 0)
        t1 = float(entry_t1.get() if entry_t1.get() else 0)
        d = float(entry_d.get() if entry_d.get() else 0)
        kw = float(entry_kw.get() if entry_kw.get() else 0)
        ts = float(entry_ts.get() if entry_ts.get() else 0)
        ns = float(entry_ns.get() if entry_ns.get() else 0)
        n1 = float(entry_n1.get() if entry_n1.get() else 1)
        p = float(entry_p.get() if entry_p.get() else 0.5)
        sampling = int(entry_sampling.get() if entry_sampling.get() else 1000)
    except ValueError:
        messagebox.showerror("Błąd", "Wprowadź poprawne liczby w polach")
        return

    new_signal(variant, A, T, t1, d, kw, ts, ns, n1, p, sampling)

    info_label.config(text=f"Zatwierdzono: {SYGNALY[index]}, A={A}, T={T}, t1={t1}, d={d}, kw={kw}, ts={ts}, n1={n1}, ns={ns}, p={p} sampling={sampling}")

def update_fields(event=None):
    if not lb.curselection():
        return
    index = lb.curselection()[0]
    _, need_T, need_t1, need_d, need_kw, need_ts, need_ns, need_n1, need_p = PARAMS_NEEDED[index]

    if need_T:
        entry_T.config(state='normal')
    else:
        entry_T.config(state='disabled')
        entry_T.delete(0, tk.END)

    if need_t1:
        entry_t1.config(state='normal')
    else:
        entry_t1.config(state='disabled')
        entry_t1.delete(0, tk.END)

    if need_d:
        entry_d.config(state='normal')
    else:
        entry_d.config(state='disabled')
        entry_d.delete(0, tk.END)

    if need_kw:
        entry_kw.config(state='normal')
    else:
        entry_kw.config(state='disabled')
        entry_kw.delete(0, tk.END)

    if need_ts:
        entry_ts.config(state='normal')
    else:
        entry_ts.config(state='disabled')
        entry_ts.delete(0, tk.END)

    if need_ns:
        entry_ns.config(state='normal')
    else:
        entry_ns.config(state='disabled')
        entry_ns.delete(0, tk.END)

    if need_n1:
        entry_n1.config(state='normal')
    else:
        entry_n1.config(state='disabled')
        entry_n1.delete(0, tk.END)

    if need_p:
        entry_p.config(state='normal')
    else:
        entry_p.config(state='disabled')
        entry_p.delete(0, tk.END)

root = tk.Tk()
root.title("Signal manipulator")
root.geometry("1200x700")

label_signals = tk.Label(root, text="Stworzone sygnały:", font=("Arial", 10, "bold"))
label_signals.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="w")

signals_list_canvas = tk.Canvas(root, height=200, width=800, bg="white")
scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=signals_list_canvas.yview)
signals_list_frame = tk.Frame(signals_list_canvas, bg="white")

signals_list_frame.bind(
    "<Configure>",
    lambda e: signals_list_canvas.configure(scrollregion=signals_list_canvas.bbox("all"))
)

signals_list_canvas.create_window((0, 0), window=signals_list_frame, anchor="nw")
signals_list_canvas.configure(yscrollcommand=scrollbar.set)

signals_list_canvas.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
scrollbar.grid(row=1, column=2, sticky="ns", pady=5)

lb = tk.Listbox(root, width=40, height=11)
for i, s in enumerate(SYGNALY, start=1):
    lb.insert(i, s)
lb.grid(row=2, column=0, padx=10, pady=10, sticky="n")

frame_params = tk.Frame(root)
frame_params.grid(row=2, column=1, padx=10, pady=10, sticky="n")

# Amplituda
label_A = tk.Label(frame_params, text="Amplituda A:")
label_A.grid(row=0, column=0, padx=5, pady=5, sticky="e")
entry_A = tk.Entry(frame_params, width=15)
entry_A.grid(row=0, column=1, padx=5, pady=5)
entry_A.insert(0, "0")

# Okres
label_T = tk.Label(frame_params, text="Okres T:")
label_T.grid(row=1, column=0, padx=5, pady=5, sticky="e")
entry_T = tk.Entry(frame_params, width=15)
entry_T.grid(row=1, column=1, padx=5, pady=5)
entry_T.insert(0, "0")

# czas początkowy t1
label_t1 = tk.Label(frame_params, text="Czas początkowy t1:")
label_t1.grid(row=2, column=0, padx=5, pady=5, sticky="e")
entry_t1 = tk.Entry(frame_params, width=15)
entry_t1.grid(row=2, column=1, padx=5, pady=5)
entry_t1.insert(0, "0")

# długość sygnału d
label_d = tk.Label(frame_params, text="Długośc sygnału d:")
label_d.grid(row=3, column=0, padx=5, pady=5, sticky="e")
entry_d = tk.Entry(frame_params, width=15)
entry_d.grid(row=3, column=1, padx=5, pady=5)
entry_d.insert(0, "0")

# współczynnik wypełnienia kw
label_kw = tk.Label(frame_params, text="Współczynnik wypełnienia kw:")
label_kw.grid(row=4, column=0, padx=5, pady=5, sticky="e")
entry_kw = tk.Entry(frame_params, width=15)
entry_kw.grid(row=4, column=1, padx=5, pady=5)
entry_kw.insert(0, "0")

# czas skoku ts
label_ts = tk.Label(frame_params, text="Czas skoku ts:")
label_ts.grid(row=5, column=0, padx=5, pady=5, sticky="e")
entry_ts = tk.Entry(frame_params, width=15)
entry_ts.grid(row=5, column=1, padx=5, pady=5)
entry_ts.insert(0, "0")

# nr próbki n1
label_n1 = tk.Label(frame_params, text="Nr próbki n1:")
label_n1.grid(row=6, column=0, padx=5, pady=5, sticky="e")
entry_n1 = tk.Entry(frame_params, width=15)
entry_n1.grid(row=6, column=1, padx=5, pady=5)
entry_n1.insert(0, "-1")

# nr próki dla której następuje skok 

label_ns = tk.Label(frame_params, text="Nr próbki skoku ns:")
label_ns.grid(row=7, column=0, padx=5, pady=5, sticky="e")
entry_ns = tk.Entry(frame_params, width=15)
entry_ns.grid(row=7, column=1, padx=5, pady=5)
entry_ns.insert(0, "0")

# prawdopodobieństwo p dla szumu impulsowego
label_p = tk.Label(frame_params, text="Prawdopodobieństwo p:")
label_p.grid(row=8, column=0, padx=5, pady=5, sticky="e")
entry_p = tk.Entry(frame_params, width=15)
entry_p.grid(row=8, column=1, padx=5, pady=5)
entry_p.insert(0, "0")

# próbkowanie
label_sampling = tk.Label(frame_params, text="Próbkowanie [Hz]:")
label_sampling.grid(row=9, column=0, padx=5, pady=5, sticky="e")
entry_sampling = tk.Entry(frame_params, width=15)
entry_sampling.grid(row=9, column=1, padx=5, pady=5)
entry_sampling.insert(0, "1000")

# Przycisk zatwierdź
button_enter = tk.Button(frame_params, text="Zatwierdź sygnał", command=enter)
button_enter.grid(row=10, column=0, columnspan=2, pady=15)

# Przyciks wczytaj z pliku
button_upload = tk.Button(frame_params, text="Wczytaj z pliku", command=load_file)
button_upload.grid(row=11, column=0, columnspan=2, pady=5)

label_combo = tk.Label(frame_params, text="Wybierz ilość przedziałow w hisotgramie:")
label_combo.grid(row=0, column=2, padx=5, pady=5, sticky="e")

info_label = tk.Label(root, text="", fg="blue")
info_label.grid(row=3, column=0, columnspan=2, pady=10)

combo = ttk.Combobox(frame_params, values=[5, 10, 15, 20])
combo.grid(row=1, column=2, columnspan=2, pady=5)
combo.current(1)

combo_1_label = tk.Label(frame_params, text="Sygnał 1 do operacji matematycznych:")
combo_1_label.grid(row=3, column=2, padx=5, pady=5, sticky="e")
combo_1_list_signal_1_math = ttk.Combobox(frame_params, values=[f"{idx+1}. {type(s).__name__}" for idx, s in enumerate(list_of_signals)], state="readonly")
combo_1_list_signal_1_math.grid(row=4, column=2, columnspan=2, pady=5)

combo_2_label = tk.Label(frame_params, text="Sygnał 2 do operacji matematycznych:")
combo_2_label.grid(row=5, column=2, padx=5, pady=5)
combo_2_list_signal_2_math = ttk.Combobox(frame_params, values=[f"{idx+1}. {type(s).__name__}" for idx, s in enumerate(list_of_signals)], state="readonly")
combo_2_list_signal_2_math.grid(row=6, column=2, columnspan=2, pady=5)

combo_3_label = tk.Label(frame_params, text="Wybierz operację matematyczną:")
combo_3_label.grid(row=7, column=2, padx=5, pady=5)
combo_3_list_operation = ttk.Combobox(frame_params, values=["dodawanie", "odejmowanie", "mnożenie", "dzielenie"], state="readonly")
combo_3_list_operation.grid(row=8, column=2, columnspan=2, pady=5)
combo_3_list_operation.current(0)

def chosen_signal_to_do_math(event=None):
    index1 = combo_1_list_signal_1_math.current()
    index2 = combo_2_list_signal_2_math.current()
    return list_of_signals[index1], list_of_signals[index2]

combo_1_list_signal_1_math.bind("<<ComboboxSelected>>", chosen_signal_to_do_math)
combo_2_list_signal_2_math.bind("<<ComboboxSelected>>", chosen_signal_to_do_math)

do_math_button = tk.Button(frame_params, text="Wykonaj operację", command=lambda: operate_signals(chosen_signal_to_do_math(), combo_3_list_operation.get()))
do_math_button.grid(row=9, column=2, columnspan=2, pady=10)

def operate_signals(signals, op):
    if signals[0] is None or signals[1] is None:
        messagebox.showerror("Błąd", "Potrzebne są co najmniej dwa sygnały")
        return
    if signals[0].sampling != signals[1].sampling:
        messagebox.showerror("Błąd", "Sygnały muszą mieć takie samo próbkowanie")
        return
    s1 = signals[0]
    s2 = signals[1]

    # dopasuj długość

    t_start = min(s1.t[0], s2.t[0])
    t_end   = max(s1.t[-1], s2.t[-1])
    t_common = np.arange(t_start, t_end, 1/s1.sampling)

    data1 = []
    for t in t_common:
        idx = np.argmin(np.abs(s1.t - t))
        if np.abs(s1.t[idx] - t) < (1 / s1.sampling):
            data1.append(s1.signal[idx])
        else:
            data1.append(0.0)

    data2 = []
    for t in t_common:
        idx = np.argmin(np.abs(s2.t - t))
        if np.abs(s2.t[idx] - t) < (1 / s2.sampling):
            data2.append(s2.signal[idx])
        else:
            data2.append(0.0)

    data1 = np.array(data1)
    data2 = np.array(data2)
    
    if op == 'dodawanie':
        result_data = data1 + data2
    elif op == 'odejmowanie':
        result_data = data1 - data2
    elif op == 'mnożenie':
        result_data = data1 * data2
    else:
        result_data = np.divide(data1, data2, out=np.zeros_like(data1), where=data2!=0)

    new_sig = sg.Signal.from_array(
        t_common,
        result_data,
        t1=t_common[0],
        sampling=s1.sampling
    )

    new_sig.t = t_common
    new_sig.discrete_signal = s1.discrete_signal or s2.discrete_signal

    list_of_signals.append(new_sig)
    update_signals_display()

def signal_params_to_labels(signal):

    avereage_value_label = tk.Label(frame_params, text=f"Średnia wartość sygnału: {signal.mean_value():.2f}")
    avereage_value_label.grid(row=0, column=5, padx=5, pady=5)

    average_abs_value_label = tk.Label(frame_params, text=f"Średnia wartość bezwzględna sygnału: {signal.mean_abs_value():.2f}")
    average_abs_value_label.grid(row=1, column=5, padx=5, pady=5)

    variance_label = tk.Label(frame_params, text=f"Wariancja sygnału: {signal.variance():.2f}")
    variance_label.grid(row=2, column=5, padx=5, pady=5)

    RMS_value_label = tk.Label(frame_params, text=f"Wartość skuteczna (RMS) sygnału: {signal.rms_value():.2f}")
    RMS_value_label.grid(row=3, column=5, padx=5, pady=5)

    power_label = tk.Label(frame_params, text=f"Moc sygnału: {signal.power():.2f}")
    power_label.grid(row=4, column=5, padx=5, pady=5)

lb.bind('<<ListboxSelect>>', update_fields)

lb.selection_set(0)
update_fields()

root.mainloop()