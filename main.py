import tkinter as tk
import Signal as sg
from tkinter import messagebox, filedialog
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
    (True, False, True, True, False, False),
    (True, False, True, True, False, False),
    (True, True,  True, True, False, False), 
    (True, True,  True, True, False, False), 
    (True, True,  True, True, False, False), 
    (True, True,  True, True, True, False), 
    (True, True,  True, True, True, False),
    (True, True,  True, True, True, False), 
    (True, False, True, True, False, True), 
    (True, False, True, False, False, False), 
    (True, True,  True, True, False, False), 
]

list_of_signals = []  # lista stworzonych sygnałów
signal_frames = []  # przechowywanie ramek sygnałów

def update_signals_display():
    """Odświeża wyświetlenie listy sygnałów"""
    global signal_frames
    
    # Wyczyść stare ramki
    for frame in signal_frames:
        frame.destroy()
    signal_frames = []
    
    # Utwórz nowe ramki dla każdego sygnału
    for idx, signal in enumerate(list_of_signals):
        frame = tk.Frame(signals_list_frame, relief=tk.RAISED, borderwidth=1, bg="#f0f0f0")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        signal_type = type(signal).__name__
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
        
        # Przycisk usuń
        btn_delete = tk.Button(frame, text="Usuń", width=8, bg="#ff6b6b", fg="white",
                              command=lambda i=idx: delete_signal(i))
        btn_delete.pack(side=tk.LEFT, padx=2, pady=5)
        
        signal_frames.append(frame)
    
    # Odśwież scroll region
    signals_list_frame.update_idletasks()
    signals_list_canvas.config(scrollregion=signals_list_canvas.bbox("all"))

def save_signal_bin(signal, idx):
    """Zapisz sygnał do pliku binarnego"""
    filename = filedialog.asksaveasfilename(
        defaultextension=".bin",
        filetypes=[("Binary files", "*.bin"), ("All files", "*.*")],
        initialfile=f"signal_{idx+1}.bin"
    )
    if filename:
        try:
            signal.save_to_bin(filename)
            messagebox.showinfo("Sukces", f"Sygnał zapisany do {filename}")
        except Exception as e:
            messagebox.showerror("Błąd", f"Błąd przy zapisie: {str(e)}")

def save_signal_txt(signal, idx):
    """Zapisz sygnał do pliku tekstowego"""
    filename = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        initialfile=f"signal_{idx+1}.txt"
    )
    if filename:
        try:
            signal.save_to_txt(filename)
            messagebox.showinfo("Sukces", f"Sygnał zapisany do {filename}")
        except Exception as e:
            messagebox.showerror("Błąd", f"Błąd przy zapisie: {str(e)}")

def show_signal_chart(signal, idx):
    """Pokaż wykres czasowy sygnału"""
    if not hasattr(signal, 'signal') or signal.signal is None:
        signal.generate_signal()

    plt.figure(figsize=(12, 6))
    plt.plot(signal.t, signal.signal, linewidth=1.5, color='blue')
    plt.title(f"Sygnał #{idx+1} - Wykres czasowy (A={signal.A}, d={signal.d}, fs={signal.sampling}Hz)", fontsize=12)
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def show_signal_histogram(signal, idx):
    """Pokaż histogram sygnału"""

    if not hasattr(signal, 'signal') or signal.signal is None:
        signal.generate_signal()
    
    plt.figure(figsize=(12, 6))
    plt.hist(signal.signal, bins=50, edgecolor='black', alpha=0.7, color='green')
    plt.title(f"Sygnał #{idx+1} - Histogram (A={signal.A}, d={signal.d}, fs={signal.sampling}Hz)", fontsize=12)
    plt.xlabel("Wartość")
    plt.ylabel("Liczba próbek")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def delete_signal(idx):
    """Usuń sygnał z listy"""
    if 0 <= idx < len(list_of_signals):
        list_of_signals.pop(idx)
        update_signals_display()

def new_signal(variant, A=0, T=0, t1=0, d=0, kw=0, ts=0, sampling=1000):
    print(f"new_signal({variant}, A={A}, T={T}, t1={t1}, d={d}, kw={kw}, ts={ts}, sampling={sampling})")
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
    
    update_signals_display()

def enter():
    if not lb.curselection():
        messagebox.showerror("Błąd", "Wybierz typ sygnału")
        return

    index = lb.curselection()[0]  # 0..10
    variant = index + 1

    try:
        A = float(entry_A.get() if entry_A.get() else 0)
        T = float(entry_T.get() if entry_T.get() else 0)
        t1 = float(entry_t1.get() if entry_t1.get() else 0)
        d = float(entry_d.get() if entry_d.get() else 0)
        kw = float(entry_kw.get() if entry_kw.get() else 0)
        ts = float(entry_ts.get() if entry_ts.get() else 0)
        sampling = int(entry_sampling.get() if entry_sampling.get() else 1000)
    except ValueError:
        messagebox.showerror("Błąd", "Wprowadź poprawne liczby w polach")
        return

    new_signal(variant, A, T, t1, d, kw, ts, sampling)

    info_label.config(text=f"Zatwierdzono: {SYGNALY[index]}, A={A}, T={T}, t1={t1}, d={d}, kw={kw}, ts={ts}, sampling={sampling}")

def update_fields(event=None):
    if not lb.curselection():
        return
    index = lb.curselection()[0]
    need_A, need_T, need_t1, need_d, need_kw, need_ts = PARAMS_NEEDED[index]

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

root = tk.Tk()
root.title("Signal manipulator")
root.geometry("1000x700")

# Panel z listą sygnałów (na górze)
label_signals = tk.Label(root, text="Stworzone sygnały:", font=("Arial", 10, "bold"))
label_signals.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="w")

# Canvas z scrollbarem dla listy sygnałów
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

lb = tk.Listbox(root, width=40, height=10)
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

# Próbkowanie
label_sampling = tk.Label(frame_params, text="Próbkowanie [Hz]:")
label_sampling.grid(row=6, column=0, padx=5, pady=5, sticky="e")
entry_sampling = tk.Entry(frame_params, width=15)
entry_sampling.grid(row=6, column=1, padx=5, pady=5)
entry_sampling.insert(0, "1000")

# Przycisk zatwierdź
button = tk.Button(frame_params, text="Zatwierdź sygnał", command=enter)
button.grid(row=7, column=0, columnspan=2, pady=15)

info_label = tk.Label(root, text="", fg="blue")
info_label.grid(row=3, column=0, columnspan=2, pady=10)

lb.bind('<<ListboxSelect>>', update_fields)

lb.selection_set(0)
update_fields()

root.mainloop()