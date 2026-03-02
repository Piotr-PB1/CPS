import tkinter as tk
from tkinter import messagebox

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
    (True, False, False, True, False, False),
    (True, False, False, True, False, False),
    (True, True,  True, True, False, False), 
    (True, True,  True, True, False, False), 
    (True, True,  True, True, False, False), 
    (True, True,  True, True, True, False), 
    (True, True,  False, True, True, False),
    (True, True,  True, True, True, False), 
    (True, False, True, True, False, True), 
    (True, False, True, False, False, False), 
    (True, True,  True, True, False, False), 
]

def new_signal(variant, A, T, t1, d, kw, ts):
    print(f"new_signal({variant}, A={A}, T={T}, t1={t1}, d={d}, kw={kw}, ts={ts})")
    #tworzenie nowego sygnału na podstawie wariantu i parametrów

def zatwierdz():
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
    except ValueError:
        messagebox.showerror("Błąd", "Wprowadź poprawne liczby w polach")
        return

    new_signal(variant, A, T, t1)

    info_label.config(text=f"Zatwierdzono: {SYGNALY[index]}, A={A}, T={T}, t1={t1}, d={d}, kw={kw}, ts={ts}")

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
root.geometry("700x450")

lb = tk.Listbox(root, width=40, height=15)
for i, s in enumerate(SYGNALY, start=1):
    lb.insert(i, s)
lb.grid(row=0, column=0, padx=10, pady=10, rowspan=4, sticky="n")

frame_params = tk.Frame(root)
frame_params.grid(row=0, column=1, padx=10, pady=10, sticky="n")

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

# Przycisk zatwierdź
button = tk.Button(frame_params, text="Zatwierdź", command=zatwierdz)
button.grid(row=6, column=0, columnspan=2, pady=15)

info_label = tk.Label(root, text="", fg="blue")
info_label.grid(row=4, column=0, columnspan=2, pady=10)

lb.bind('<<ListboxSelect>>', update_fields)

lb.selection_set(0)
update_fields()

root.mainloop()