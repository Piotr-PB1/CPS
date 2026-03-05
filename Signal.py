import numpy as np
import matplotlib.pyplot as plt
import struct
import os

class Signal:
    def __init__(self, A, t1, d, T=0, kw=0, ts=0):
        self.A = A
        self.t1 = t1
        self.d = d
        self.T = T
        self.kw = kw
        self.ts = ts
        self.signal = None
    
    def generate_signal(self, sampling=1000):
        pass

    def avg_value(self):
        if self.signal is None:
            self.generate_signal(sampling=1000)
        return 1/self.d * np.trapezoid(self.signal, dx=self.ts if self.ts else 1/1000)
    
    def avg_abs_value(self):
        if self.signal is None:
            self.generate_signal(sampling=1000)
        return 1/self.d * np.trapezoid(np.abs(self.signal), dx=self.ts if self.ts else 1/1000)
    
    def variance(self):
        if self.signal is None:
            self.generate_signal(sampling=1000)
        return 1/self.d * np.trapezoid((self.signal - self.avg_value())**2, dx=self.ts if self.ts else 1/1000)
    
    def power(self):
        if self.signal is None:
            self.generate_signal(sampling=1000)
        return np.mean(self.signal**2)
    
    def rms_value(self):
        if self.signal is None:
            self.generate_signal(sampling=1000)
        return np.sqrt(1/self.d * np.trapezoid(self.signal**2, dx=self.ts if self.ts else 1/1000))

    # def __init__(self, name, signal, time, time_0, omega, sampling):
    #     self.name = name
    #     self.signal = signal
    #     self.amplitude = (np.max(signal) - np.min(signal)) / 2
    #     self.avg_value = np.mean(signal)
    #     self.avg_abs_value = np.mean(np.abs(signal))
    #     self.rms_value = np.sqrt(np.mean(signal**2))
    #     self.variance = np.var(signal)
    #     self.power = np.mean(signal**2)

    #     self.time_0 = time_0
    #     self.time = time
    #     self.omega = omega
    #     self.frequency = omega / (2 * np.pi)
    #     self.period = 1 / self.frequency
    #     self.sampling = sampling

    # def __str__(self):
    #     return f"{self.name}: {self.value}"
    
    # def to_figute(self):
    #     pass

    # def to_histogram(self, bins):
    #     pass

    # def save_signal_to_bin_file(self, filename):
    #     with open(filename, 'wb') as file:
    #         file.write(struct.pack("<fi?", self.time_0, self.sampling, ))

    # def load_signal_from_bin_file(filename):
    #     signal_array = np.fromfile(filename, dtype=np.float32)
    #     return signal_array.tolist()

    # def save_signal_to_txt_file(signal, filename):
    #     with open(filename, 'w') as file:
    #         for value in signal:
    #             file.write(f"{value}\n")

    # def load_signal_from_txt_file(filename):
    #     with open(filename, 'r') as file:
    #         signal = [float(line.strip()) for line in file]
    #     return signal





    # parametry

    # metody
        # zapis, odczyt - to chyba też można zrobić w ogólnej funkcji wykorzystującej obiekt, histogram, wykres, 
        # obliczenie parametrów, wyliczenie sygnału?
        # wykersy można zroić w osobnej metodzie, gdzie parametrem będzie obiekt 
        # SPRWADZIĆ WYKŁĄDY - mogą tam być gotowe przykłady 


class S1(Signal):
    def __init__(self, A, t1, d, sampling=1000):
        self.A = A
        self.t1 = t1
        self.d = d
        self.sampling = sampling
        self.signal = None
    
    def generate_signal(self, sampling=None):
        if sampling is None:
            sampling = self.sampling
        num_samples = int(self.d * sampling)
        self.signal = np.random.uniform(-self.A, self.A, num_samples)
        return self.signal
    
    def save_to_bin(self, filename):
        """Zapisz sygnał do pliku binarnego"""
        if not hasattr(self, 'signal') or self.signal is None:
            self.generate_signal()
        with open(filename, 'wb') as f:
            self.signal.astype(np.float32).tofile(f)
    
    def save_to_txt(self, filename):
        """Zapisz sygnał do pliku tekstowego"""
        if not hasattr(self, 'signal') or self.signal is None:
            self.generate_signal()
        np.savetxt(filename, self.signal, fmt='%.6f')

class S2(Signal):
    def __init__(self, A, t1, d, sampling=1000):
        self.A = A
        self.t1 = t1
        self.d = d
        self.sampling = sampling
        self.signal = None
    
    def generate_signal(self, sampling=None):
        if sampling is None:
            sampling = self.sampling
        num_samples = int(self.d * sampling)
        self.signal = np.random.normal(0, self.A, num_samples)
        return self.signal
    
    def save_to_bin(self, filename):
        """Zapisz sygnał do pliku binarnego"""
        if not hasattr(self, 'signal') or self.signal is None:
            self.generate_signal()
        with open(filename, 'wb') as f:
            self.signal.astype(np.float32).tofile(f)
    
    def save_to_txt(self, filename):
        """Zapisz sygnał do pliku tekstowego"""
        if not hasattr(self, 'signal') or self.signal is None:
            self.generate_signal()
        np.savetxt(filename, self.signal, fmt='%.6f')

class S3(Signal):
    def __init__(self, A, T, t1, d, sampling=1000):
        self.A = A
        self.T = T
        self.t1 = t1
        self.d = d
        self.sampling = sampling
        self.signal = None
    
    def generate_signal(self, sampling=None):
        if sampling is None:
            sampling = self.sampling
        num_samples = int(self.d * sampling)
        self.signal = self.A * np.sin((2*np.pi/self.T) * np.arange(num_samples) / sampling)
        return self.signal

    def save_to_bin(self, filename):
        """Zapisz sygnał do pliku binarnego"""
        if not hasattr(self, 'signal') or self.signal is None:
            self.generate_signal()
        with open(filename, 'wb') as f:
            self.signal.astype(np.float32).tofile(f)
    
    def save_to_txt(self, filename):
        """Zapisz sygnał do pliku tekstowego"""
        if not hasattr(self, 'signal') or self.signal is None:
            self.generate_signal()
        np.savetxt(filename, self.signal, fmt='%.6f')