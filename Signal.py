import numpy as np
import matplotlib.pyplot as plt
import struct

class Signal:
    def __init__(self, name, signal, time, time_0, omega, sampling):
        self.name = name
        self.signal = signal
        self.amplitude = (np.max(signal) - np.min(signal)) / 2
        self.avg_value = np.mean(signal)
        self.avg_abs_value = np.mean(np.abs(signal))
        self.rms_value = np.sqrt(np.mean(signal**2))
        self.variance = np.var(signal)
        self.power = np.mean(signal**2)

        self.time_0 = time_0
        self.time = time
        self.omega = omega
        self.frequency = omega / (2 * np.pi)
        self.period = 1 / self.frequency
        self.sampling = sampling

    def __str__(self):
        return f"{self.name}: {self.value}"
    
    def to_figute(self):
        pass

    def to_histogram(self, bins):
        pass

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