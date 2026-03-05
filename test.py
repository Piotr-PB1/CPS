import numpy as np
import matplotlib.pyplot as plt
import Signal as sg

signal = sg.S1(1, 1, 1)
signal.generate_signal()

plt.plot(signal.signal)
plt.show()