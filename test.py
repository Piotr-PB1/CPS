import numpy as np
import matplotlib.pyplot as plt
import Signal as sg

signal = sg.S5(1, 1, 1, 1)
signal.generate_signal()

plt.plot(signal.signal)
plt.show()