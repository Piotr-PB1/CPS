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
    
    def generate_signal(self):
        pass

    def avg_value(self):
        if self.signal is None:
            self.generate_signal()
        return 1/self.d * np.trapezoid(self.signal, dx=self.ts if self.ts else 1/1000)
    
    def avg_abs_value(self):
        if self.signal is None:
            self.generate_signal()
        return 1/self.d * np.trapezoid(np.abs(self.signal), dx=self.ts if self.ts else 1/1000)
    
    def variance(self):
        if self.signal is None:
            self.generate_signal()
        return 1/self.d * np.trapezoid((self.signal - self.avg_value())**2, dx=self.ts if self.ts else 1/1000)
    
    def power(self):
        if self.signal is None:
            self.generate_signal()
        return np.mean(self.signal**2)
    
    def rms_value(self):
        if self.signal is None:
            self.generate_signal()
        return np.sqrt(1/self.d * np.trapezoid(self.signal**2, dx=self.ts if self.ts else 1/1000))


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
        self.discrete_signal = False
    
    def generate_signal(self):
        self.t = np.linspace(self.t1, self.d+self.t1, int(self.d*self.sampling))
        self.signal = np.random.uniform(-self.A, self.A, self.t.shape)
        return self.signal

class S2(Signal):
    def __init__(self, A, t1, d, sampling=1000):
        self.A = A
        self.t1 = t1
        self.d = d
        self.sampling = sampling
        self.signal = None
        self.discrete_signal = False
    
    def generate_signal(self):
        self.t = np.linspace(self.t1, self.d+self.t1, int(self.d*self.sampling))
        self.signal = np.random.normal(0, self.A, self.t.shape)
        return self.signal

class S3(Signal):
    def __init__(self, A, T, t1, d, sampling=1000):
        self.A = A
        self.T = T
        self.t1 = t1
        self.d = d
        self.sampling = sampling
        self.signal = None
        self.discrete_signal = False
    
    def generate_signal(self):
        self.t = np.linspace(self.t1, self.d+self.t1, int(self.d*self.sampling))
        self.signal = self.A * np.sin((2*np.pi/self.T)*self.t)
        return self.signal

class S4(Signal):
    def __init__(self, A, T, t1, d, sampling=1000):
        self.A = A
        self.T = T
        self.t1 = t1
        self.d = d
        self.sampling = sampling
        self.signal = None
        self.discrete_signal = False
    
    def generate_signal(self):
        self.t = np.linspace(self.t1, self.d+self.t1, int(self.d*self.sampling))
        self.signal = 0.5*self.A*(np.sin((2*np.pi/self.T)*self.t) + np.abs(np.sin((2*np.pi/self.T)*self.t)))
        return self.signal

class S5(Signal):
    def __init__(self, A, T, t1, d, sampling=1000):
        self.A = A
        self.t1 = t1
        self.d = d
        self.T = T
        self.sampling = sampling
        self.signal = None
        self.discrete_signal = False

    def generate_signal(self):
        self.t = np.linspace(self.t1, self.d+self.t1, int(self.d*self.sampling))
        self.signal = self.A * np.abs(np.sin((2*np.pi/self.T)*self.t))
        return self.signal
    
class S6(Signal):
    def __init__(self, A, T, t1, d, kw, sampling=1000):
        self.A = A
        self.t1 = t1
        self.d = d
        self.T = T
        self.kw = kw
        self.sampling = sampling
        self.signal = None
        self.discrete_signal = False

    def generate_signal(self):
        self.t = np.linspace(self.t1, self.d+self.t1, int(self.d*self.sampling))
        self.signal = np.where((self.t % self.T) < (self.kw * self.T), self.A, 0)
        return self.signal
    
class S7(Signal):
    def __init__(self, A, T, t1, d, kw, sampling=1000):
        self.A = A
        self.t1 = t1
        self.d = d
        self.T = T
        self.kw = kw
        self.sampling = sampling
        self.signal = None
        self.discrete_signal = False

    def generate_signal(self):
        self.t = np.linspace(self.t1, self.d+self.t1, int(self.d*self.sampling))
        self.signal = np.where((self.t % self.T) < (self.kw * self.T), -self.A, self.A)
        return self.signal
    
class S8(Signal):
    def __init__(self, A, T, t1, d, kw, sampling=1000):
        self.A = A
        self.t1 = t1
        self.d = d
        self.T = T
        self.kw = kw
        self.sampling = sampling
        self.signal = None
        self.discrete_signal = False

    def generate_signal(self):
        self.t = np.linspace(self.t1, self.d+self.t1, int(self.d*self.sampling))
        # calculate time within each period
        tau = self.t % self.T
        # rising slope up to kw*T, then falling slope
        self.signal = np.where(
            tau < (self.kw * self.T),
            (self.A / (self.kw * self.T)) * tau,
            (-self.A / ((1 - self.kw) * self.T)) * (tau - self.kw * self.T) + self.A
        )
        return self.signal
    
class S9(Signal):
    def __init__(self, A, t1, d, ts, sampling=1000):
        self.A = A
        self.t1 = t1
        self.d = d
        self.ts = ts
        self.sampling = sampling
        self.signal = None
        self.discrete_signal = False

    def generate_signal(self):
        self.t = np.linspace(self.t1, self.d+self.t1, int(self.d*self.sampling))
        # create a signal that is A for ts seconds and 0 otherwise, repeating every ts seconds
        self.signal = np.where(self.t < self.ts , 0, self.A)
        return self.signal
    
class S10(Signal):
    def __init__(self, A, t1, d, ts, sampling=1000):
        self.A = A
        self.t1 = t1
        self.d = d
        self.ts = ts
        self.sampling = sampling
        self.signal = None
        self.discrete_signal = True

    def generate_signal(self):
        self.t = np.linspace(self.t1, self.d+self.t1, int(self.d*self.sampling))
        # create a signal that is A for ts seconds and 0 otherwise, repeating every ts seconds
        self.signal = np.where(self.t < self.ts , 0, -self.A)
        return self.signal