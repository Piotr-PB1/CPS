import numpy as np
import struct
import os

MAGIC = b"SIGF"
VERSION = 1

class Signal:
    def __init__(self, A=0, t1=0.0, d=0.0, T=0.0, kw=0.0, ts=0.0, ns=0, n1=0, p=0.0, sampling=1000):
        self.A = A
        self.t1 = float(t1)
        self.d = float(d)
        self.T = float(T)
        self.kw = float(kw)
        self.ts = float(ts)
        self.ns = int(ns)
        self.n1 = int(n1)
        self.p = float(p)
        self.sampling = float(sampling)
        self.signal = None    
        self.t = None         
        self.discrete_signal = False
        self.quantization_signal = None

    def __str__(self):
        return f"{type(self).__name__}(A={self.A}, T={self.T}, t1={self.t1}, d={self.d}, kw={self.kw}, ts={self.ts}, ns={self.ns}, n1={self.n1}, p={self.p}, sampling={self.sampling})"

    def generate_signal(self):
        raise NotImplementedError()

    @classmethod
    def from_array(cls, t_values, signal_values, t1=0.0, sampling=1):
        obj = cls(0, t1, (t_values[-1] - t_values[0]) if len(t_values) > 1 else 0.0)
        obj.t = np.array(t_values)
        obj.signal = np.array(signal_values)
        obj.sampling = sampling
        obj.discrete_signal = True
        return obj

    def _ensure_signal(self):
        if self.signal is None:
            try:
                self.generate_signal()
            except Exception:
                pass

    def save_to_bin(self, filename):

        self._ensure_signal()
        if self.signal is None:
            raise ValueError("Brak próbek")

        data = np.asarray(self.signal, dtype=np.float64)
        n = data.size

        is_complex = np.iscomplexobj(data)
        is_discrete = self.discrete_signal

        type_id = int(type(self).__name__[1:]) if type(self).__name__.startswith("S") else 0

        flags = (1 if is_complex else 0) | (2 if is_discrete else 0)

        with open(filename, "wb") as f:

            f.write(MAGIC)
            f.write(struct.pack("B", VERSION))
            f.write(struct.pack("B", flags))

            f.write(struct.pack("B", type_id))

            f.write(struct.pack("d", self.A))
            f.write(struct.pack("d", self.T))
            f.write(struct.pack("d", self.t1))
            f.write(struct.pack("d", self.d))
            f.write(struct.pack("d", self.kw))
            f.write(struct.pack("d", self.ts))
            f.write(struct.pack("i", self.ns))
            f.write(struct.pack("i", self.n1))
            f.write(struct.pack("d", self.p))
            f.write(struct.pack("d", self.sampling))
            f.write(struct.pack("?", self.discrete_signal))

            f.write(struct.pack("I", n))

            if is_complex:
                # zapisz n jako liczba próbek complex
                real_imag = np.empty(data.size * 2, dtype=np.float64)
                real_imag[0::2] = data.real.ravel()
                real_imag[1::2] = data.imag.ravel()
                f.write(real_imag.tobytes())
            else:
                f.write(data.tobytes())

    def save_to_txt(self, filename):

        self._ensure_signal()

        with open(filename,"w") as f:

            f.write(f"{type(self).__name__}\n")

            f.write(f"A {self.A}\n")
            f.write(f"T {self.T}\n")
            f.write(f"t1 {self.t1}\n")
            f.write(f"d {self.d}\n")
            f.write(f"kw {self.kw}\n")
            f.write(f"ts {self.ts}\n")
            f.write(f"ns {self.ns}\n")
            f.write(f"n1 {self.n1}\n")
            f.write(f"p {self.p}\n")
            f.write(f"sampling {self.sampling}\n")
            f.write(f"Discrete {int(self.discrete_signal)}\n")

            f.write(f"N {len(self.signal)}\n\n")

            for t,v in zip(self.t,self.signal):
                f.write(f"{t} {v}\n")

    @classmethod
    def load_from_bin(cls, filename):

        with open(filename, "rb") as f:

            if f.read(4) != MAGIC:
                raise ValueError("Zły format")

            version = struct.unpack("B", f.read(1))[0]
            flags = struct.unpack("B", f.read(1))[0]

            is_complex = bool(flags & 1)

            type_id = struct.unpack("B", f.read(1))[0]

            A = struct.unpack("d", f.read(8))[0]
            T = struct.unpack("d", f.read(8))[0]
            t1 = struct.unpack("d", f.read(8))[0]
            d = struct.unpack("d", f.read(8))[0]
            kw = struct.unpack("d", f.read(8))[0]
            ts = struct.unpack("d", f.read(8))[0]
            ns = struct.unpack("i", f.read(4))[0]
            n1 = struct.unpack("i", f.read(4))[0]
            p = struct.unpack("d", f.read(8))[0]
            sampling = struct.unpack("d", f.read(8))[0]
            discrete_signal = struct.unpack("?", f.read(1))[0]

            n = struct.unpack("I", f.read(4))[0]

            if is_complex:
                raw = np.frombuffer(f.read(n * 16), dtype=np.float64)
                real = raw[0::2]
                imag = raw[1::2]
                data = real + 1j * imag
            else:
                data = np.frombuffer(f.read(n * 8), dtype=np.float64)

        module = __import__(__name__)
        class_name = f"S{type_id}"
        klass = getattr(module, class_name, cls)

        obj = object.__new__(klass)

        obj.A = A
        obj.T = T
        obj.t1 = t1
        obj.d = d
        obj.kw = kw
        obj.ts = ts
        obj.ns = ns
        obj.n1 = n1
        obj.p = p
        obj.sampling = sampling
        obj.discrete_signal = discrete_signal

        obj.signal = data
        obj.t = t1 + np.arange(len(data)) / sampling

        return obj

    @classmethod
    def load_from_txt(cls, filename):
        with open(filename) as f:
            type_name = f.readline().strip()
            params = {}
            while True:
                line = f.readline().strip()
                if line.startswith("N"):
                    n = int(line.split()[1])
                    break
                k, v = line.split()
                params[k] = float(v)
            t = []
            data = []

            for line in f:
                if line.strip() == "":
                    continue

                tv, v = line.split()
                t.append(float(tv))
                data.append(float(v))

        module = __import__(__name__)
        klass = getattr(module, type_name, cls)

        obj = object.__new__(klass)

        obj.A = params.get("A",0)
        obj.T = params.get("T",0)
        obj.t1 = params.get("t1",0)
        obj.d = params.get("d",0)
        obj.kw = params.get("kw",0)
        obj.ts = params.get("ts",0)
        obj.ns = int(params.get("ns",0))
        obj.n1 = int(params.get("n1",0))
        obj.p = params.get("p",0)
        obj.sampling = params.get("sampling",1)

        obj.t = np.array(t)
        obj.signal = np.array(data)

        obj.discrete_signal = bool(params.get("Discrete",0))

        return obj

    def mean_value(self):
        if self.signal is None:
            self.generate_signal()
        return np.mean(self.signal)
    
    def mean_abs_value(self):
        if self.signal is None:
            self.generate_signal()
        return np.mean(np.abs(self.signal))

    def variance(self):
        if self.signal is None:
            self.generate_signal()
        return np.mean((self.signal - np.mean(self.signal)) ** 2)
    
    def power(self):
        if self.signal is None:
            self.generate_signal()
        return np.mean(np.abs(self.signal) ** 2)
    
    def rms_value(self):
        if self.signal is None:
            self.generate_signal()
        return np.sqrt(self.power())

    def quantization(self, level):

        if level <= 1:
            raise ValueError("Liczba poziomów kwantyzacji musi być większa od 1")

        levels = 2**level
        min_val = np.min(self.signal)
        max_val = np.max(self.signal)
        
        if max_val == min_val:
            return
        
        step = (max_val - min_val) / levels
        
        scaled = (self.signal - min_val) / step
        
        indices = np.floor(scaled).astype(int)
        
        indices = np.clip(indices, 0, levels - 1)
        
        self.signal = min_val + (indices + 0.5) * step
            

    def extrapolation(self, type, oversample=10, sinc_samples=10):

        if type == "zero":
            Ts = 1.0 / self.sampling
            T_new = Ts / oversample
            
            t_new = np.arange(0, len(self.signal) * Ts - T_new/2, T_new)
            signal_out = np.zeros_like(t_new)
            
            for i, t in enumerate(t_new):
                idx = int(t / Ts)
                if idx >= len(self.signal):
                    idx = len(self.signal) - 1
                signal_out[i] = self.signal[idx]
            
            self.t = self.t1 + t_new
            self.signal = signal_out
            return signal_out

        elif type == "sinc":
            Ts = 1.0 / self.sampling
            T_new = Ts / oversample
            
            t_new = np.arange(0, len(self.signal) * Ts - T_new/2, T_new)
            signal_out = np.zeros_like(t_new)
            
            for i, t in enumerate(t_new):
                t_norm = t / Ts
                
                value = 0.0
                center = int(t_norm)
                start = max(0, center - sinc_samples // 2)
                end = min(len(self.signal), center + sinc_samples // 2 + 1)
                
                for n in range(start, end):
                    x = t_norm - n
                    if x == 0:
                        sinc_val = 1.0
                    else:
                        sinc_val = np.sin(np.pi * x) / (np.pi * x)
                    value += self.signal[n] * sinc_val
                
                signal_out[i] = value
            
            self.t = self.t1 + t_new
            self.signal = signal_out
            return signal_out
        else:
            raise ValueError("Nieznany typ ekstrapolacji")

    def mse(self, other):

        if len(self.signal) != len(other.signal):
            raise ValueError("Sygnały muszą mieć tę samą długość")
        
        return np.mean((self.signal - other.signal) ** 2)
    
    def snr_db(self, other):

        if len(self.signal) != len(other.signal):
            raise ValueError("Sygnały muszą mieć tę samą długość")
        
        signal_power = np.sum(other.signal ** 2)
        noise_power = np.sum((self.signal - other.signal) ** 2)
        
        if noise_power == 0:
            return np.inf
        
        return 10 * np.log10(signal_power / noise_power)
    
    def psnr_db(self, other):

        if len(self.signal) != len(other.signal):
            raise ValueError("Sygnały muszą mieć tę samą długość")
        
        mse_val = self.mse(other)
        
        if mse_val == 0:
            return np.inf
        
        max_val = np.max(np.abs(other.signal))
        
        return 10 * np.log10(max_val ** 2 / mse_val)
    
    def max_difference(self, other):

        if len(self.signal) != len(other.signal):
            raise ValueError("Sygnały muszą mieć tę samą długość")
        
        return np.max(np.abs(self.signal - other.signal))
    
    def enob(self, other):

        snr = self.snr_db(other)
        if snr == np.inf:
            return np.inf
        return (snr - 1.76) / 6.02

# ---------------- konkretne sygnały ----------------

class S1(Signal):  # szum jednostajny
    def __init__(self, A, t1, d, sampling=1000):
        super().__init__(A, t1, d, sampling=sampling)
        self.discrete_signal = False

    def generate_signal(self):
        n = max(1, int(round(self.d * self.sampling)))
        self.t = np.linspace(self.t1, self.t1 + self.d, n, endpoint=False)
        self.signal = np.random.uniform(-self.A, self.A, size=n)
        return self.signal

class S2(Signal):  # szum gaussowski
    def __init__(self, A, t1, d, sampling=1000):
        super().__init__(A, t1, d, sampling=sampling)
        self.discrete_signal = False

    def generate_signal(self):
        n = max(1, int(round(self.d * self.sampling)))
        self.t = np.linspace(self.t1, self.t1 + self.d, n, endpoint=False)
        self.signal = np.random.normal(0.0, self.A, size=n)
        return self.signal

class S3(Signal):  # sinusoidalny
    def __init__(self, A, T, t1, d, sampling=1000):
        super().__init__(A, t1, d, T=T, sampling=sampling)
        self.discrete_signal = False

    def generate_signal(self):
        n = max(1, int(round(self.d * self.sampling)))
        self.t = np.linspace(self.t1, self.t1 + self.d, n, endpoint=False)
        omega = 2.0 * np.pi / self.T if self.T != 0 else 0.0
        self.signal = self.A * np.sin(omega * (self.t - self.t1))
        return self.signal

class S4(Signal):  # sinus jednopołówkowo wyprostowany
    def __init__(self, A, T, t1, d, sampling=1000):
        super().__init__(A, t1, d, T=T, sampling=sampling)
        self.discrete_signal = False

    def generate_signal(self):
        n = max(1, int(round(self.d * self.sampling)))
        self.t = np.linspace(self.t1, self.t1 + self.d, n, endpoint=False)
        omega = 2.0 * np.pi / self.T if self.T != 0 else 0.0
        s = np.sin(omega * (self.t - self.t1))
        self.signal = 0.5 * self.A * (s + np.abs(s))
        return self.signal

class S5(Signal):  # abs(sin)
    def __init__(self, A, T, t1, d, sampling=1000):
        super().__init__(A, t1, d, T=T, sampling=sampling)
        self.discrete_signal = False

    def generate_signal(self):
        n = max(1, int(round(self.d * self.sampling)))
        self.t = np.linspace(self.t1, self.t1 + self.d, n, endpoint=False)
        omega = 2.0 * np.pi / self.T if self.T != 0 else 0.0
        self.signal = self.A * np.abs(np.sin(omega * (self.t - self.t1)))
        return self.signal

class S6(Signal):  # sygnał prostokątny (nie-symetryczny)
    def __init__(self, A, T, t1, d, kw, sampling=1000):
        super().__init__(A, t1, d, T=T, kw=kw, sampling=sampling)
        self.discrete_signal = False

    def generate_signal(self):
        n = max(1, int(round(self.d * self.sampling)))
        self.t = np.linspace(self.t1, self.t1 + self.d, n, endpoint=False)
        tau = (self.t - self.t1) % self.T
        self.signal = np.where(tau < (self.kw * self.T), self.A, 0.0)
        return self.signal

class S7(Signal):  # prostokątny symetryczny
    def __init__(self, A, T, t1, d, kw, sampling=1000):
        super().__init__(A, t1, d, T=T, kw=kw, sampling=sampling)
        self.discrete_signal = False

    def generate_signal(self):
        n = max(1, int(round(self.d * self.sampling)))
        self.t = np.linspace(self.t1, self.t1 + self.d, n, endpoint=False)
        self.signal = np.where(((self.t - self.t1) % self.T) < (self.kw * self.T), -self.A, self.A)
        return self.signal

class S8(Signal):  # trójkątny
    def __init__(self, A, T, t1, d, kw, sampling=1000):
        super().__init__(A, t1, d, T=T, kw=kw, sampling=sampling)
        self.discrete_signal = False

    def generate_signal(self):
        n = max(1, int(round(self.d * self.sampling)))
        self.t = np.linspace(self.t1, self.t1 + self.d, n, endpoint=False)
        tau = (self.t - self.t1) % self.T
        a = self.kw * self.T
        with np.errstate(divide='ignore', invalid='ignore'):
            up = (self.A / a) * tau
            down = (-self.A / (self.T - a)) * (tau - a) + self.A
            self.signal = np.where(tau < a, up, down)
        return self.signal

class S9(Signal):  # skok jednostkowy w czasie ts
    def __init__(self, A, t1, d, ts, sampling=1000):
        super().__init__(A, t1, d, ts=ts, sampling=sampling)
        self.discrete_signal = False

    def generate_signal(self):
        n = max(1, int(round(self.d * self.sampling)))
        self.t = np.linspace(self.t1, self.t1 + self.d, n, endpoint=False)
        self.signal = np.where(self.t < self.ts, 0.0, self.A)
        return self.signal

class S10(Signal): # impuls jednostkowy (dyskretny) 
    def __init__(self, A, n1, ns, sampling=1): 
        super().__init__(A, t1=0.0, d=0.0, sampling=sampling) 
        self.n1 = int(n1) 
        self.ns = int(ns) 
        self.discrete_signal = True 
        
    def generate_signal(self): 


        self.t = np.arange(-10, 11)
        idx = np.argmin(np.abs(self.t - self.ns))

        self.signal = np.zeros_like(self.t, dtype=float)
        self.signal[idx] = self.A

        return self.signal



        self.t = np.arange(-10+self.ns, 11 + self.ns)
        self.signal = np.where(self.t == self.ns, self.A, 0.0) 
        return self.signal
    
class S11(Signal):  # szum impulsowy (dyskretny) - losowy impuls z prawdopodobieństwem p
    def __init__(self, A, t1, d, p, sampling=1000):
        super().__init__(A, t1, d, p=p, sampling=sampling)
        self.discrete_signal = True

    def generate_signal(self):
        n = max(1, int(round(self.d * self.sampling)))
        self.t = self.t1 + np.arange(n) / self.sampling
        rnd = np.random.random(n)
        self.signal = np.where(rnd < self.p, self.A, 0.0)
        return self.signal