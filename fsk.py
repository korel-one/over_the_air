import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq

def str2bits(str):
    res = bin(int.from_bytes(str.encode('ascii'), 'big'))[2:]
    return '0'*(8 - (len(res) % 8)) + res

class SoundCommunication:

    def __init__(self, FS, symRate, msgSymLen, sync):
        self.symRate= symRate
        self.FS = FS
        #samples per second
        self.symsamp = np.int(FS/symRate) + np.int(FS/symRate)%2
        self.symlen = msgSymLen
        self.T = self.create_t(1/symRate, self.symsamp)
        shaping = np.empty(self.symsamp)
        shaping[:self.symsamp//2] = np.hstack((np.linspace(0, 1, self.symsamp//16), np.ones(self.symsamp//2 - self.symsamp//16)))
        shaping[self.symsamp//2:] = shaping[:self.symsamp//2][::-1]

        self.freqbot = 1000
        self.freqtop = 2000

        self.pulse = np.zeros((2, self.symsamp))
        self.pulse[0] = - shaping*np.sin(self.T * 2 * np.pi * 1355)
        self.pulse[1] =   shaping*np.sin(self.T * 2 * np.pi * 1645)

        self.sync = np.sin(self.create_t(10/symRate, 10*self.symsamp) * 2 * np.pi * 1504)
        # normalize both pulses to same power

    def create_t(self, length_seconds, samples):
        return np.linspace(0, length_seconds, samples)

    def send(self, binstream):
        synclen = self.sync.size
        S = np.zeros(self.symsamp*self.symlen + synclen)
        S[:synclen] = self.sync
        msg = binstream

        for i, bit in enumerate(msg):
            S[synclen + i*self.symsamp:synclen + (i+1)*self.symsamp]\
                        += self.pulse[int(bit, 2)]

        return S


    def _correlate(self, x, y):
        """periodic correlation"""
        if len(y) < len(x):
            yold = y
            y = np.zeros(x.shape)
            y[:yold.size] = yold
        return ifft(fft(x) * fft(y).conj()).real

    def decode(self, W):
        corr = self._correlate(W[:self.sync.size + self.symsamp*self.symlen], self.sync)

        start_sync = np.argmax(np.abs(corr)) # take the absolute value because the microphone
                                            # could inverse + and -
        if corr[start_sync] < 0:
            W = -W

        start_msg = start_sync + self.sync.size
        end_msg = start_msg + self.symlen*self.symsamp
        W = W[start_msg:end_msg]
        if W.size < end_msg - start_msg:
            #padd to avoid slow processing in fft
            W = np.hstack((W, np.zeros(end_msg - start_msg - W.size)))
        res = np.zeros(self.symlen, dtype=np.intp)
        frqcies = fftfreq(self.symsamp, 1/self.FS)
        for i in range(self.symlen):
            win = W[i*self.symsamp:(i+1)*self.symsamp]
            c0 = np.correlate(win, self.pulse[0])
            c1 = np.correlate(win, self.pulse[1])
            res[i] = 1 if c1 > c0 else 0

        return ''.join(map(str, res))
