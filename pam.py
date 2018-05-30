import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq
from scipy.fftpack import rfft, irfft, rfftfreq

def str2bits(str):
    res = bin(int.from_bytes(str.encode('ascii'), 'big'))[2:]
    return '0'*(8 - (len(res) % 8)) + res

#TODO add coding
class SoundCommunication:

    def __init__(self, FS, symRate, freqbot, freqtop, msgSymLen = 180*8, sync = None):
        if sync: print("specified sync discarded")
        sync = '1100001111100011011101101101011001111100100100001000010100111000001000110011110011110110101110100110'
        self.symRate= symRate
        self.FS = FS
        #samples per second
        self.symsamp = np.int(FS/symRate) + np.int(FS/symRate)%2
        self.symlen = msgSymLen
        self.synclen = len(sync)
        self.T = self.create_t(1/symRate, self.symsamp)
        shaping = np.empty(self.symsamp)
        shaping[:self.symsamp//2] = np.hstack((np.linspace(0, 1, self.symsamp//8), np.ones(self.symsamp//2 - self.symsamp//8)))
        shaping[self.symsamp//2:] = shaping[:self.symsamp//2][::-1]

        self.freqbot = freqbot
        self.freqtop = freqtop

        self.pulse = np.zeros((2, self.symsamp))
        self.pulse[0] = - shaping*np.sin(self.T * 2 * np.pi * (freqtop + freqbot)/2)
        self.pulse[1] =   shaping*np.sin(self.T * 2 * np.pi * (freqtop + freqbot)/2)

        self.sync = sync
        # normalize both pulses to same power

    def create_t(self, length_seconds, samples):
        return np.linspace(0, length_seconds, samples)

    def send(self, binstream):
        total_len = self.synclen + self.symlen
        S = np.zeros(self.symsamp*total_len)
        msg = self.sync + binstream
        assert len(binstream) <= self.symlen
        binstream = binstream + '0'*(self.symlen - len(binstream))

        for i, bit in enumerate(msg):
            S[i*self.symsamp:(i+1)*self.symsamp]\
                        += self.pulse[int(bit, 2)]

        return S


    def _correlate(self, x, y):
        """periodic correlation"""
        if len(y) < len(x):
            yold = y
            y = np.zeros(x.shape)
            y[:yold.size] = yold
        return ifft(fft(x) * fft(y).conj()).real

    def bandpass_filter(self, W, low, high):
        fW = rfft(W)
        freqs = rfftfreq(W.size, 1/self.FS)
        mask = np.logical_not(np.logical_and(freqs > low, freqs < high))
        fW[mask] = 0.
        return irfft(fW)

    def decode(self, W):
        W = self.bandpass_filter(W, self.freqbot, self.freqtop)
        correlating_sync = self.send('')
        corr = self._correlate(W[:(self.symlen + self.synclen) * self.symsamp], correlating_sync)

        start_sync = np.argmax(np.abs(corr)) # take the absolute value because the microphone
                                            # could inverse + and -
        if corr[start_sync] < 0:
            W = -W

        res = np.zeros(self.symlen, dtype=np.intp)

        start_samp = start_sync + self.synclen*self.symsamp
        for i in range(self.symlen):
            jitter_max = 0#int(self.FS*(1/self.symRate)/8)
            #win = W[max(0, start_samp - jitter_max):min(self.symsamp*self.symlen, start_samp  + self.symsamp)]
            win = W[start_samp:start_samp  + self.symsamp]
            win_corr = np.correlate(win, self.pulse[1])
            max_idx = np.argmax(np.abs(win_corr))

            #if abs(max_idx - jitter_max) > 10:
            #    start_samp += self.symsamp + max_idx - jitter_max
            #else:
            #    start_samp += self.symsamp
            start_samp += self.symsamp

            res[i] = 1 if win_corr[max_idx] > 0 else 0

        return ''.join(map(str, res))
