import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq

def str2bits(str):
    res = bin(int.from_bytes(str.encode('ascii'), 'big'))[2:]
    return '0'*(8 - (len(res) % 8)) + res

class SoundCommunication2:

    def __init__(self, FS, symRate, msgSymLen, sync):
        assert len(sync) % 4 == 0
        self.symRate= symRate
        self.FS = FS
        #samples per second
        self.symsamp = np.int(FS/symRate) + np.int(FS/symRate)%2
        self.symlen = msgSymLen
        self.synclen = len(sync)//4
        self.T = self.create_t(1/symRate, self.symsamp)
        shaping = np.empty(self.symsamp)
        shaping[:self.symsamp//2] = np.linspace(0, 1, self.symsamp//2)
        shaping[self.symsamp//2:] = shaping[:self.symsamp//2][::-1]
        self.tunenum = 16
        self.freqbot = 1000
        self.freqtop = 2000
        #TODO set frequencies exactly
        self.pulsefreq = np.linspace(self.freqbot, self.freqtop, self.tunenum + 2)[1:-1] #discard frequencies
        self.fr_spacing = self.pulsefreq[1] - self.pulsefreq[0]
        #too close to edges
        self.modpulse = dict()
        for i, freq in enumerate(self.pulsefreq):
            self.modpulse[i] = np.sin(self.T * 2 * np.pi * freq) * shaping
        self.sync = sync
        # normalize both pulses to same power

    def create_t(self, length_seconds, samples):
        return np.linspace(0, length_seconds, samples)

    def chunks(self, msg):
        chunk =''
        for b in msg:
            chunk += b
            if len(chunk) == 4:
                yield chunk
                chunk = ''

    def send(self, binstream):
        total_len = self.synclen + self.symlen
        S = np.zeros(self.symsamp*total_len)
        msg = self.sync + binstream

        chunk = ''
        for i, chunk in enumerate(self.chunks(msg)):
            S[i*self.symsamp:(i+1)*self.symsamp]\
                        += self.modpulse[int(chunk, 2)]

        return S


    def _correlate(self, x, y):
        """periodic correlation"""
        if len(y) < len(x):
            yold = y
            y = np.zeros(x.shape)
            y[:yold.size] = yold
        return ifft(fft(x) * fft(y).conj()).real

    def decode(self, W):
        correlating_sync = self.send('')
        corr = self._correlate(W[:(self.symlen + self.synclen) * self.symsamp], correlating_sync)

        start_sync = np.argmax(np.abs(corr)) # take the absolute value because the microphone
                                            # could inverse + and -

        start_msg = start_sync + self.synclen*self.symsamp
        end_msg = start_msg + self.symlen*self.symsamp
        W = W[start_msg:end_msg]
        if W.size < end_msg - start_msg:
            #padd to avoid slow processing in fft
            W = np.hstack((W, np.zeros(end_msg - start_msg - W.size)))
        res = np.zeros(self.symlen, dtype=np.intp)
        frqcies = fftfreq(self.symsamp, 1/self.FS)
        for i in range(self.symlen):
            Fwindow = fft(W[i*self.symsamp:(i+1)*self.symsamp])
            freq_power = np.empty(self.pulsefreq.shape)
            for sym_idx, freq in enumerate(self.pulsefreq):
                freq_idx = np.argmin(np.abs(frqcies - freq)) #closest frequency
                assert abs(frqcies[freq_idx] - freq) < self.fr_spacing/3, "frequency not close enough to tune"
                freq_power[sym_idx] = np.sum(np.abs(Fwindow[freq_idx])**2)
            res[i] = np.argmax(freq_power)

        def tobin(n):
            b = bin(n)[2:]
            return '0'*(4 - len(b)) + b

        res_str = ''.join(map(tobin, res))

        return res_str
