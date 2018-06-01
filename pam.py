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
        shaping = np.exp(-(((np.arange(self.symsamp) - self.symsamp//2)/(self.symsamp/5))**2))
        self.shaping = shaping

        self.freqbot = freqbot
        self.freqtop = freqtop

        self.pulse = np.zeros((2, self.symsamp))

        self.sync = sync
        # normalize both pulses to same power

    def create_t(self, length_seconds, samples):
        return np.linspace(0, length_seconds, samples)

    def corr_signal(self):
        sinc = self.send(b'')
        return sinc[:self.synclen*self.symsamp]

    def send(self, msg):
        assert type(msg) == bytes or type(msg) == bytearray
        binstream = ''
        for byte in msg:
            for i in range(8):
                #convert into string of bits
                bit = str(((byte << i) & 0b10000000) >> 7)
                assert bit in {'0','1'}
                binstream += bit
        total_len = self.synclen + self.symlen
        S = np.zeros(self.symsamp*total_len)
        assert len(binstream) <= self.symlen
        binstream = binstream + '0'*(self.symlen - len(binstream))
        msg = self.sync + binstream

        for i, bit in enumerate(msg):
            S[i*self.symsamp:(i+1)*self.symsamp]\
                        += self.shaping * (-1 if bit == '0' else 1)
        S *= np.sin(np.arange(S.size)/self.FS * 2 * np.pi * (self.freqtop + self.freqbot)/2)

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

    def decode(self, W, debug=False):
        W = np.copy(W)
        #W = self.bandpass_filter(W, self.freqbot, self.freqtop)
        correlating_sync = self.corr_signal()

        corr = np.correlate(W[:4*self.FS], correlating_sync)
        W *= np.sin(np.arange(W.size)/self.FS * 2 * np.pi * (self.freqtop + self.freqbot)/2)


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
            assert win.size == self.symsamp
            if debug:
                eye_win = W[start_samp - self.symsamp//2:start_samp + self.symsamp + self.symsamp//2]
                eye_corr = np.correlate(eye_win, self.shaping)
                plt.plot(eye_corr)
            win_corr = np.correlate(win, self.shaping)
            max_idx = np.argmax(np.abs(win_corr))

            #if abs(max_idx - jitter_max) > 10:
            #    start_samp += self.symsamp + max_idx - jitter_max
            #else:
            #    start_samp += self.symsamp
            start_samp += self.symsamp

            res[i] = 1 if win_corr[max_idx] > 0 else 0
        if debug: plt.show()

        binstream = ''.join(map(str, res))
        assert len(binstream) % 8 == 0
        r = bytearray()
        for i in range(0, len(binstream), 8):
            r += int(binstream[i:i+8], 2).to_bytes(1, 'big')
        return r
