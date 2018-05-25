import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

def str2bits(str):
    res = bin(int.from_bytes(str.encode('ascii'), 'big'))[2:]
    return '0'*(8 - (len(res) % 8)) + res

class SoundCommunication2:

    def __init__(self, FS, symRate, msgSymLen, sync = '0000000011111111'):
        self.symRate= symRate
        self.FS = FS
        self.symlen = msgSymLen
        self.synclen = len(sync)//4
        self.T = self.create_t(1/symRate)
        self.modpulse = dict()
        for i in range(16):
            self.modpulse[i] = np.sin(self.T * 2 * np.pi * (1015 + i*(1000/16)))
        self.sync = sync
        # normalize both pulses to same power

    def create_t(self, length_seconds):
        return np.linspace(0, length_seconds, int(self.FS*length_seconds))

    def chunks(self, msg):
        chunk =''
        for b in msg:
            chunk += b
            if len(chunk) == 4:
                yield chunk
                chunk = ''

    def send(self, binstream):
        symLen = len(binstream)//4
        S = np.zeros(np.int((symLen + self.synclen)* self.FS/self.symRate))
        msg = self.sync + binstream

        chunk = ''
        for i, chunk in enumerate(self.chunks(msg)):
            S[int(i*self.FS/self.symRate):int((i + 1)*self.FS/self.symRate)]\
                        += self.modpulse[int(chunk, 2)]
        #S /= np.max(np.abs(S))

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
        corr = self._correlate(W[:np.int(self.symlen*self.FS/self.symRate)], correlating_sync)

        start_sync = np.argmax(np.abs(corr)) # take the absolute value because the microphone
                                            # could inverse + and -

        start_msg = start_sync + np.int(self.synclen*self.FS/self.symRate)
        end_msg = start_msg + np.int(self.symlen*self.FS/self.symRate)
        W = W[start_msg:end_msg]
        if W.size < end_msg - start_msg:
            #padd to avoid slow processing in fft
            W = np.hstack((W, np.zeros(end_msg - start_msg - W.size)))
        corr = np.zeros((16, W.size))
        for tune in range(16):
            corr[tune] = self._correlate(W, self.modpulse[tune])
        res = ''

        def tobin(n):
            b = bin(n)[2:]
            return '0'*(4 - len(b)) + b

        sel_corr = corr[:,::int(np.floor(self.FS/self.symRate))]
        res = ''.join(map(tobin, np.argmax(sel_corr, axis=0)))

        return res
