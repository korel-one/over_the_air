import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq
from scipy.fftpack import rfft, irfft, rfftfreq

def str2bits(str):
    res = bin(int.from_bytes(str.encode('ascii'), 'big'))[2:]
    return '0'*(8 - (len(res) % 8)) + res

def build_shape(symRate, symsamp, sysamp): #200,220
    # [-110 .. 109]
    centered_ = np.arange(symsamp) - symsamp//2
    
    # [-2.5, 2.47727273]
    centered_ = centered_/(sysamp/5)
    
    #centered Gaussian shape
    return np.exp(-(centered_**2))
    

#TODO add coding
class SoundCommunication:

    def __init__(self, FS, symRate, freqbot, freqtop, msgSymLen = 180*8, sync = None):
        if sync: 
            print("specified sync discarded")

        self.symlen = msgSymLen
            
        self.sync = '1100001111100011011101101101011001111100100100001000010100111000001000110011110011110110101110100110'
        self.synclen = len(self.sync)
        
        # symbols per second
        self.symRate= symRate
        
        # samples per sympol 44100/200
        self.symsamp = np.int(FS/symRate) + np.int(FS/symRate)%2
        
        # sampling rate
        self.FS = FS
        
        self.shaping = build_shape(symRate, self.symsamp, self.symsamp)
        
        self.freqbot = freqbot
        self.freqtop = freqtop

    # returns sinc in samples
    def corr_signal(self):
        sinc = self.send()
        return sinc[:self.synclen*self.symsamp]

    def send(self, binstream=str()):
        # message_len + sync_len (bits number)
        total_len = self.synclen + self.symlen
        
        # total number of samples = 220 samples per symbols * symbols number
        # 338800
        S = np.zeros(self.symsamp*total_len)
        
        # weird
        assert len(binstream) <= self.symlen
        # weird
        binstream = binstream + '0'*(self.symlen - len(binstream))
        
        msg = self.sync + binstream
        for i, bit in enumerate(msg):
            # with step e.g. 220 samples per symbol
            l = i*self.symsamp
            r = (i+1)*self.symsamp
            
            S[l:r] += self.shaping * (-1 if bit == '0' else 1)

        freq = (self.freqtop + self.freqbot)/2
        
        # TODO: explain this!!!
        S *= np.sin(np.arange(S.size)/self.FS * 2 * np.pi * freq)
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
        #W = np.copy(W)
        #W = self.bandpass_filter(W, self.freqbot, self.freqtop)
        
        # samples of sinc
        correlating_sync = self.corr_signal()
        
        # correlation complete message with sinc
        corr = np.correlate(W[:(self.symlen + self.synclen) * self.symsamp], correlating_sync)

        #correlating_sync = np.zeros(self.synclen * self.symsamp)
        #for i, bit in enumerate(self.sync):
        #    correlating_sync[i*self.symsamp:(i+1)*self.symsamp]\
        #                += self.shaping * (-1 if bit == '0' else 1)
        #corr = self._correlate(W[:(self.symlen + self.synclen) * self.symsamp], correlating_sync)
        
        freq = (self.freqtop + self.freqbot)/2
        
        # TODO: explain this!!!
        W = np.copy(W)
        W *= np.sin(np.arange(W.size)/self.FS * 2 * np.pi * freq)

        start_sync = np.argmax(np.abs(corr)) # take the absolute value because the microphone
                                            # could inverse + and -
        if corr[start_sync] < 0:
            W = -W

        # shift to the payload position
        start_samp = start_sync + self.synclen*self.symsamp
            
        res = np.zeros(self.symlen, dtype=np.intp)
        
        # TODO: explain why do we iterate over sync + message????
        for i in range(self.symlen):
            jitter_max = 0#int(self.FS*(1/self.symRate)/8)
            #win = W[max(0, start_samp - jitter_max):min(self.symsamp*self.symlen, start_samp  + self.symsamp)]
            
            # sample every 220 samples 
            win = W[start_samp:start_samp  + self.symsamp]
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

        return ''.join(map(str, res))
