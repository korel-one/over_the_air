import sounddevice as sd
import numpy as np

def str2bits(str):
    res = bin(int.from_bytes(str.encode('ascii'), 'big'))[2:]
    return '0'*(8 - (len(res) % 8)) + res

class SoundCommunication:

    def __init__(self, FS, pulseLength, messageBitLength):
        self.FS = FS
        self.pulseLength = pulseLength
        self.pulse = {
                '0': np.sin(self.create_t(self.pulseLength) * 2*np.pi * 1659.9994994994995),
                '1': np.sin(self.create_t(self.pulseLength) * 2*np.pi * 1279)
        }
        # normalize both pulses to same power
        self.pulse['1'] = self.pulse['1']*np.sqrt(np.sum(self.pulse['0']**2) / np.sum(self.pulse['1']**2))
        synT = self.create_t(1)

        self.synchro = self.send('0000000011111111', add_synchro = False)
        self.bitlen = messageBitLength

    def create_t(self, length_seconds):
        return np.linspace(0, length_seconds, int(self.FS*length_seconds))

    def send(self, binstream, add_synchro = True):
        #assert len(binstream) == self.bitlen
        SigLen = self.pulse['0'].size
        sync_size = self.synchro.size if add_synchro else 0
        S = np.empty(len(binstream)*SigLen + sync_size)
        #prepend synchronization at beginning
        if add_synchro:
            S[:sync_size] = self.synchro

        c = 0
        for b in binstream:
            #for every bit, set its respective pulse
            S[c*SigLen + sync_size:(c+1)*SigLen + sync_size] = self.pulse[b]
            c += 1
        return S

    def decode(self, W):
        corr = np.correlate(W[:self.synchro.size*10], self.synchro, 'valid')
        start_syn = np.argmax(np.abs(corr)) # take the absolute value because the microphone
                                            # could inverse + and -, then invert the signal
                                            # accordingly
        print(start_syn/self.FS, corr[start_syn])
        if corr[start_syn] < 0:
            W = -W

        start = start_syn + self.synchro.size
        result = ''
        rlen = int((W.size)/(self.pulseLength*self.FS))
        for i in range(self.bitlen):
            symStart = start + int(i*self.pulseLength*self.FS)
            symEnd = symStart + int(self.pulseLength*self.FS)
            S0corr = np.sum(W[symStart:symEnd] * self.pulse['0'])
            S1corr = np.sum(W[symStart:symEnd] * self.pulse['1'])
            print(S0corr, S1corr, '0' if S0corr > S1corr else '1')
            result += '0' if S0corr > S1corr else '1'
        return result
