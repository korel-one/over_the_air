import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import rfft, irfft, rfftfreq
from scipy.fftpack import fftfreq, fft, ifft

class Tester:
    def __init__(self, communicator):
        self._comm = communicator

    def addBandpassNoise(self, W, sigma_pass, sigma_out):
        """add gaussian colored noise with stddev sigma_pass in
        the 1kHz - 2Khz frequency range and stddev sigma_out
        outside the frequency range above"""
        lowerFreq = 1000 #1Khz
        higherFreq = 2000
        v = W.size
        v-=1
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v+=1
        white_noise = np.random.normal(0, 1, v)

        freq = rfftfreq(white_noise.size, d = 1/self._comm.FS)
        f_wnoise = rfft(white_noise)

        # If our original signal time was in seconds, this is now in Hz
        in_band_mask = np.logical_or(
                np.logical_and(freq > lowerFreq, freq < higherFreq),
                np.logical_and(freq > -higherFreq,freq < -lowerFreq)
                )
        out_of_band_mask = np.logical_not(in_band_mask)
        f_wnoise[in_band_mask] *= sigma_pass
        f_wnoise[out_of_band_mask] *= sigma_out
        return W + irfft(f_wnoise)[:W.size]

    def plot_frequencies(self, R, fromfrq=900, tofrq=2100):
        #size of R becomes next power of 2
        Rold = R
        v = R.size
        v-=1
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v+=1
        R = np.zeros(v)
        R[:Rold.size] = Rold
        freq = np.roll(fftfreq(R.size, 1/self._comm.FS), R.size//2)
        Rtrans = np.roll(fft(R), R.size//2)
        b = np.argmax(freq > fromfrq)
        t = np.argmax(freq > tofrq)
        plt.plot(freq[b:t], np.abs(Rtrans[b:t]))
        plt.show()

    def addRandomShift(self, W, lowerBoundShift = 0., upperBoundShift = 1.):
        assert lowerBoundShift >= 0.
        assert upperBoundShift > lowerBoundShift
        shift = np.random.random()*(upperBoundShift - lowerBoundShift) + lowerBoundShift
        intshift = int(np.floor(shift * self._comm.FS))
        leftover = self._comm.FS*shift - intshift
        movW = np.interp(np.arange(W.size) + leftover, np.arange(W.size), W)
        return np.hstack((np.zeros(intshift), movW))

    def diff(self, bytes_a, bytes_b, chunk_len=1):
        assert type(bytes_a) in {bytes, bytearray}
        assert type(bytes_b) in {bytes, bytearray}
        def to_bitstring(bytes):
            res =''
            for byte in bytes:
                for i in range(8)[::-1]:
                    res += str((byte >> i) & 1)
            return res
        def chunks(msg):
            chunk =''
            for b in msg:
                chunk += b
                if len(chunk) == chunk_len:
                    yield chunk
                    chunk = ''

        a = to_bitstring(bytes_a)
        b = to_bitstring(bytes_b)

        assert len(a) == len(b), "Length is not the same"

        res = ''
        err = 0
        for i, (l, r) in enumerate(zip(chunks(a), chunks(b))):
            if l == r:
                res += l
            else:
                res += 'X'*chunk_len
                err+=1
        return res, err/(i+1), err

    def sendWithNoiseShift(self, r, sigma_pass, sigma_out, lowerBoundShift = 0., upperBoundShift = 1.):
        R = self._comm.send(r)
        Wshift = self.addRandomShift(R, lowerBoundShift, upperBoundShift)
        W = self.addBandpassNoise(Wshift, sigma_pass, sigma_out)
        return W

    def pad(self, W, padding=0.5):
        return np.hstack((np.zeros(int(self._comm.FS*padding)), W, np.zeros(int(self._comm.FS*padding))))
