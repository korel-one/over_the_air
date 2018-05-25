import numpy as np
from scipy.fftpack import rfft, irfft, rfftfreq

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
                np.logical_and(freq < lowerFreq, freq > higherFreq),
                np.logical_and(freq > -higherFreq,freq < -lowerFreq)
                )
        out_of_band_mask = np.logical_not(in_band_mask)
        f_wnoise[in_band_mask] *= sigma_pass
        f_wnoise[out_of_band_mask] *= sigma_out
        return W + irfft(f_wnoise)[:W.size]

    def addRandomShift(self, W, lowerBoundShift = 0., upperBoundShift = 1.):
        assert lowerBoundShift >= 0.
        assert upperBoundShift > lowerBoundShift
        shift = np.random.random()*(upperBoundShift - lowerBoundShift) + lowerBoundShift
        intshift = int(np.floor(shift * self._comm.FS))
        leftover = self._comm.FS*shift - intshift
        movW = np.interp(np.arange(W.size) + leftover, np.arange(W.size), W)
        return np.hstack((np.zeros(intshift), movW))

    def sendWithNoiseShift(self, r, sigma_pass, sigma_out, lowerBoundShift = 0., upperBoundShift = 1.):
        R = self._comm.send(r)
        Wshift = self.addRandomShift(R, lowerBoundShift, upperBoundShift)
        W = self.addBandpassNoise(Wshift, sigma_pass, sigma_out)
        return W
