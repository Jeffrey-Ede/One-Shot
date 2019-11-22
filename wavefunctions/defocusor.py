"""
Propagate exit wavefunction to various focuses.

Based on C++ code from https://github.com/morawatur/PyEWRecRepo
"""

import tensorflow as tf
import sonnet as snt

import numpy as np

def fft_to_diff(self, x):
    """Change diffraction pattern to fft layout or vice versa."""
    w = x.get_shape()[0]
    h = x.get_shape()[1]

    #Cut into four segments
    tl = x[w - w//2:, h - h//2:]
    tr = x[:w//2, h - h//2:]
    br = x[:w//2, :h//2]
    bl = x[w - w//2:, :h//2]

    #Concatenate
    x = tf.concat(
        [tf.concat([tl, tr], axis=0), tf.concat([bl, br], axis=0)], 
        axis=1
        )

    return x


class DefocusWave(snt.AbstractModule):
    """Defocus square wavefunction."""

    def __init__(
        self, 
        wavelength,
        name="defocus_wave"
        ):

        self._wavelength = wavelength
        self._px_size = px_size

    def _build(self, wave, defocus):

        

        return amplitudes

    def calc_transfer_fn(self, wave_size, px_size, defocus):
        """Contrast transfer function for defocus."""

        #Distances on image
        rec_px_size = 1. / (wave_size*px_size)
        rec_origin = -1. / (2.*px_size)
        line = tf.linspace(
            start=rec_origin,
            stop=rec_origin+rec_px_size*wave_size,
            num=wave_size
            )
        rec_x_dist, rec_y_dist = tf.meshgrid(line, line)
        
        rec_square_dist = rec_x_dist**2 + rec_y_dist**2

        ctf_coeff = np.pi*self._wavelenght*defocus

        phase = ctf_coeff*rec_square_dist
        ctf = tf.complex(
            real=tf.cos(phase),
            imag=tf.sin(phase)
            )

        return ctf

    def propagate_wave(self, wave, ctf):
        
        fft = tf.fft2d(wave)
        ctf = fft_to_diff(ctf)

        fft_prop = fft*ctf
        wave = tf.ifft2d(wave)

        return wave

    def propagate_to_defocus(wave, defocus):

        ctf = self.calc_transfer_fn(wave.get_shape()[1:3], defocus)
        wave = self.propagate_wave(wave, ctf)

        return wave
