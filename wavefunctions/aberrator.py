"""
TensorFlow implementation of wavefunction aberration. Based on a C++/OpenCL kernel in

https://github.com/JJPPeters/clTEM/blob/master/kernels/generate_tem_image.cl
"""

import tensorflow as tf
import numpy as np

import cv2


## Utility
def scale0to1(img):
    """Rescale image between 0 and 1"""

    img = img.astype(np.float32)

    min = np.min(img)
    max = np.max(img)

    if np.absolute(min-max) < 1.e-6:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img.astype(np.float32)


def disp(img):
    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(img))
    cv2.waitKey(0)
    return


#Functions used in kernel this script is based on
def cModSq(a):
    return tf.abs(a)**2

def cMult(a, b):
    return a*b

cConj = tf.conj

def cPow(a, n):
    return a**n

#Utility
def create_line(wave_size, px_size):
    rec_px_size = 1. / (wave_size*px_size)
    rec_origin = -1. / (2.*px_size)
    line = tf.linspace(
        start=rec_origin,
        stop=rec_origin+rec_px_size*wave_size,
        num=wave_size
        )
    return line

#Apply aberration
def aberrate_wave(
    wave, 
    wavelength,
	C10, C12, C21, C23, C30, C32, C34, C41, C43, C45, C50, C52, C54, C56,
    obj_ap,
	beta,
	delta,
    px_size
    ):
    """Apply aberration to wavefunction
    
    Args:
        wave: Complex wavefunction with shape `[height, width, 1]` to be aberrated 
        wavelength: Wavelength of electron beam in Angstroms
        C10, ..., c56: Aberration coefficients
        obj_ap: Size/convergence of the objective aperture in mrad
        beta: Size/convergence of the condenser aperture semiangle in mrad
        delta: Defocus spread (a term incorporating the chromatic aberrations, see Kirkland 2nd ed., equation 3.41)
        px_size: [height, width] of individual pixels in meters.

    Returns:
        Aberrated wavefunction
    """

    wave_shape = wave.get_shape().as_list()

    wave_x = tf.real(wave)
    wave_y = tf.imag(wave)

    obj_ap2 = obj_ap * 0.001 / wavelength
    beta2 = beta * 0.001 / wavelength

    #Distances from center
    line_x = create_line(wave_shape[0], px_size[0])
    if wave_shape[1] == wave_shape[0] and px_size[1] == px_size[0]:
        line_y = line_x
    else:
        line_y = create_line(wave_shape[1], px_size[1])

    k_x, k_y = tf.meshgrid(line_x, line_y) #Include factor of pi?
    k = tf.sqrt(k_x**2 + k_y**2 + 1.e-10)

    #Apply aberrations
    w = wavelength*tf.complex(k_x, k_y)
    wc = cConj(w)

    # TODO: check the 0.25 factor here is correct (it was 0.5, but Kirkland 2nd ed. eq. 3.42 disagrees)
    temporalCoh = tf.exp( -0.25 * np.pi**2  * delta**2 * cModSq(w)*cModSq(w) / wavelength**2 )
    spatialCoh = tf.exp( -1.0 * np.pi*np.pi * beta2*beta2 * cModSq(w) * 
                        cPow( (C10 + C30*cModSq(w) + C50*cModSq(w)*cModSq(w)), 2)  / wavelength**2 )
    tC10 = 0.5 * C10 * cModSq(w)
    tC12 = 0.5 * cMult(C12, cPow(wc, 2))
    tC21 = cMult(C21, cMult(cPow(wc, 2), w)) / 3.0
    tC23 = cMult(C23, cPow(wc, 3)) / 3.0
    tC30 = 0.25 * C30 * cModSq(w)*cModSq(w)
    tC32 = 0.25 * cMult(C32, cMult(cPow(wc, 3), w))
    tC34 = 0.25 * cMult(C34, cPow(wc, 4))

    tC41 = 0.2 * cMult(C41, cMult(cPow(wc, 3), cPow(w ,2)))
    tC43 = 0.2 * cMult(C43, cMult(cPow(wc, 4), w))
    tC45 = 0.2 * cMult(C45, cPow(wc, 5))
    tC50 = C50 * cModSq(w)*cModSq(w)*cModSq(w) / 6.0
    tC52 = cMult(C52, cMult(cPow(wc, 4), cPow(w ,2))) / 6.0
    tC54 = cMult(C54, cMult(cPow(wc, 5), w)) / 6.0
    tC56 = cMult(C56, cPow(wc, 6)) / 6.0

    cchi = (tC10 + tf.real(tC12) + tf.real(tC21) + tf.real(tC23) + tC30 + tf.real(tC32) + tf.real(tC34) + tf.real(tC41) + 
        tf.real(tC43) + tf.real(tC45) + tf.real(tC50) + tf.real(tC52) + tf.real(tC54) + tf.real(tC56))
    chi = 2.0 * np.pi * cchi / wavelength

    cos_chi = tf.cos(chi)
    sin_chi = tf.sin(chi)
    coherence = temporalCoh * spatialCoh
    output = tf.complex(
        real=coherence * (wave_x * cos_chi + wave_y * sin_chi),
        imag=coherence * (wave_y * cos_chi - wave_x * sin_chi)
        )

    output = tf.where(
        k < obj_ap2,
        output,
        tf.zeros(wave_shape, dtype=tf.complex64) #Aperture blocks propagation
        )

    return output


if __name__ == "__main__":

    def read_params(file):
        """Read aberrations applied to output image at CCD"""

        return 

    from scipy.misc import imread

    #Example amplitude and phase
    loc = f"//ads.warwick.ac.uk/shared/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/wavefunctions/output_refined/25/1/"
    amplitude_filepath = loc + "EW_amplitude.tif"
    phase_filepath = loc + "EW_phase.tif"

    amplitude = imread(amplitude_filepath, mode='F')
    phase = imread(phase_filepath, mode='F')

    wave = amplitude*(np.cos(phase) + 1j*np.sin(phase))

    wave = wave[128:128+256, 128:128+256]

    wave_ph = tf.placeholder(tf.complex64, shape=(256,256))

    wavelength = 1

    aberrated_wave = aberrate_wave(
        wave=wave_ph,
        wavelength=80,
        C10=0e3, 
        C12=1+1j, 
        C21=1+1j, 
        C23=1+1j, 
        C30=1e8, 
        C32=0, 
        C34=0, 
        C41=0, 
        C43=0,
        C45=0, 
        C50=0,
        C52=0,
        C54=0,
        C56=0,
        obj_ap=150,
	    beta=1,
	    delta=10,
        px_size=[512, 512]
        )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #Only use required GPU memory

    with tf.Session(config=config) as sess:

        feed_dict = {wave_ph: wave}
        output = sess.run(aberrated_wave, feed_dict=feed_dict)

        while True:
            print("output.real")
            disp(output.real)
            print("wave.real")
            disp(wave.real)