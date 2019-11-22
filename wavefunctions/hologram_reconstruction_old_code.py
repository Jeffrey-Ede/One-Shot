import numpy as np
import glob
import cv2

import arrayfire as af

from skimage.measure import compare_ssim as ssim
from scipy.misc import imread
from scipy.optimize import minimize


def scale0to1(img):
        """Rescale image between 0 and 1"""
        min = np.min(img)
        max = np.max(img)

        if min == max:
            img.fill(0.5)
        else:
            img = (img-min) / (max-min)

        return img.astype(np.float32)


class Utility(object):

    def __init__(self):
        pass

    @staticmethod
    def np_to_af(np_arr, dtype=af.Dtype.f32):
        return af.Array(np_arr.ctypes.data, np_arr.shape, np_arr.dtype.char).as_type(dtype)

    @staticmethod
    def fft_shift(fft):
        return af.shift(fft, fft.dims()[0]//2 + fft.dims()[0]%2, fft.dims()[1]//2 + fft.dims()[1]%2)

    @staticmethod
    def disp(arr):
        cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
        cv2.imshow('CV_Window', scale0to1(arr.__array__()))
        cv2.waitKey(0)
        return

    @staticmethod
    def disp_af(arr):
        cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
        cv2.imshow('CV_Window', scale0to1(arr.__array__()))
        cv2.waitKey(0)
        return

    @staticmethod
    def af_phase(img):
        return np.angle(img.__array__())
        #f = lambda x: x if x < np.pi else np.pi-x
        #vecf = np.vectorize(f)
        #return vecf(img)

    @staticmethod
    def af_amplitude(img):
        return np.abs(img.__array__())

    @staticmethod
    def disp_af_complex_amp(fft, log_of_fft=True):
        amp = np.log(np.absolute(fft)) if log_of_fft else np.absolute(fft)

        cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
        cv2.imshow('CV_Window', scale0to1(amp))
        cv2.waitKey(0)

        return

    @staticmethod
    def disp_af_complex_phase(fft):
        
        cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
        cv2.imshow('CV_Window', scale0to1(af_phase(fft)))
        cv2.waitKey(0)

        return

    @staticmethod
    def save_af_as_npy(arr, filename, save_loc=""):

        if filename[-4:-1] != ".npy":
            filename += ".npy"
        if save_loc[-1] != "/":
            save_loc += "/"

        np.save(save_loc+filename+".npy", arr.__array__())

        return

    @staticmethod
    def get_radial_freq_hist(img, mean=1.0): #Currently unused
        
        abs_shifted_fft = np.abs(np.fft.fftshift(np.fft.fft2(img)))

        rows = cols = int(abs_shifted_fft.shape[0])
        mid_row = mid_col = int(np.ceil(abs_shifted_fft.shape[0]/2))
        max_rad = int(np.ceil(np.sqrt((mid_row)**2 + (mid_col)**2)))+1
        radial_profile = np.zeros(max_rad)

        for col in range(cols):
            for row in range(rows):
                radius = np.sqrt((row-mid_row+1)**2 + (col-mid_col+1)**2)
                idx = int(np.ceil(radius))

                radial_profile[idx] += abs_shifted_fft[col][row]

        return radial_profile

    @staticmethod
    def af_padded_fft2(img, pad_val=0., pad_periods=0):
        #side = img.dims()[0]
        #padded_img = af.constant(0., (1+pad_periods)*side, (1+pad_periods)*side)
        #padded_img[:side, :side] = img
        #fft = af.fft2(padded_img)

        fft = af.fft2(img)

        return fft

    @staticmethod
    def af_unpadded_ifft2(fft, pad_periods=0):
        side = fft.dims()[0] // (1+pad_periods)
        return af.ifft2(fft)[:side, :side]


###########################################################################################

class EWREC(Utility):
    
    def __init__(self, 
                 stack_dir, 
                 wavelength, 
                 rel_pos=None, 
                 rel_pos_method="phase_corr", 
                 series_type = "cubic",
                 series_alternating=True, #Focuses alternating about in-focus position
                 series_middle = None, #Middle focus index only needed if series alternates about centre. If not
                                       #provided, the halfway index will be chosen
                 remove_outer=0,
                 series_increasing=True,
                 defocus_base=None, 
                 defocus_search_range=[0., 10.], #nm**-1?
                 reconstruction_side=None,
                 defocus_sweep_num=10, #Number of defocuses to try initial sweep to find the defocus
                 defocus_search_criteria = ["gradient_plateau"],
                 preprocess_fn=None,
                 param_refinement = False, #Time-consuming refinement of relative positions and defocus values
                 nn_refinement=False, #TODO
                 report_progress=True,
                 pad_periods=0,
                 pad_val=0.,
                 size=512,
                 num_iter=20
                 ): #Number of periods of signal to pad it by for fft

        self.stack_dir = stack_dir
        self.num_iter = num_iter

        self.stack = self.get_stack_from_tifs(stack_dir)

        if remove_outer:
            self.stack = self.stack[remove_outer:-remove_outer]

        if preprocess_fn:
            self.stack = preprocess_fn(self.stack)

        print(self.stack[0].shape)
        self.stack = [cv2.resize(x, (size, size)) for x in self.stack]
        print(self.stack[0].shape)

        self.display_iter_nums = report_progress

        self.pad_periods = pad_periods
        self.pad_value = pad_val

        #Focal series meta
        self.wavelength = wavelength
        self.series_type = series_type
        self.series_alternating = series_alternating
        self.series_mid = series_middle - remove_outer
        self.series_increasing = series_increasing
        self.defocus_base = defocus_base
        self.defocuses = self.defocus_initial_estimate()

        self.stack_side = size#reconstruction_side
        self.stack = self.crop_stack()


    @staticmethod
    def get_stack_from_tifs(dir):

        if dir[-1] != "/":
            dir += "/"

        files = glob.glob(dir+"*.tif")
        stack = [imread(file, mode='F').T for file in files]

        return stack

    def crop_stack(self):
        """Crop parts of images from a stack"""

        c = self.stack[0].shape[0]//2
        b = c - self.stack_side//2
        t = b + self.stack_side
        crops = [x[b:t,b:t]  for x in self.stack]

        return crops

    def correct_cropped_stack_size(self, stack):
        crop_side = min(int(np.log2(stack[0].shape[0]))**2, int(np.log2(self.reconstruction_side))**2)
        cropped_stack = self.resize_stack(self.cropped_stack, crop_side)
        return cropped_stack

    @staticmethod
    def resize_stack(stack, side):
        return [cv2.resize(img, (side, side)) for img in stack]

    def rel_pos_estimate(self, method="phase_corr", stack=None, rel_to_top_left=True):

        if not stack:
            stack = self.stack

        rel_pos = []
        if method == "phase_corr":
            for i in range(1, len(self.stack)):
                rel_pos.append(cv2.phaseCorrelate(self.stack[i-1], self.stack[i]))

        if rel_to_top_left:
            #chain relative positions from the centermost and find the position closest to the mean
            pos = [[0., 0.]]*len(rel_pos)
            for i, dx, dy in enumerate(rel_pos[1:], 1):
                pos[i][0] = pos[i-1][0]+ rel_pos[i][0]
                pos[i][1] = pos[i-1][1]+ rel_pos[i][1]
            mean = [0., 0.]
            for i in range(len(pos)):
                mean[0] += pos[i][0]
                mean[1] += pos[i][1]
            mean[0] /= len(pos)
            mean[1] /= len(pos)
            dists = [(x-mean[0])**2+(y-mean[1])**2 for x, y in pos]      
            idx = dists.index(min(dists))
            
            half_side = self.stack_side/2
            
            return [(half_side+mean[0]-x, half_side+mean[1]-y) for x, y in pos]
        else:
            return rel_pos

    @staticmethod
    def calc_transfer_func(side, wavelength, defocus_change, pad_periods = 0, spher_aber_coeff=None, 
                           aperture_mask=None):
    
        px_dim = 1.+pad_periods

        ctf_coeff = np.pi * wavelength * defocus_change
    
        rec_px_width = 1.0 / (side*px_dim)
        rec_origin = -1.0 / (2.0*px_dim)

        rec_x_dist = rec_origin + rec_px_width * af.range(side, side, dim=0)
        rec_y_dist = rec_origin + rec_px_width * af.range(side, side, dim=1)
        rec_dist2 = rec_x_dist*rec_x_dist + rec_y_dist*rec_y_dist

        ctf_phase = ctf_coeff*rec_dist2

        if spher_aber_coeff:
            ctf_phase += 0.5 * np.pi * wavelength**3 * spher_aber_coeff * rec_dist2**2

        ctf = af.cos(ctf_phase) + complex(0, 1)*af.sin(ctf_phase)

        if aperture_mask:
            ctf *= aperture_mask

        return ctf.as_type(af.Dtype.c32)

    def fft_to_diff(self, fft):
        return self.fft_shift(fft)

    def diff_to_fft(self, diff):
        return self.fft_to_diff(diff)

    def propagate_wave(self, img, ctf):

        fft = self.af_padded_fft2(img, self.pad_value, self.pad_periods)
        ctf = self.diff_to_fft(ctf)
        propagation = self.af_unpadded_ifft2(fft*ctf, self.pad_periods)

        return propagation

    def propagate_to_focus(self, img, defocus, wavelength, pad_periods=0):

        ctf = self.calc_transfer_func(
            side=int(img.dims()[0]*(1+pad_periods)),
            wavelength=wavelength,
            defocus_change=-defocus,
            pad_periods=pad_periods
            )
     
        return self.propagate_wave(img, ctf)

    def propagate_back_to_defocus(self, exit_wave, defocus, wavelength, pad_periods=0):
    
        ctf = self.calc_transfer_func(
            side=int(exit_wave.dims()[0]*(1+pad_periods)),
            wavelength=wavelength,
            defocus_change=defocus,
            pad_periods=pad_periods
            )

        return self.propagate_wave(exit_wave, ctf)


    def reconstruct(self, stack_on_gpu=False):
        """GPU accelerate wavefunction reconstruction and mse calculation"""

        num_iter = self.num_iter

        stack = self.stack
        defocuses = self.defocuses

        stack_gpu = stack if stack_on_gpu else [self.np_to_af(img) for img in stack]

        width = stack[0].shape[0]
        height = stack[0].shape[1]
        exit_wave = af.constant(0, width, height)
        for i in range(num_iter):

            if self.display_iter_nums:
                print("Iteration {0} of {1}".format(i+1, num_iter))

            exit_wave = 0
            for img, idx in zip(stack_gpu, range(len(stack_gpu))):

                #print("Propagation {0} of {1}".format(idx+1, len(stack)))
                exit_wave += self.propagate_to_focus(img, defocuses[idx], self.wavelength)

            exit_wave /= len(stack)

            for idx in range(len(stack)):
                amp = af.abs(stack_gpu[idx])
                stack_gpu[idx] = self.propagate_back_to_defocus(exit_wave, defocuses[idx], self.wavelength)
                stack_gpu[idx] *= amp / af.abs(stack_gpu[idx])

        #TODO: Remove this line! It's for testing
        exit_wave = self.propagate_back_to_defocus(exit_wave, 0, self.wavelength)

        return exit_wave

    def reconstruction_loss(self, stack_gpu, defocus_incr, defocus_ramp):

        defocuses = [incr*ramp for incr, ramp in zip(defocus_incr, defocus_ramp)]
        reconstruction = reconstruct(stack_gpu.copy(), defocuses, stack_on_gpu=True)

        #Use the wavefunction to recreate the original images
        deconstruction = [self.propagate_back_to_defocus(reconstruction, defocus, self.wavelength) \
           for defocus in defocuses]

        losses = [0.]*len(stack_gpu)
        for i in range(len(losses)):
            collapse = af.abs(deconstruction[i])**2
            collapse *= af.mean(stack_gpu[i]) / af.mean(collapse)
            
            losses[i] = af.mean((stack_gpu[i]-collapse)**2)

        return np.max(losses)

    def defocus_initial_estimate(self):
        #Try various defocuses until one is found that matches the expected pattern

        if self.series_type == "linear":
            gen = lambda x: x
        elif self.series_type == "quadratic":
            gen = lambda x: x**2
        elif self.series_type == "cubic":
            gen = lambda x: x**3

        mid = (self.series_mid if self.series_mid else len(self.stack) // 2) if self.series_alternating else 0
        defocus_dir = 1.0 if self.series_increasing else -1.0
    
        defocus_ramp = [self.defocus_base*defocus_dir*np.sign(x-mid)*gen(x-mid) for x in range(len(self.stack))]

        return defocus_ramp 

    def reconstruction_loss_arbitrary_params(self, centres, defocuses):

        stack = self.crop_stack(centres)
        stack_gpu = [self.np_to_af(img, af.Dtype.c32) for img in stack]
        reconstruction = reconstruct(stack_gpu.copy(), defocuses, stack_on_gpu=True)

        losses = [0.]*len(stack_gpu)
        for i in range(len(losses)):
            collapse = af.abs(deconstruction[i])**2
            collapse *= af.mean(stack_gpu[i]) / af.mean(collapse)
            
            losses[i] = af.mean((stack_gpu[i]-collapse)**2)

        return np.max(losses)

    def refine_params(self):
        
        x0 = [x for x, _ in self.rel_pos] + [y for _, y in self.rel_pos] + self.defocuses

        def loss(x):
            len = len(x)
            centres = [[0.,0.]]*(len//3)
            for i in range(len):
                centres[i][0] = x[i]
                centres[i][1] = x[i+len]

            return self.reconstruction_loss_arbitrary_params(centres, x[(2*len//3):])

        refinement = minimize(
            loss,
            x0,
            method='trust-krylov',
            tol=1e-6,
            iter=100
            )
        
        x = refinement.x
        len = len(x)
        centres = [[0.,0.]]*(len//3)
        for i in range(len):
            centres[i][0] = x[i]
            centres[i][1] = x[i+len]

        return centres, x[(2*len//3):]

if __name__ == "__main__":
    
    #stack_dir1 = "Y:/HCSS6/Shared305/JEOL2100/2100/Users/Jeffrey-Ede/focal-series-initial/series2/"
    stack_dir2 = "Y:/HCSS6/Shared305/JEOL2100/2100/Users/Jeffrey-Ede/downsampled_focal_series/series200/"
    ewrec = EWREC(
        stack_dir=stack_dir2,
        wavelength=2.51e-12,
        defocus_base=1.e6,
        series_type = "quadratic",
        series_middle=6,
        remove_outer=0,
        series_increasing=True,
        reconstruction_side=512,
        )
    wave = ewrec.reconstruct()

    util = Utility()
    util.disp(util.af_phase(wave))
    util.disp(util.af_amplitude(wave))

