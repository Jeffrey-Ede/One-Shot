"""Preprocess dataset so it's ready for machine learning.

Author: Jeffrey M. Ede
Email: j.m.ede@warwick.ac.uk
"""

import numpy as np

import os
from scipy.misc import imread

from shutil import copyfile

#Location of Jon's simulation outputs
PARENT_DIR = r"//flexo.ads.warwick.ac.uk/shared41/Microscopy/Jeffrey-Ede/models/wavefunctions/output_single_refined/"
SAVE_DIR = r"F:/wavefunctions_single_refined/"

#Image cropping
IMAGE_SIZE = 512
CROP_SIZE = 320
crop_start = (IMAGE_SIZE - CROP_SIZE) // 2

#Function to list full filepaths to items in a directory
full_listdir = lambda dir: [dir+f for f in os.listdir(dir)]

#Directories containing groups of example directories
dirs = full_listdir(PARENT_DIR)
dirs = [f+"/" for f in dirs]

#Get individual example directories
example_dirs = []
for dir in dirs:
    example_dirs += full_listdir(dir)
example_dirs = [f+"/" for f in example_dirs] 

#Remove example directories without any content
example_dirs = [f for f in example_dirs if len(os.listdir(f)) > 0]

#Crop centers from images
num_examples = len(example_dirs)
#print(num_examples); quit()
for i, dir in enumerate(example_dirs):
    print(f"Preparing example {i} of {num_examples}")

    #Read files
    amplitide_filepath = dir + "EW_amplitude.tif"
    phase_filepath = dir + "EW_phase.tif"

    cif_num = int(amplitide_filepath.split("/")[-3])

    amplitude = imread(amplitide_filepath, mode='F')
    phase = imread(phase_filepath, mode='F')

    #Crop images
    amplitude = amplitude[crop_start:crop_start+CROP_SIZE, crop_start:crop_start+CROP_SIZE]
    phase = phase[crop_start:crop_start+CROP_SIZE, crop_start:crop_start+CROP_SIZE]

    #Proprocessing
    amplitude /= np.mean(amplitude)

    wavefunction = (amplitude*(np.cos(phase) + 1.j*np.sin(phase))).astype(np.complex64)

    subset = "train" if i <= 0.8*num_examples else "val"

    #Save data to new location
    save_filepath = SAVE_DIR + f"wavefunctions/{subset}/{i}.npy"

    copyfile(dir+"Diff.json", SAVE_DIR + f"meta/{subset}/{i}-{cif_num}.json")

    np.save(save_filepath, wavefunction)
