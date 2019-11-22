"""Preprocess dataset so it's ready for machine learning.

Author: Jeffrey M. Ede
Email: j.m.ede@warwick.ac.uk
"""

import numpy as np

import os
from scipy.misc import imread

import json

#Location of Jon's simulation outputs
PARENT_DIR = r"//flexo.ads.warwick.ac.uk/shared41/Microscopy/Jeffrey-Ede/models/wavefunctions/output/"
SAVE_FILEPATH = r"F:/wavefunctions/thicknesses.npy"

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
example_dirs = [f + "/" for f in example_dirs]

#Remove example directories without any content
example_dirs = [f for f in example_dirs if len(os.listdir(f)) > 0]

#Crop centers from images
num_examples = len(example_dirs)
thicknesses = []
for i, dir in enumerate(example_dirs):
    print(f"Example {i} of {num_examples}")

    with open(dir+"Diff.json", "r") as f:
        meta = json.load(f)

    t = meta["simulation area"]["z"]["finish"] - meta["simulation area"]["z"]["start"]
    thicknesses.append(t)

thicknesses = np.asarray(thicknesses)
np.save(SAVE_FILEPATH, thicknesses)
