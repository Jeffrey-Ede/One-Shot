"""Partition data into training, validation and test sets by journal of publication"""

import os
from shutil import copyfile

PARENT_DIR = r"F:\wavefunctions_refined_fifth\wavefunctions\\"

TRAIN_DIR = r"H:\wavefunctions_refined_fifth\train\\"
VAL_DIR = r"H:\wavefunctions_refined_fifth\val\\"
TEST_DIR = r"H:\wavefunctions_refined_fifth\test\\"

#Get CIF-wavefunction pairs
cif_wave = os.listdir(r"F:\wavefunctions_refined_fifth\meta\\")
cif_wave = [x.split(".")[0].split("-") for x in cif_wave if x[-5:] == ".json"]
cif_wave = {int(x[0]): int(x[1]) for x in cif_wave}

by_cif = {}
for k in cif_wave:
    if cif_wave[k] in by_cif:
        by_cif[cif_wave[k]].append(k)
    else:
        by_cif[cif_wave[k]] = [k]

VAL_ITERS = [i for i in range(11416, 12631)]
TEST_ITERS = [i for i in range(8489, 11416)]
TRAIN_ITERS = [i for i in range(13000) if not (i in VAL_ITERS) and not (i in TEST_ITERS)]

for mode_num, (iters, dir) in enumerate(zip([TRAIN_ITERS, VAL_ITERS, TEST_ITERS], [TRAIN_DIR, VAL_DIR, TEST_DIR])):
    num_iters = len(iters)
    for i in iters:
        print(f"Mode {mode_num}, Iter {i} of {num_iters}")

        try:
            for j in by_cif[i]:
                filename = f"{j}.npy"
                copyfile(PARENT_DIR+filename, dir+filename)
        except:
            continue