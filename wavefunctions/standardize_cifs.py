import os
from shutil import copyfile

from diffpy.structure import loadStructure

PARENT_DIR = r"Z:\Jeffrey-Ede\crystal_structures\inorganic_no_H\\"
SAVE_LOC = r"Z:\Jeffrey-Ede\crystal_structures\standardized_inorganic_no_H\\"

base_files = os.listdir(PARENT_DIR)
files = [PARENT_DIR+f for f in base_files]

num_files = len(files)
for i, (file, base_file) in enumerate(zip(files, base_files)):
    print(f"File {i} of {num_files}")

    struct = loadStructure(file)
    struct.write(SAVE_LOC+base_file, "cif")
