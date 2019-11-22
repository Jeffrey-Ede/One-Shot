import os
from shutil import copyfile

from diffpy.structure import loadStructure

PARENT_DIR = r"Z:\Jeffrey-Ede\crystal_structures\cifs_no_H\\"
SAVE_LOC = r"C:\dump\simplified_cifs\\"

files = [PARENT_DIR+f+r"\felix.cif" for f in os.listdir(PARENT_DIR)]

for i, file in enumerate(files[:10]):
    struct = loadStructure(file)

    copyfile(file, SAVE_LOC+f"start-{i}.cif")
    struct.write(SAVE_LOC+f"end-{i}.cif", "cif")
