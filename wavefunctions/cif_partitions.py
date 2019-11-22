from diffpy.Structure import loadStructure
from diffpy.Structure.expansion import supercell 
from diffpy.structure.symmetryutilities import positionDifference

from CifFile import ReadCif

import numpy as np
import cv2

from PIL import Image

from periodictable import elements

import pickle

import os

from urllib.request import urlopen
from random import shuffle

parent = r"C:\Users\Jeffrey Ede\Downloads\\"
children = ["COD-selection.txt"] + [f"COD-selection ({i}).txt" for i in range(1, 6)]
save_loc = r"Z:\Jeffrey-Ede\crystal_structures\inorganic_no_H\\"

atom_enums = { e.symbol: e.number for e in elements }
atom_enums["D"] = atom_enums["H"]


def process_elem_string(string):
    """Strips ion denotions from names e.g. "O2+" becomes "O"."""

    elem = ""
    for i, c in enumerate(string):
        try:
            int(c)
            break
        except:
            elem += c

    return elem


num_downloaded = 0
for child in children:
    selection = parent + child

    with open(selection, "r") as f:
        urls = f.read()
        urls = urls.split("\n")
        urls = urls[:-1]

        shuffle(urls)

        num_cifs0 = num_downloaded

        temp_filename = f"{save_loc}tmp.cif"
        for i, url in enumerate(urls):

            try:
                #Download file
                download = urlopen(url).read()

                #Create temporary copy to load structure from
                with open(temp_filename, "wb") as w:
                    w.write(download)

                atom_list = loadStructure(temp_filename).tolist()

                #Make sure it doesn't contain hydrogen
                contains_H = False
                for atom in atom_list:
                    elem_num = atom_enums[process_elem_string(atom.element)]
                
                    if elem_num == 1:
                        contains_H = True
                        break

                if contains_H:
                    continue

                num_downloaded += 1

            except:
                pass

        num_cifs = num_downloaded - num_cifs0
        print(num_cifs)

