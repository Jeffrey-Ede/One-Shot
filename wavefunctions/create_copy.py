import os
from shutil import copy


SOURCE = f"//ads.warwick.ac.uk/shared/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/wavefunctions/"
SINK = "//Desktop-sa1evjv/h/sink/wavefunctions/"

WHITELIST = [str(i) for i in range(1, 41)]

for item in os.listdir(SOURCE):
    path = SOURCE+item
    if os.path.isdir(item) and item in WHITELIST:
        os.makedirs(SINK+item)
        for subitem in os.listdir(path):
            subpath = path + "/" + subitem
            dst = SINK + item + "/" + subitem
            if os.path.isdir(subpath):
                os.makedirs(dst)
            elif subitem.split(".")[-1] in ["py", "png"] or subitem == "notes.txt":
                copy(subpath, dst)
    elif os.path.isfile(path):
        try:
            copy(path, SINK+item)
        except:
            continue