import os
import subprocess
import multiprocessing
import json
import numpy as np
import random

num_gpus = 1

#COD CIFS from materials science journals without H (Richard specified no H)
CIFS_DIR = r"\\flexo.ads.warwick.ac.uk\shared41\Microscopy\Jeffrey-Ede\crystal_structures\standardized_inorganic_no_H"
cif_filepaths = [r"Z:\Jeffrey-Ede\crystal_structures\standardized_inorganic_no_H\666.cif"]

PARENT_DIR = r"\\flexo.ads.warwick.ac.uk\shared41\Microscopy\Jeffrey-Ede\models\wavefunctions"
default_json_filepath = os.path.join(PARENT_DIR, "default.json")
failed_file_filepath = os.path.join(PARENT_DIR, "failed_files.txt")

EXE_DIR = r"\\flexo.ads.warwick.ac.uk\shared41\Microscopy\Jeffrey-Ede\models\wavefunctions\clTEM_files"
exe_filepath = os.path.join(EXE_DIR, "clTEM_cmd.exe")

OUTPUT_DIR = os.path.join(PARENT_DIR, "output_single")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

NUM_REPEATS = 5000

CONFIG_DIR = os.path.join(PARENT_DIR, "temp_single")
config_filepath = os.path.join(CONFIG_DIR, "current_config_single.json")
if not os.path.exists(CONFIG_DIR):
    os.makedirs(CONFIG_DIR)

with open(default_json_filepath, "r") as f:
    default_config = json.load(f)


failed_paths = []
with open(failed_file_filepath, 'r') as ff:
    lines = ff.readlines()
    for l in lines:
        failed_paths.append(l.rstrip())

# print(default_config)

def random_config():
    """Change default configuration to random configuration."""

    config = default_config.copy()
    
    # Things to randomise
    # Voltage (use some presets)
    # aperture size
    # convergence
    # defocus spread
    voltages = [300, 200, 80]

    config["microscope"]["voltage"] = random.choice(voltages)
    config["microscope"]["aperture"] = np.random.uniform(5, 30)
    config["microscope"]["delta"] = np.random.uniform(0, 20)
    config["microscope"]["alpha"] = np.random.uniform(0.1, 2)

    # aberrations
    config["microscope"]["aberrations"]["C10"]["val"] = np.random.uniform(-30, 30)

    config["microscope"]["aberrations"]["C12"]["mag"] = np.random.uniform(-50, 50)
    config["microscope"]["aberrations"]["C12"]["ang"] = np.random.uniform(0, 180)

    config["microscope"]["aberrations"]["C21"]["mag"] = np.random.uniform(-1000, 1000)
    config["microscope"]["aberrations"]["C21"]["ang"] = np.random.uniform(0, 180)

    config["microscope"]["aberrations"]["C23"]["mag"] = np.random.uniform(-1000, 1000)
    config["microscope"]["aberrations"]["C23"]["ang"] = np.random.uniform(0, 180)

    config["microscope"]["aberrations"]["C30"]["val"] = np.random.uniform(-500, 500)

    return config

def do_sim(cif_filepath):
    #
    # This is a real bodge to match the device to the thread....
    #
    device = 1 #int(multiprocessing.current_process().name[-1]) - 1

    device_string = "0:%s" % device

    if cif_filepath in failed_paths:
        return

    out_paths = []

    for repetition in range(NUM_REPEATS):
        cif_name = os.path.splitext(os.path.basename(cif_filepath))[0]
        out_filepath = os.path.join(OUTPUT_DIR, cif_name)
        out_repeat_filepath = os.path.join(out_filepath, str(repetition))

        if os.path.exists(os.path.join(out_repeat_filepath, 'Image.tif')):
            continue  # get out this loop as we already have data here

        out_paths.append(out_repeat_filepath)

    if len(out_paths) == 0:
        return

    print("\n\nSimulating on device:" + device_string + " using file: " + cif_filepath)
    #print("\n\n\n")

    # for repetition in range(NUM_REPEATS):  # Number of times to go through CIFs
    #     #
    #     # Create output folder
    #     #

    #     # make a folder for each cif
    #     cif_name = os.path.splitext(os.path.basename(cif_filepath))[0]
    #     out_filepath = os.path.join(OUTPUT_DIR, cif_name)
        
    #     # make a folder for each repetition
    #     out_repeat_filepath = os.path.join(out_filepath, str(repetition))

    #     # while os.path.exists(out_repeat_filepath):
    #     #     counter += 1
    #     #     out_repeat_filepath = os.path.join(out_filepath, str(counter))

    #     if os.path.exists(os.path.join(out_repeat_filepath, 'Image.tif')):
    #         continue  # get out this loop as we already have data here

    for out_path in out_paths:
        
        try:

            if not os.path.exists(out_path):
                os.makedirs(out_path)

            #
            # Randomise the simulation parameters
            #

            # Save random configuration
            config = random_config()
            with open(config_filepath, "w") as f:
                json.dump(config, f)

            #
            # Randomise the structure inputs
            #

            # randomise the cell depth (between 5 nm and 100 nm)
            cell_depth = np.random.uniform(50, 1000)
            cell_widths = np.random.uniform(50, 100)
            cell_string = "%s,%s,%s" % (cell_widths, cell_widths, cell_depth)

            # randomise the zone axis (only up to 2)
            zone_h = np.random.randint(0, 3)
            zone_k = np.random.randint(0, 3)
            zone_l = np.random.randint(0, 3)
            zone_string = "%s,%s,%s" % (zone_h, zone_k, zone_l)

            # random tilt perturbations (normal distribution)
            tilt_a = np.random.normal(scale=0.1)
            tilt_b = np.random.normal(scale=0.1)
            tilt_c = np.random.normal(scale=0.1)
            tilt_string = "%s,%s,%s" % (tilt_a, tilt_b, tilt_c)

            #
            # Do the simulation
            #

            # FNULL = open(os.devnull, 'w') # used to suppress the output
            subprocess.call([exe_filepath, cif_filepath, "-s"+cell_string, "-z"+zone_string, "-t"+tilt_string, "-o" + out_path, "-d"+device_string, "-c" + config_filepath])#, stdout=FNULL)

            #if not os.path.exists(os.path.join(out_path, 'Image.tif')):
            #    # file didn't simulate
            #    with open(failed_file_filepath, "a") as ff:
            #        ff.write(cif_filepath + "\n")

            #    return
        except:
            continue


if __name__ == "__main__":

    # with multiprocessing.Pool(num_gpus) as p:
    #     p.map(do_sim, cif_filepaths)

    with open(failed_file_filepath, "a") as ff:
        ff.write("\n*\nStarting run\n*\n")

    for cf in cif_filepaths:
        do_sim(cf)
