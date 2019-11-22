import os
import json
import numpy as np

#COD CIFS from materials science journals without H (Richard specified no H)
CIFS_DIR = r"\\flexo.ads.warwick.ac.uk\shared41\Microscopy\Jeffrey-Ede\crystal_structures\inorganic_no_H\\"
cif_filepaths =[CIFS_DIR + f for f in os.listdir(CIFS_DIR)]

PARENT_DIR = r"\\flexo.ads.warwick.ac.uk\shared41\Microscopy\Jeffrey-Ede\models\wavefunctions\\"
default_json_filepath = PARENT_DIR+"default.json"

NUM_REPEATS = 3

CONFIG_PATH = ""

with open(default_json_filepath, "r") as f:
    default_config = json.load(f)

print(default_config)

def random_config():
    """Change default configuration to random configuration."""

    config = default_config.copy()
    
    #Change some setting to random values
    config[""] = np.random.random()

    return config


if __name__ == "__main__":
    for _ in range(NUM_REPEATS): #Number of times to go through CIFs
        for cif_filepath in cif_filepaths:

            #Save random configuration
            config = random_config()
            with open(CONFIG_PATH, "w") as f:
                json.dump(config, f)

            cmd = ".\clTEM.exe"
            cmd += " --"

            os.system(cmd)