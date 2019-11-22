import re
import matplotlib.pyplot as plt
import numpy as np

take_ln = False
moving_avg = True
save = True
save_val = True
window_size = 1_000
dataset_num = 39#35
rand_walk = True
mean_from_last = 20_000
remove_repeats = True #Caused by starting from the same counter multiple times
clip_val = 100

log_loc = f"//ads.warwick.ac.uk/shared/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/wavefunctions/{dataset_num}/"
mse_indicator = "Loss:"

log_file = log_loc + "log.txt"
val_file = log_loc + "val_log.txt"

notes_file = log_loc + "notes.txt"
with open(notes_file, "r") as f:
    for l in f:
        print(l)

switch = False
losses = []
vals = []
losses_iters = []
with open(log_file, "r") as f:

    numbers = []
    for line in f:
        numbers += line.split(",")

    #print(numbers)
    #print(len(numbers))
    vals = [re.findall(r"([-+]?\d*\.\d+|\d+)", x)[0] for x in numbers if "Val" in x]
    numbers = [re.findall(r"([-+]?\d*\.\d+|\d+)", x)[0] for x in numbers if mse_indicator in x]
    #print(numbers)
    losses = [min(float(x), 25) for x in numbers]

    losses_iters = [i for i in range(1, len(losses)+1)]
try:
    switch = False
    val_losses = []
    val_iters = []
    with open(val_file, "r") as f:
        for line in f:
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)

            for i in range(1, len(numbers), 2):
                val_losses.append(float(numbers[i]))
                val_iters.append(float(numbers[i-1]))
except:
    print("No val log {}".format(val_file))

def moving_average(a, n=window_size):
    ret = np.cumsum(np.insert(a,0,0), dtype=float)
    return (ret[n:] - ret[:-n]) / float(n)

losses = moving_average(np.array(losses[:])) if moving_avg else np.array(losses[:])
losses_iters = moving_average(np.array(losses_iters[:])) if moving_avg else np.array(losses[:])
val_losses = moving_average(np.array(val_losses[:])) if moving_avg else np.array(val_losses[:])
val_iters = moving_average(np.array(val_iters[:])) if moving_avg else np.array(val_iters[:])

print(np.mean((losses[(len(losses)-mean_from_last):-3000])[np.isfinite(losses[(len(losses)-mean_from_last):-3000])]))

#if save:
#    save_loc = ("//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/denoiser-multi-gpu-"+
#                str(dataset_num)+"/log.npy")
#    np.save(save_loc, losses)

#if save_val:
#    save_loc = ("//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/denoiser-multi-gpu-"+
#                str(dataset_num)+"/val_log.npy")
#    np.save(save_loc, val_losses)

#    save_loc = ("//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/denoiser-multi-gpu-"+
#                str(dataset_num)+"/val_iters.npy")
#    np.save(save_loc, val_iters)

#print(losses)
#print(losses_iters)

plt.plot(losses_iters, np.log(losses) if take_ln else losses)
#plt.plot(val_iters, np.log(val_losses) if take_ln else val_losses)
plt.show()



