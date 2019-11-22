import re
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
fontsize = 9
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['axes.titlepad'] = 7
mpl.rcParams['savefig.dpi'] = 300
plt.rcParams["figure.figsize"] = [4,3]

take_ln = False
moving_avg = True
save = True
save_val = True
window_size = 1_000
dataset_num = 19
rand_walk = True
mean_from_last = 20_000
remove_repeats = True #Caused by starting from the same counter multiple times
clip_val = 100

log_loc = f"//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/wavefunctions/{dataset_num}/"
mse_indicator = "Loss:"

log_file = log_loc + "log.txt"
val_file = log_loc + "val_log.txt"

f = plt.figure()
ax = f.add_subplot(111)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(labeltop=False, labelright=False)

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

plt.ylabel('Absolute Error')
plt.xlabel('Training Iterations (x10$^3$)')

plt.minorticks_on()

save_loc =  f"Z:/Jeffrey-Ede/models/wavefunctions/{dataset_num}/training_losses.png"
plt.savefig( save_loc, bbox_inches='tight', )


plt.show()
