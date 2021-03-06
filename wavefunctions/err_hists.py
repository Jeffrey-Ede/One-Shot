import numpy as np
import re

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
plt.rcParams["figure.figsize"] = [4, 3]

take_ln = True
moving_avg = True
save = True
save_val = True
window_size = 2500
dataset_num = 8
mean_from_last = 20000
remove_repeats = True #Caused by starting from the same counter multiple times

scale = 1.0
ratio = 1.3 # 1.618
width = scale * 3.3
height = (width / 1.618)
num_data_to_use = 20000
num_hist_bins = 200
mse_x_to = 0.012

f = plt.figure()

#for i in range(1, 9):
#    log_loc = ("Z:/Jeffrey-Ede/models/stem-random-walk-nin-20-"+str(i)+"/")
#    notes_file = log_loc+"notes.txt"
#    with open(notes_file, "r") as f:
#        for l in f: print(l)

#7
#
#
#Reflection padding.
#10
#First convolution 3x3. ReLU activation.
#Better learning policy. Earlier beta decay.
#No non-linearity after final convolutions. Output in [-1,1].
#13
#KNN infilling. Note this does not make sense with noisy edges!
#Ancillary inner network trainer
#Signal combined with uniform noise to make low-duration areas less meaningful.
#16
#Spiral
#Wasserstein fine-tuning of non-spiral
#40x compression, lr 0.0002
#19
#40x compression, lr 0.00015
#Spiral, LR 0.00010, No Noise
#CReLU, 40x compression spiral
#22
#First kernel size decreased from 11 to 5
#3x3 kernels for first convolutions
#RMSProp
#25
#Momentum optimizer, momentum 0.9
#Momentum optimizer, momentum 0.0
#AdaGrad optimizer
#28
#Nesterov
#1/20 coverage spiral
#1/100 coverage spiral
#31
#1/10 coverage spiral
#1/40 coverage. Quartic loss with loss capping. Noisy.
#Quartic loss learning rate decreased from 0.00010 to 0.00005. Noisy.
#34
#40x compression with 1_000_000 training iterations. Noisy.
#1_000_000 iterations; learning rate decreased from 0.00010 to 0.00005. Noisy.
#1/20 coverage spiral. Adversarial training with 5x usual adversarial loss.
#37
#1/20 coverage spiral. Adversarial training with usual adversarial loss.
#Mean squared error without Huberization
#Weighted loss to try and reduce spatially systematic errors.
#40
#MSE without huberization, with capper
#No Huberization, with capper, adjusted for systematic errors
#Tiled input 10x to investigate overparameterization
#43
#Capped MSE. Repeat of 40
#Capped MSE. Losses adjusted based on systematic error. Repeat of 41
#
#46
#
#
#
#
#
#


mode = 4

model_dir = ""

if mode == 1:
    model_dir1 = "Z:/Jeffrey-Ede/models/wavefunctions/19/"
    model_dir2 = "Z:/Jeffrey-Ede/models/wavefunctions/39/"

    labels_sets = [["$n$=1, Transforms", 
                    "$n$=1, Transforms, Parameters", 
                    "$n$=1, Transforms, Parameters, Materials",
                    "$n$=3, Transforms", 
                    "$n$=3, Transforms, Parameters", 
                    "$n$=3, Transforms, Parameters, Materials"
                    ]]
    sets = [[model_dir1+"train_err2.npy", 
             model_dir1+"2nd_val_err.npy", 
             model_dir1+"val_err.npy",
             model_dir2+"train_err.npy",
             model_dir2+"val_err.npy",
             model_dir2+"train_err2.npy"
             ]]

    halfway = 2

elif mode == 2:
    labels_sets = [["$n$=1, Transforms", 
                    "$n$=1, Transforms, Parameters",
                    "$n$=3, Transforms", 
                    "$n$=3, Transforms, Parameters"
                    ]]
    sets = [[
        "Z:/Jeffrey-Ede/models/wavefunctions/34/train_loc.npy", 
        "Z:/Jeffrey-Ede/models/wavefunctions/34/val_loc.npy",
        "Z:/Jeffrey-Ede/models/wavefunctions/37/train_loc.npy", 
        "Z:/Jeffrey-Ede/models/wavefunctions/37/val_loc.npy"
        ]]

    halfway = 1

elif mode == 3:
    labels_sets = [["$n$=3, Transforms", 
                    "$n$=3, Transforms, Parameters, Materials"
                    ]]
    sets = [[
        "Z:/Jeffrey-Ede/models/wavefunctions/40/train_err.npy", 
        "Z:/Jeffrey-Ede/models/wavefunctions/40/val_err.npy"
        ]]

    halfway = -1
elif mode == 4:
    labels_sets = [["$n$=1, Transforms", 
                    "$n$=1, Transforms, Parameters",
                    "$n$=1, Transforms, Parameters, Materials",
                    "$n$=3, Transforms",
                    "$n$=3, Transforms, Parameters",
                    "$n$=3, Transforms, Parameters, Materials"
                    ]]
    sets = [[
        "Z:/Jeffrey-Ede/models/wavefunctions/24/train_err2.npy",
        "Z:/Jeffrey-Ede/models/wavefunctions/24/val_err2.npy",
        "Z:/Jeffrey-Ede/models/wavefunctions/24/unseen_err2.npy",
        "Z:/Jeffrey-Ede/models/wavefunctions/38/train_err2.npy",
        "Z:/Jeffrey-Ede/models/wavefunctions/38/val_err2.npy",
        "Z:/Jeffrey-Ede/models/wavefunctions/38/unseen_err2.npy"
        ]]

    halfway = 2


max_mse = 1.4088
for i, (data_nums, labels) in enumerate(zip(sets, labels_sets)):
    for j, dataset_num in enumerate(data_nums):
        hist_file = model_dir+dataset_num
        mses = np.load(hist_file)

        max_mse = max(np.max(mses), max_mse)
print("max_mse:", max_mse)


losses_sets = []
iters_sets = []
for i, (data_nums, labels) in enumerate(zip(sets, labels_sets)):

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(labeltop=False, labelright=False)
    for j, dataset_num in enumerate(data_nums):
        hist_file = model_dir+dataset_num

        mses = np.load(hist_file)
        #for x in mses: print(x)
        print(np.mean(mses), np.std(mses))
        #mses = [x for x in mses if x < 5]
        
        bins, edges = np.histogram(mses, 100, (0., max_mse))
            
        edges = 0.5*(edges[:-1] + edges[1:])
        if j == halfway+1:
            plt.gca().set_prop_cycle(None)

        linestyle = '-' if j <= halfway else '--'
        plt.plot(edges, bins, label=labels[j], linewidth=1, linestyle=linestyle)

    plt.ylabel('Frequency')
    plt.xlabel('Mean Absolute Error')
    if mode == 1:
        plt.ylim(top=990)
    elif mode == 2:
        plt.ylim(top=1150)

    plt.legend(loc='upper right', frameon=True, fontsize=8, facecolor='white', framealpha=.5, edgecolor="white")

    plt.axvline(x=0.75, color='black', linestyle='--', linewidth=0.6)

    save_loc =  "Z:/Jeffrey-Ede/models/wavefunctions/" + f"hist{mode}.png"
    plt.savefig( save_loc, bbox_inches='tight', )

    plt.gcf().clear()
