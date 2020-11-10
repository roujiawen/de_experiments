import shutil
import os
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/Users/work/de_exper")
from analysis import get_existing_folders
from ali_ang_plane import find_ind_at_gen
from importlib import import_module
exper_config = import_module("output.{}".format("E10-0"))
POPULATION_SIZE = getattr(exper_config, "POPULATION_SIZE")

# plot ang plane at different generations


for plot_gen in [50, 150, 350, 500]:
    plt.figure()

    for exper in ["E10-0", "E10-1"]:
        folders = get_existing_folders(exper)
        for folder in folders:
            ang = [find_ind_at_gen(folder, plot_gen, i)["fitness"] for i in range(POPULATION_SIZE)]
            ali = [find_ind_at_gen(folder, plot_gen, i)["global_stats"][1] for i in range(POPULATION_SIZE)]
            plt.scatter(ang, ali, s=19, alpha=0.5, marker="x", color="crimson")

    # plt.title("optimize, generation={}".format(plot_gen))
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Angular Momentum")
    plt.ylabel("Alignment")
    plt.grid(True)
    plt.savefig("output_plots/{}".format("AngAliPlanePlot_opt_ang_gen_{}.png".format(plot_gen)))
