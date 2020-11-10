import shutil
import os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import sys
sys.path.insert(0, "/Users/work/de_exper")
from analysis import get_existing_folders, read_ind_data
from ali_ang_plane import find_ind_at_gen
from importlib import import_module
exper_config = import_module("output.{}".format("E9-0"))
POPULATION_SIZE = getattr(exper_config, "POPULATION_SIZE")

def getImage(path, zoom):
    return OffsetImage(plt.imread(path), zoom=zoom, dpi_cor=200)

# plot ang plane at different generations

samples = [
# ["0", "/Users/work/de_exper/output/E9-0_spring", 437, 56],
["1", "/Users/work/de_exper/output/E9-0_spring", 428, 64, 0.2, 80, 50],
# ["2", "/Users/work/de_exper/output/E9-1_mercer", 351, 41],
["3", "/Users/work/de_exper/output/E9-0_mercer", 257, 7, 0.4, -100, 150]
]

for plot_gen in [500]: #[50, 150, 350, 500]:
    fig, ax = plt.subplots(dpi=200)

    for exper in ["E9-0", "E9-1"]:
        folders = get_existing_folders(exper)
        for folder in folders:
            fit_set = [find_ind_at_gen(folder, plot_gen, i)["fitness"] for i in range(POPULATION_SIZE)]
            _, ang, ali = zip(*fit_set)
            plt.scatter(ang, ali, s=19, alpha=0.5, marker="x", color="crimson")

    for sample in samples:
        id, folder, g, i, zoom, xbox, ybox = sample
        imgpath = "/Users/work/de_exper/paper_plots/scatter_samples/{}.png".format(id)
        jsonpath = "{}/Gen{}_{}.json".format(folder, g, i)
        _, ang, ali = read_ind_data(folder, g, i)["fitness"]
        ab = AnnotationBbox(getImage(imgpath, zoom), (ang, ali), xybox=(xbox, ybox),
                    xycoords='data',
                    boxcoords="offset points",
                    pad=0.2,
                    arrowprops=dict(arrowstyle="->")
                    )
        ax.add_artist(ab)

    # plt.title("optimize, generation={}".format(plot_gen))
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Angular Momentum")
    plt.ylabel("Alignment")
    plt.grid(True)
    plt.savefig("output_plots/{}".format("AngAliPlanePlot_opt_ang-ali_gen_{}.png".format(plot_gen)),
        dpi=200)
