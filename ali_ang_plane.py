from analysis import \
    get_existing_folders, get_last_gen_pairs, read_ind_data, get_last_gen_pngs, get_gen_ind_pairs
from imgcat import imgcat
import shutil
import os

def make_scatter1():
    # plot ang-ali plane, color coded by exper number
    import matplotlib.pyplot as plt

    plt.figure()

    exper ="E9-0"
    folders = get_existing_folders(exper)
    for folder in folders:
        gen_ind_pairs = get_last_gen_pairs(folder)
        fit_set = [read_ind_data(folder, g, i)["fitness"] for (g, i) in gen_ind_pairs]
        _, ang, ali = zip(*fit_set)
        plt.scatter(ang, ali,s=19, alpha=0.5, marker="x", label=folder)

    exper ="E9-1"
    folders = get_existing_folders(exper)
    for folder in folders:
        gen_ind_pairs = get_last_gen_pairs(folder)
        fit_set = [read_ind_data(folder, g, i)["fitness"] for (g, i) in gen_ind_pairs]
        _, ang, ali = zip(*fit_set)
        plt.scatter(ang, ali,s=19, alpha=0.5, marker="x", label=folder)

    plt.title("E9-0, E9-1: optimize(ang-ali), last generation")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Angular Momentum")
    plt.ylabel("Alignment")
    plt.legend()
    plt.grid(True)
    plt.show()

class _Getch:
    """Gets a single character from standard input.  Does not echo to the
screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()

class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()

getch = _Getch()

def creat_dir_if_not_there(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def interactive_labeling():
    start_at = 1

    exper = "E8-5"
    folders = get_existing_folders(exper)
    folder = folders[2] # E9-0, E9-1, E8-4 is done
    print folder
    pngs = sorted(get_last_gen_pngs(folder))

    for count, png in enumerate(pngs):
        if count + 1 < start_at:
            continue
        path = folder+"/last_gens/"+png
        imgcat(open(path))
        ch = getch()
        if ch in map(str, range(10)):
            copy_into_folder = folder+"/last_gens/"+ch
            creat_dir_if_not_there(copy_into_folder)
            shutil.copy(path, copy_into_folder)
            print "{} => {}".format(png, ch)
            print
            print
            print
        elif ch=='\r':
            print "Stopped. ^{}/100: {} @ {}".format(count+1, png, folder.split("/")[1])
            break
        else:
            print "Invalid input '{}'!".format(ch)
            print "Stopped. ^{}/100: {} @ {}".format(count+1, png, folder.split("/")[1])
            break


def get_gen_pairs(subfolder):
    import os
    all_files = os.listdir(subfolder)
    all_pngs = filter(lambda s: s[-4:]==".png", all_files)
    gen_ind_pairs = map(lambda s: tuple(map(int, s[3:-4].split("_"))), all_pngs)
    return gen_ind_pairs


def make_scatter2():
    # plot ang-ali plane, color coded by patterns
    import matplotlib.pyplot as plt
    import os
    from collections import defaultdict
    labels = {"0": "other",
              "1": "blob + satellite",
              "2": "blob + stream",
              "3": "milling",
              "4": "two aligned circles",
              "5": "stars",
              "6": "blob + 2 satellites",
              "7": "blob + 3 satellites"}

    expers = ["E9-0", "E9-1"]
    data = defaultdict(lambda: {"ang":[], "ali":[]})

    plt.figure()

    for exper in expers:
        folders = get_existing_folders(exper)
        for folder in folders:
            subfolders = filter(os.path.isdir, map(lambda x: folder + "/last_gens/" +str(x), range(10)))
            for subfolder in subfolders:
                gen_ind_pairs = get_gen_pairs(subfolder)
                if len(gen_ind_pairs) == 0:
                    continue
                fit_set = [read_ind_data(folder, g, i)["fitness"] for (g, i) in gen_ind_pairs]
                _, ang, ali = zip(*fit_set)
                data[subfolder[-1]]["ang"] += ang
                data[subfolder[-1]]["ali"] += ali

    for key in data:
        plt.scatter(data[key]["ang"], data[key]["ali"], s=19, alpha=0.8, marker="x", label=labels[key])

    plt.title("E9-0, E9-1: optimize(ang-ali), last generation")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Angular Momentum")
    plt.ylabel("Alignment")
    plt.legend()
    plt.grid(True)
    plt.show()

def find_ind_at_gen(folder, g, i):
    all_pairs = get_gen_ind_pairs(folder)
    for_this_ind = filter(lambda p: p[1]==i, all_pairs)
    avail_gens = map(lambda p: p[0], for_this_ind)
    gens_le_g = filter(lambda x: x <= g, avail_gens)
    chosen_gen = max(gens_le_g)
    return read_ind_data(folder, chosen_gen, i)

def make_scatter3():
    # plot evolution: gen=50, 100, 150, ...
    import matplotlib.pyplot as plt
    import os
    from collections import defaultdict
    labels = {"0": "other",
              "1": "blob + satellite",
              "2": "blob + stream",
              "3": "milling",
              "4": "two aligned circles",
              "5": "stars",
              "6": "blob + 2 satellites",
              "7": "blob + 3 satellites"}

    expers = ["E9-0", "E9-1"]
    plot_gen = 50
    data = defaultdict(lambda: {"ang":[], "ali":[]})

    plt.figure()

    for exper in expers:
        folders = get_existing_folders(exper)
        for folder in folders:
            subfolders = filter(os.path.isdir, map(lambda x: folder + "/last_gens/" +str(x), range(10)))
            for subfolder in subfolders:
                gen_ind_pairs = get_gen_pairs(subfolder)
                if len(gen_ind_pairs) == 0:
                    continue
                fit_set = [find_ind_at_gen(folder, plot_gen, i)["fitness"] for (g, i) in gen_ind_pairs]
                _, ang, ali = zip(*fit_set)
                data[subfolder[-1]]["ang"] += ang
                data[subfolder[-1]]["ali"] += ali

    for key in data:
        plt.scatter(data[key]["ang"], data[key]["ali"], s=19, alpha=0.8, marker="x", label=labels[key])

    plt.title("E9-0, E9-1: optimize(ang-ali), generation {}".format(plot_gen))
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Angular Momentum")
    plt.ylabel("Alignment")
    plt.legend()
    plt.grid(True)
    plt.show()

def make_scatter4():
    # wanted to plot ang plane -> no plane because no ali data -> only lines
    import matplotlib.pyplot as plt
    import os
    from collections import defaultdict
    labels = {"0": "other",
              "1": "blob + satellite",
              "2": "blob + >2 satellites",
              "3": "two aligned circles",
              "4": "milling",
              "5": "blob + HC satellite",
              "6": "blob + >2 HC satellites",
              "7": "two aligned HCs",
              "8": "three aligned circles"
              }
    param_names = ["Alignment Range", "Adhesion", "Interaction Force", "Gradient Intensity", "Alignment Force", "Noise Intensity", "Angular Inertia", "Velocity", "Cell Density", "Gradient Direction", "Interaction Range"]
    dict_to_list = lambda d: [d[each] for each in param_names]
    expers = ["E8-4", "E8-5"]
    data = defaultdict(lambda: {p:[] for p in param_names})

    plt.figure()

    for exper in expers:
        folders = get_existing_folders(exper)
        for folder in folders:
            subfolders = filter(os.path.isdir, map(lambda x: folder + "/last_gens/" +str(x), range(10)))
            for subfolder in subfolders:
                gen_ind_pairs = get_gen_pairs(subfolder)
                if len(gen_ind_pairs) == 0:
                    continue
                params_t = [dict_to_list(read_ind_data(folder, g, i)["gene"]) for (g, i) in gen_ind_pairs]
                params = zip(*params_t)
                for i, each in enumerate(param_names):
                    data[subfolder[-1]][each] += params[i]

    for key in data:
        x_data = []
        y_data = []
        for i, param in enumerate(param_names):
            x_data.append([i]*len(data[key][param]))
            y_data.append(data[key][param])
        plt.scatter(x_data, y_data, s=19, alpha=0.5, marker="x", label=labels[key])

    plt.title("E8-4, E8-5: optimize ang, last generation")
    # plt.xlim(-0.05, 1.05)
    # plt.ylim(-0.05, 1.05)
    # plt.xlabel("Params")
    # plt.ylabel("Alignment")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    make_scatter3()
