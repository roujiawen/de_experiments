CLUSTERS = ["spring", "broome", "mercer"]

def get_existing_folders(exper):
    """
    e.g.
    Input: E3
    Output: [output/E3_spring, output/E3_broome, output/E3_mercer]
    """
    import os
    all_names = ["output/" + exper + "_" + cluster for cluster in CLUSTERS]
    return filter(os.path.isdir, all_names)

def get_gen_ind_pairs(folder):
    """
    Input: E3_spring
    Output: [(0, 0), (0, 1),...(199, 37)]
    """
    import os
    all_files = os.listdir(folder)
    all_jsons = filter(lambda s: s[:3]=="Gen" and s[-5:]==".json", all_files)
    gen_ind_pairs = map(lambda s: tuple(map(int, s[3:-5].split("_"))), all_jsons)
    return gen_ind_pairs

def overwrite_create(dst):
    """
    Input: E3_spring/last_gen
    """
    import os
    import shutil
    # Remove folder if already exists
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    # Create destination folder
    os.mkdir(dst)

def read_ind_data(folder, gen, ind):
    """
    Input: E3_spring, 0, 57
    """
    import json
    with open(folder+"/Gen{}_{}.json".format(gen, ind), "r") as infile:
        data = json.load(infile)
    return data

def get_fitness(folder, gen, ind):
    """
    Input: E3_spring, 0, 57
    """
    ind_data = read_ind_data(folder, gen, ind)
    return ind_data["fitness"]

def separate_last_generation(exper):
    """
    e.g. Input: E3
    """
    import os
    from collections import defaultdict
    import shutil

    folders = get_existing_folders(exper)
    for folder in folders:
        gen_ind_pairs = get_gen_ind_pairs(folder)

        # Get last generation of every individual
        last_gens = defaultdict(lambda: 0)
        for gen, ind in gen_ind_pairs:
            last_gens[ind] = max(last_gens[ind], gen)

        # Create new destination folder
        dst_folder = folder+"/last_gens"
        overwrite_create(dst_folder)

        # Copy last generations into destination folder
        for ind in range(max(last_gens.keys())):
            shutil.copy(folder+"/Gen{}_{}.png".format(last_gens[ind], ind), dst_folder)

def separate_high_fitness(exper, top_n=50):
    """
    e.g. Input: E3
    """
    from queue import PriorityQueue
    import shutil

    folders = get_existing_folders(exper)
    for folder in folders:
        gen_ind_pairs = get_gen_ind_pairs(folder)

        # Get top top_n fittest individual
        que = PriorityQueue()

        for gen, ind in gen_ind_pairs[:top_n]:
            fitness = get_fitness(folder, gen, ind)
            que.put((fitness, (gen, ind)))

        for gen, ind in gen_ind_pairs[top_n:]:
            fitness = get_fitness(folder, gen, ind)
            que.put((fitness, (gen, ind)))
            que.get()

        # Create new destination folder
        dst_folder = folder+"/fittest_top_{}".format(top_n)
        overwrite_create(dst_folder)

        # Copy last generations into destination folder
        for rank in range(top_n, 0, -1):
            (_, (gen, ind)) = que.get()
            shutil.copy(folder+"/Gen{}_{}.png".format(gen, ind),
                        dst_folder+"/{}_Gen{}_{}.png".format(str(rank).zfill(2), gen, ind))

def plot_multiple_fitness_trajectories(list_of_expers):
    import shutil
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    from importlib import import_module
    import numpy as np

    colors=iter(cm.rainbow(np.linspace(0,1,len(list_of_expers))))
    # Create new plot
    plt.figure(figsize=(15,15), dpi=100)

    for exper in list_of_expers:
        color = next(colors)
        # Read number of generations from experiment config file
        exper_config = import_module("output.{}".format(exper))
        NUM_GENERATION = getattr(exper_config, "NUM_GENERATION")
        POPULATION_SIZE = getattr(exper_config, "POPULATION_SIZE")

        folders = get_existing_folders(exper)
        for folder in folders:
            gen_ind_set = set(get_gen_ind_pairs(folder))

            # Calculate fitness trajectory
            means = []
            stds = []
            maxs = []
            individuals = []
            # Generation 0
            for ind in range(POPULATION_SIZE):
                individuals.append(get_fitness(folder, 0, ind))
            means.append(np.mean(individuals))
            stds.append(np.std(individuals))
            maxs.append(np.max(individuals))
            # Generation 1, 2,... N
            for gen in range(1, NUM_GENERATION+1):
                for ind in range(POPULATION_SIZE):
                    if (gen, ind) in gen_ind_set:
                        individuals[ind] = get_fitness(folder, gen, ind)
                means.append(np.mean(individuals))
                stds.append(np.std(individuals))
                maxs.append(np.max(individuals))

            [means, stds, maxs] = map(np.array, [means, stds, maxs])
            # Plot
            plt.plot(range(0, NUM_GENERATION+1), maxs, label=folder.split("/")[1], color=color)
            # plt.fill_between(range(0, NUM_GENERATION+1), means+stds, means-stds, facecolor=color, alpha=0.2)

    plt.xlabel("Generations")
    plt.ylabel("Angular Momentum")
    plt.ylim(0, 1)
    plt.title("Population fitness evolution trajectories")
    plt.legend()

    # Create new destination folder
    dst_folder = "output/fitness_trajectories_{}".format("_".join(sorted(list_of_expers)))
    overwrite_create(dst_folder)
    # Save plot
    plt.savefig(dst_folder+"/fitness_trajectories.png")

def plot_single_fitness_trajectory(exper):
    import shutil
    import matplotlib.pyplot as plt

    # Read number of generations from experiment config file
    from importlib import import_module
    exper_config = import_module("output.{}".format(exper))
    NUM_GENERATION = getattr(exper_config, "NUM_GENERATION")
    POPULATION_SIZE = getattr(exper_config, "POPULATION_SIZE")

    folders = get_existing_folders(exper)
    for folder in folders:
        gen_ind_set = set(get_gen_ind_pairs(folder))

        # Calculate fitness trajectory
        fitness_trajectory = []
        individuals = []
        # Generation 0
        for ind in range(POPULATION_SIZE):
            individuals.append(get_fitness(folder, 0, ind))
        fitness_trajectory.append(sum(individuals) / POPULATION_SIZE)
        # Generation 1, 2,... N
        for gen in range(NUM_GENERATION):
            for ind in range(POPULATION_SIZE):
                if (gen, ind) in gen_ind_set:
                    individuals[ind] = get_fitness(folder, gen, ind)
            fitness_trajectory.append(sum(individuals) / POPULATION_SIZE)

        # Create new destination folder
        dst_folder = folder+"/fitness_trajectory"
        overwrite_create(dst_folder)

        # Make plot
        plt.figure()
        plt.plot(range(0, NUM_GENERATION+1), fitness_trajectory)
        plt.xlabel("Generations")
        plt.ylabel("Angular Momentum")
        plt.ylim(0, 1)
        plt.title("Population fitness evolution trajectory")
        # Save plot
        plt.savefig(dst_folder+"/fitness_trajectory.png")

def plot_fitness_trajectory(arg):
    """
    e.g. Input: E3
         Input: [E3, E3-1, ...]
    """
    if isinstance(arg, list):
        plot_multiple_fitness_trajectories(arg)
    else:
        plot_single_fitness_trajectory(arg)

def make_hypersearch_table(list_of_expers, outfile):
    # Assuming PARAM_LIMITS doesn't change with the experiments
    import pandas as pd
    import numpy as np
    from importlib import import_module
    exper_config = import_module("output.{}".format(list_of_expers[0]))
    PARAM_LIMITS = getattr(exper_config, "PARAM_LIMITS")

    def get_last_gen_pairs(folder):
        import os
        all_files = os.listdir(folder+"/last_gens")
        all_pngs = filter(lambda s: s[-4:]==".png", all_files)
        gen_ind_pairs = map(lambda s: tuple(map(int, s[3:-4].split("_"))), all_pngs)
        return gen_ind_pairs

    def get_diversity(folder):
        """
        The total pairwise distance in last generation genes. Each dimension
        is scaled to have range 1.
        The larger the score, the larger the avg distance, the higher the diversity.
        """

        def calc_dist(gene1, gene2):
            squares = [((gene1[param] - gene2[param]) / (upper - lower)) ** 2
                       for param, [lower, upper] in PARAM_LIMITS.items()
                       if upper - lower != 0]
            return np.sqrt(np.sum(squares))

        gen_ind_pairs = get_last_gen_pairs(folder)
        genes = []
        for gen, ind in gen_ind_pairs:
            ind_data = read_ind_data(folder, gen, ind)
            genes.append(ind_data["gene"])

        total_pw_dist = 0
        for i in range(len(genes)):
            for j in range(i+1, len(genes)):
                total_pw_dist += calc_dist(genes[i], genes[j])
        return total_pw_dist

    def get_mean_std_fitness(folder):
        gen_ind_pairs = get_last_gen_pairs(folder)
        fitnesses = []
        for gen, ind in gen_ind_pairs:
            ind_data = read_ind_data(folder, gen, ind)
            fitnesses.append(ind_data["fitness"])

        return np.mean(fitnesses), np.std(fitnesses)

    def get_max_fitness(folder):
        import os
        all_files = os.listdir(folder+"/fittest_top_50")
        all_pngs = filter(lambda s: s[-4:]==".png", all_files)
        rank_gen_ind = map(lambda s: tuple(s[:-4].split("_")), all_pngs)
        best = filter(lambda r_g_i: int(r_g_i[0]) == 1, rank_gen_ind)[0]
        gen, ind = int(best[1][3:]), int(best[2])
        best_ind_data = read_ind_data(folder, gen, ind)
        return best_ind_data["fitness"]

    data = []
    for exper in list_of_expers:
        exper_config = import_module("output.{}".format(exper))
        SCALING_PARAM = getattr(exper_config, "SCALING_PARAM")
        CROSSOVER_RATE = getattr(exper_config, "CROSSOVER_RATE")

        folders = get_existing_folders(exper)
        for folder in folders:
            diversity = get_diversity(folder)
            max_fitness = get_max_fitness(folder)
            avg_fitness, std_fitness = get_mean_std_fitness(folder)

            data.append([folder.split("/")[1], SCALING_PARAM, CROSSOVER_RATE,
                         diversity, max_fitness, avg_fitness, std_fitness])


    df = pd.DataFrame(data, columns=['experiment', 'scaling_param_F',
        'crossover_rate_Cr', 'diversity_score', 'max_fitness', 'avg_fitness', 'std_fitness'])

    print df
    df.to_csv("./hypersearch_analysis_{}.csv".format(outfile))



for each in ["E6-2", "E6-3"]:
    separate_last_generation(each)
    separate_high_fitness(each)
    plot_fitness_trajectory(each)

# -------------- ARCHIVE ----------------

# E5s = map(lambda x: "E5-"+str(x), range(12))
# E5s.append("E4-1")
# make_hypersearch_table(E5s, "E5")

# -------------- ARCHIVE ----------------

# E5s = map(lambda x: "E5-"+str(x), range(12))

# for each in E5s:
#     separate_last_generation(each)
#     separate_high_fitness(each)
#     plot_fitness_trajectory(each)

# plot_fitness_trajectory(E5s)


# -------------- ARCHIVE ----------------

# # Analysis 1
# separate_last_generation("E4-1")
# # Analysis 2
# separate_high_fitness("E4-1")
# # Analysis 3
# plot_fitness_trajectory("E4-1")

# # Analysis 1
# separate_last_generation("E4-2")
# # Analysis 2
# separate_high_fitness("E4-2")
# # Analysis 3
# plot_fitness_trajectory("E4-2")

# # Analysis 4
# plot_fitness_trajectory(["E4-1", "E4-2", "E3"])
