CLUSTERS = ["spring", "broome", "mercer"]

def get_existing_folders(exper):
    """
    e.g.
    Input: E3
    Output: [E3_spring, E3_broome, E3_mercer]
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
    from importlib import import_module

    # Create new plot
    plt.figure()

    for exper in list_of_expers:
        # Read number of generations from experiment config file
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
            plt.plot(range(0, NUM_GENERATION+1), fitness_trajectory)

    plt.xlabel("Generations")
    plt.ylabel("Angular Momentum")
    plt.ylim(0, 1)
    plt.title("Population fitness evolution trajectories")

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


# # Analysis 1
# separate_last_generation("E3-2")
# # Analysis 2
# separate_high_fitness("E3-2")
# # Analysis 3
# plot_fitness_trajectory("E3-2")

# Analysis 4
plot_fitness_trajectory(["E3-2", "E3-1", "E3"])
