CLUSTERS = ["spring", "broome", "mercer"]

def get_existing_folders(exper_name):
    """
    e.g.
    Input: E3
    Output: [E3_spring, E3_broome, E3_mercer]
    """
    import os
    all_names = ["output/" + exper_name + "_" + cluster for cluster in CLUSTERS]
    return filter(os.path.isdir, all_names)

def separate_last_generation(exper_name):
    """
    e.g. Input: E3
    """
    import os
    from collections import defaultdict
    import shutil

    folders = get_existing_folders(exper_name)
    for folder in folders:
        all_files = os.listdir(folder)
        all_jsons = filter(lambda s: s[:3]=="Gen" and s[-5:]==".json", all_files)
        gen_ind_pairs = map(lambda s: map(int, s[3:-5].split("_")), all_jsons)

        # Get last generation of every individual
        last_gens = defaultdict(lambda: 0)
        for gen, ind in gen_ind_pairs:
            last_gens[ind] = max(last_gens[ind], gen)

        # Remove destination folder if already exists
        dst_folder = folder+"/last_gens"
        if os.path.isdir(dst_folder):
            shutil.rmtree(dst_folder)

        # Create destination folder
        os.mkdir(dst_folder)

        # Copy last generations into destination folder
        for ind in range(max(last_gens.keys())):
            shutil.copy(folder+"/Gen{}_{}.png".format(last_gens[ind], ind), dst_folder)

separate_last_generation("E3-1")


# get max generation and num of individuals
# gen_set, ind_set = map(set, zip(*gen_ind_pairs))
# max_gen = max(gen_set)
# num_ind = max(ind_set) + 1

# individuals = {x:[] for x in range(num_ind)}
# for gen, ind in gen_ind_pairs:
#     individuals[ind].append(gen)
