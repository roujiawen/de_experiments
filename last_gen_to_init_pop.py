from analysis import get_gen_ind_pairs, read_ind_data, overwrite_create
import json
from collections import defaultdict

def generate(INPUT_FOLDER, OUTPUT_FOLDER):
    # get last generation
    gen_ind_pairs = get_gen_ind_pairs(INPUT_FOLDER)
    last_gens = defaultdict(lambda: 0)
    for gen, ind in gen_ind_pairs:
        last_gens[ind] = max(last_gens[ind], gen)

    population_size = len(last_gens)
    # read genes from last generation
    genes = []
    for ind in range(population_size):
        gene = read_ind_data(INPUT_FOLDER, last_gens[ind], ind)["gene"]
        genes.append(gene)
    print "Inferred populationsize =", len(genes)

    overwrite_create(OUTPUT_FOLDER)
    output_filepath = OUTPUT_FOLDER+"/"+"genes.json"
    with open(output_filepath, "w") as write_file:
        json.dump(genes, write_file)
    print "Successfully wrote to", output_filepath

generate("output/M-ali-pb-1", "initial_pop/M-ali-pb-3")
generate("output/M-ali-pb-2", "initial_pop/M-ali-pb-4")
