import os
import random
import json
from copy import deepcopy
from operator import add

import numpy as np
from model import Model

from multiprocess_setup import *
#assert

import sys

TRIAL_NAME = sys.argv[1]

from importlib import import_module
exper_config = import_module("output.{}".format(TRIAL_NAME))
SIGNIFICANT_RANGE = getattr(exper_config, "SIGNIFICANT_RANGE")
WHICH_ORDER_PARAM = getattr(exper_config, "WHICH_ORDER_PARAM")
GENERAL_PARAMS = getattr(exper_config, "GENERAL_PARAMS")
PARAM_LIMITS = getattr(exper_config, "PARAM_LIMITS")
POPULATION_SIZE = getattr(exper_config, "POPULATION_SIZE")
NUM_GENERATION = getattr(exper_config, "NUM_GENERATION")
SCALING_PARAM = getattr(exper_config, "SCALING_PARAM")
CROSSOVER_RATE = getattr(exper_config, "CROSSOVER_RATE")
# SPACE_DIAGONAL = getattr(exper_config, "SPACE_DIAGONAL")
# INITIAL_THRESHOLD = getattr(exper_config, "INITIAL_THRESHOLD")
# DECAY_RATE = getattr(exper_config, "DECAY_RATE")
DENSITIES = getattr(exper_config, "DENSITIES")
MAX_OR_MIN = getattr(exper_config, "MAX_OR_MIN")
NUM_REPEATS = getattr(exper_config, "NUM_REPEATS")
STARTING_REP_ID = getattr(exper_config, "STARTING_REP_ID")

def randomize(param):
    lower = PARAM_LIMITS[param][0]
    upper = PARAM_LIMITS[param][1]
    return lower + np.random.random() * (upper-lower)

def gaussian(param):
    width = PARAM_LIMITS[param][1] - PARAM_LIMITS[param][0]
    return np.random.normal(scale=0.01*width)

def random_gene():
    gene = {}
    for each in PARAM_LIMITS:
        gene[each] = randomize(each)
    return gene

def check_boundary(candidate):
    """If a parameter exceeds feasible range, use a random value from the
    feasible range instead."""
    gene = {}
    for each in candidate:
        exceeds_upper = candidate[each] > PARAM_LIMITS[each][1]
        exceeds_lower = candidate[each] < PARAM_LIMITS[each][0]
        if exceeds_upper or exceeds_lower:
            gene[each] = randomize(each)
        else:
            gene[each] = candidate[each]
    return gene

def calc_dist(gene1, gene2):
    squares = [((gene1[param] - gene2[param]) / (upper - lower)) ** 2
               for param, [lower, upper] in PARAM_LIMITS.items()
               if upper - lower != 0]
    return np.sqrt(np.sum(squares))

def log_progress(text):
    with open(OUT_PATH+"progress.log", "w") as outfile:
        outfile.write(text)

def evolve():
    log_progress("Initializing...")
    # Initialize population
    population = []
    models = []

    # Random genes for initial population
    for indiv_id in range(POPULATION_SIZE):
        gene = random_gene()
        model = Model(gene, SIGNIFICANT_RANGE, WHICH_ORDER_PARAM, GENERAL_PARAMS, NUM_REPEATS)
        models.append(model)

    all_repeats = reduce(add, [zip([i]*NUM_REPEATS, # indiv_id
                                   range(NUM_REPEATS), # rep_id
                                   m.repeats # Repeat object
                                   ) for i, m in enumerate(models)])

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for indiv_id, rep_id, repeat in executor.map(model_wrapper, all_repeats):
            models[indiv_id].repeats[rep_id] = repeat

    for indiv_id in range(POPULATION_SIZE):
        model = models[indiv_id]
        model.save(OUT_PATH+"Gen0_{}".format(indiv_id))
        population.append(model)

    # Evolve
    for gen_id in range(1, NUM_GENERATION+1):
        log_progress("Evolving Generation #{}...".format(gen_id))

        # # Calculate cc_threshold (ref: Eq. 3)
        # cc_threshold = INITIAL_THRESHOLD * SPACE_DIAGONAL * (
        #     (NUM_GENERATION - gen_id) / float(NUM_GENERATION)
        # ) ** DECAY_RATE

        models = []
        repeats = []
        for indiv_id in range(POPULATION_SIZE):
            this_gene = population[indiv_id].gene
            # # Convergence control strategy (ref: 10.1109/CEC.2012.6252891)
            # dist_from_base = float("-inf")
            # num_attempts = 0
            # while dist_from_base < cc_threshold and num_attempts < 5:
            if True:
                # Mutate
                base, dif1, dif2 = np.random.choice(
                    [m.gene for m in population if m.gene != this_gene],
                    size=3,
                    replace=False)
                variant = {
                    each_param: base[each_param]
                    + SCALING_PARAM * (dif1[each_param] - dif2[each_param])
                    for each_param in this_gene
                }
                variant = check_boundary(variant)
                # Crossover
                i_rand = np.random.randint(len(this_gene))
                trial = {
                    each_param: variant[each_param]
                    if (np.random.random() <= CROSSOVER_RATE) or (j == i_rand)
                    else this_gene[each_param]
                    for j, each_param in enumerate(this_gene)
                }
                # # Calculate distance between trial and base
                # dist_from_base = calc_dist(trial, base)
                # num_attempts += 1

            # if dist_from_base < cc_threshold:
            #     # No new trial vector, give dummy trial vector
            #     trial = None

            model = Model(trial, SIGNIFICANT_RANGE, WHICH_ORDER_PARAM, GENERAL_PARAMS, NUM_REPEATS)
            models.append(model)

        all_repeats = reduce(add, [zip([i]*NUM_REPEATS, # indiv_id
                                       range(NUM_REPEATS), # rep_id
                                       m.repeats # Repeat object
                                       ) for i, m in enumerate(models)])

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for indiv_id, rep_id, repeat in executor.map(model_wrapper, all_repeats):
                models[indiv_id].repeats[rep_id] = repeat

        for indiv_id in range(POPULATION_SIZE):
            model = models[indiv_id]
            # Selection
            if MAX_OR_MIN=="MAX":
                if model.fitness and model.fitness >= population[indiv_id].fitness:
                    population[indiv_id] = model
                    model.save(OUT_PATH+"Gen{}_{}".format(gen_id, indiv_id))
            else:
                if model.fitness and model.fitness <= population[indiv_id].fitness:
                    population[indiv_id] = model
                    model.save(OUT_PATH+"Gen{}_{}".format(gen_id, indiv_id))


if __name__ == "__main__":
    import time
    start_time = time.time()
    if len(DENSITIES) == 0:
        EXPER_NAME = TRIAL_NAME
        OUT_PATH = "output/{}/".format(EXPER_NAME)
        if not os.path.exists(OUT_PATH):
            os.makedirs(OUT_PATH)
        evolve()
    elif DENSITIES[0] is None:
        for i, _ in enumerate(DENSITIES):
            EXPER_NAME = "{}_{}".format(TRIAL_NAME, str(i))
            OUT_PATH = "output/{}/".format(EXPER_NAME)
            if not os.path.exists(OUT_PATH):
                os.makedirs(OUT_PATH)
            evolve()
    else:
        for i, den in enumerate(DENSITIES):
            if i < STARTING_REP_ID:
                continue
            PARAM_LIMITS["Cell Density"]=[den, den]
            if (len(DENSITIES)>1) and all(_==DENSITIES[0] for _ in DENSITIES):
                EXPER_NAME = "{}_{}".format(TRIAL_NAME, str(i))
            else:
                EXPER_NAME = "{}".format(TRIAL_NAME)
            OUT_PATH = "output/{}/".format(EXPER_NAME)
            if not os.path.exists(OUT_PATH):
                os.makedirs(OUT_PATH)
            evolve()
    log_progress("Done! Total time: {}".format(time.time()-start_time))
