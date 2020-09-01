import os
import random
import json
from copy import deepcopy

import numpy as np
from model import Model

from multiprocess_setup import *
#assert


from diff_evo.E1 import (PARAM_LIMITS, POPULATION_SIZE, NUM_GENERATION,
                        SCALING_PARAM, CROSSOVER_RATE, TRIAL_NAME, DENSITIES,
                        MAX_OR_MIN, NUM_REPEATS, STARTING_REP_ID)


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


def evolve():
    # Initialize population
    population = []
    models = []

    # Random genes for initial population
    for indiv_id in range(POPULATION_SIZE):
        gene = random_gene()
        model = Model(gene)
        models.append(model)

    for rep_id in range(NUM_REPEATS):
        repeats = [_.repeats[rep_id] for _ in models]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for repeat, indiv_id in executor.map(model_wrapper, zip(repeats, range(POPULATION_SIZE))):
                models[indiv_id].repeats[rep_id] = repeat

    for indiv_id in range(POPULATION_SIZE):
        model = models[indiv_id]
        model.save(OUT_PATH+"Gen0_{}".format(indiv_id))
        population.append(model)

    # Evolve
    for gen_id in range(1, NUM_GENERATION+1):
        models = []
        repeats = []
        for indiv_id in range(POPULATION_SIZE):
            this_gene = population[indiv_id].gene
            # Mutate
            base, dif1, dif2 = np.random.choice(
                [m.gene for m in population if m.gene != this_gene], 3)
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
            model = Model(trial)
            models.append(model)

        for rep_id in range(NUM_REPEATS):
            repeats = [_.repeats[rep_id] for _ in models]
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for repeat, indiv_id in executor.map(model_wrapper, zip(repeats, range(POPULATION_SIZE))):
                    models[indiv_id].repeats[rep_id] = repeat

        for indiv_id in range(POPULATION_SIZE):
            model = models[indiv_id]
            # Selection
            if MAX_OR_MIN=="MAX":
                if model.fitness >= population[indiv_id].fitness:
                    population[indiv_id] = model
                    model.save(OUT_PATH+"Gen{}_{}".format(gen_id, indiv_id))
            else:
                if model.fitness <= population[indiv_id].fitness:
                    population[indiv_id] = model
                    model.save(OUT_PATH+"Gen{}_{}".format(gen_id, indiv_id))


if __name__ == "__main__":
    import time
    start_time = time.time()
    if len(DENSITIES) == 0:
        EXPER_NAME = TRIAL_NAME
        OUT_PATH = "diff_evo/{}/".format(EXPER_NAME)
        if not os.path.exists(OUT_PATH):
            os.makedirs(OUT_PATH)
        evolve()
    elif DENSITIES[0] is None:
        for i, _ in enumerate(DENSITIES):
            EXPER_NAME = "{}_{}".format(TRIAL_NAME, str(i))
            OUT_PATH = "diff_evo/{}/".format(EXPER_NAME)
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
            OUT_PATH = "diff_evo/{}/".format(EXPER_NAME)
            if not os.path.exists(OUT_PATH):
                os.makedirs(OUT_PATH)
            evolve()
    print("total run time: {}".format(time.time()-start_time))
