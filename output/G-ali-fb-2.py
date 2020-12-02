"""
Dec-02-2020
 - Copied from F-<same-suffix>; However, set gradient to zero!!
 - key: 500 generations, max_fitness, rand1bin
 Subexperiments 1 & 2 should be the same; sending to three clusters.
 If we need 10 repeated experiments, I'll try sending subexperiment 3 to all four clusters (+Crosby).

"""

# --------- Differential evolution parameters ---------
STARTING_REP_ID = 0
WHICH_ORDER_PARAM = 1 # 0: angular momentum, 1: alignment, 2: clustering,
# 3: CM_x, 4: CM_y, 5: radial distance (CF), 6: pw distance, 7: nn distance,
# 8: group migration, 9: radial distance (CM)
# Boundary conditions of the simulation arena
PERIODIC_BOUNDARY = False



DENSITIES = []  # the set of densities to run experiments with; empty -> var

NUM_GENERATION = 500
POPULATION_SIZE = 100
SCALING_PARAM = 0.8  # usually from (0,2] best 0.3
CROSSOVER_RATE = 0.9  # usually from (0,1) best 0.8
NUM_REPEATS = 5  # number of repeated runs for the same gene
SIGNIFICANT_RANGE = [1500, 2000]  # range of steps to average over the fitness

MAX_OR_MIN = "MAX"  # choose from {"MAX", "MIN"}
FITNESS_AGGREGATE = "max_fitness" # choose from {"max_fitness", "mean_fitness"}
DE_STRATEGY = "rand1bin" # choose from {"rand1bin", "best1bin"}


# --------- Feasible model parameter ranges in evolution ---------

PARAM_LIMITS = {
    "Gradient Intensity": [0.0, 0.0],
    "Cell Density": [0.01, 0.1], # matters only when DENSITIES set to empty
    "Angular Inertia": [0.0, 5.0],
    "Alignment Force": [0.0, 5.0],
    "Gradient Direction": [0.0, 0.0],
    "Alignment Range": [2.000000001, 30.0],
    # -1 maps to beta=0 (strongest repulsion)
    # and 1 maps to beta=+inf (strongest attraction)
    "Adhesion": [-1., 1.],
    "Interaction Force": [0.0, 5.0],
    "Noise Intensity": [0.0, 1.0],
    "Velocity": [0.005, 0.2],
    "Interaction Range": [2.000000001, 30.0]
}

# --------- General model parameters ---------


# How much the simulation is zoomed in
SCALE_FACTOR = 1.0
# Radius of the particles's hard core in the simulation
CORE_RADIUS = 0.1
# Linear size of the arena of the simulation
FIELD_SIZE = 10.0

GENERAL_PARAMS = {
    "Periodic Boundary": PERIODIC_BOUNDARY,
    "Scale Factor": SCALE_FACTOR,
    "Core Radius": CORE_RADIUS,
    "Field Size": FIELD_SIZE
}
