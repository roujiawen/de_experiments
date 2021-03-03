"""
Mar-3-2021
 - Within Vicsek PT param subspace: first minimize then maximize
 - Note "MAX" and "mean_fitness"
 - Copied from L-ali-pb-1; PARAM_LIMITS copied from V_PT2.py

 Subexperiments 1 & 2 should be the same; sending to three clusters.

 Subexperiments 3 & 4 should be the same (starting population should be the
 last generation of 1 & 2); sending to three clusters.

 - Added INITIAL_POP_PATH

"""

# --------- Differential evolution parameters ---------
STARTING_REP_ID = 0
WHICH_ORDER_PARAM = 1 # 0: angular momentum, 1: alignment, 2: clustering,
# 3: CM_x, 4: CM_y, 5: radial distance (CF), 6: pw distance, 7: nn distance,
# 8: group migration, 9: radial distance (CM)
# Boundary conditions of the simulation arena
PERIODIC_BOUNDARY = True

INITIAL_POP_PATH = "initial_pop/M-ali-pb-4/genes.json"

DENSITIES = []  # the set of densities to run experiments with; empty -> var

NUM_GENERATION = 100
POPULATION_SIZE = 100
SCALING_PARAM = 0.8  # usually from (0,2] best 0.3
CROSSOVER_RATE = 0.9  # usually from (0,1) best 0.8
NUM_REPEATS = 3  # number of repeated runs for the same gene
SIGNIFICANT_RANGE = [750, 1000]  # range of steps to average over the fitness

MAX_OR_MIN = "MAX"  # choose from {"MAX", "MIN"}
FITNESS_AGGREGATE = "mean_fitness" # choose from {"max_fitness", "mean_fitness"}
DE_STRATEGY = "rand1bin" # choose from {"rand1bin", "best1bin"}


# --------- Feasible model parameter ranges in evolution ---------


PARAM_LIMITS = {
    "Gradient Intensity": [0.0, 0.0],
    "Cell Density": [0.01, 0.3466],
    "Angular Inertia": [0.01, 0.01],# in Vicsek the average of neighbor directions include the agent itself
    "Alignment Force": [1.0, 1.0],
    "Gradient Direction": [0.0, 0.0],
    "Alignment Range": [10.0, 10.0], #
    "Adhesion": [0.0, 0.0], #
    "Interaction Force": [0.0, 0.0],
    "Noise Intensity": [0.0, 0.7961], #
    "Velocity": [0.03, 0.03], #
    "Interaction Range": [2.0, 2.0] #
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
