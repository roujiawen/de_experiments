"""
Apr-12-2021

3 parameter flexible; first minimize then maximize

 - Copied from O-ali-pb-1: flexible density, noise, alignment range
 - Added INITIAL_POP_PATH
 - Within Vicsek PT param subspace: minimize order param value
 - MAX_OR_MIN = MAX
 - NUM_GENERATION = 100, POPULATION_SIZE = 50, NUM_REPEATS = 3, SIGNIFICANT_RANGE = [1500, 2000]

 Subexperiments (e.g. -1, -2, -3 ...) should each be different; num_restarts = 3

 Subexperiments 3 uses the last generation of 1 as starting point.
 Subexperiments 4 uses the last generation of 2 as starting point.

"""

# --------- Differential evolution parameters ---------
STARTING_REP_ID = 0
WHICH_ORDER_PARAM = 1 # 0: angular momentum, 1: alignment, 2: clustering,
# 3: CM_x, 4: CM_y, 5: radial distance (CF), 6: pw distance, 7: nn distance,
# 8: group migration, 9: radial distance (CM)
# Boundary conditions of the simulation arena
PERIODIC_BOUNDARY = True

INITIAL_POP_PATH = "initial_pop/O-ali-pb-3/genes.json"

DENSITIES = []  # the set of densities to run experiments with; empty -> var

NUM_GENERATION = 100
POPULATION_SIZE = 50
SCALING_PARAM = 0.8  # usually from (0,2] best 0.3
CROSSOVER_RATE = 0.9  # usually from (0,1) best 0.8
NUM_REPEATS = 3  # number of repeated runs for the same gene
SIGNIFICANT_RANGE = [1500, 2000]  # range of steps to average over the fitness

MAX_OR_MIN = "MAX"  # choose from {"MAX", "MIN", ("CUSTOM", obj_value)}
FITNESS_AGGREGATE = "mean_fitness" # choose from {"max_fitness", "mean_fitness"}
DE_STRATEGY = "rand1bin" # choose from {"rand1bin", "best1bin"}


# --------- Feasible model parameter ranges in evolution ---------


PARAM_LIMITS = {
    "Gradient Intensity": [0.0, 0.0],
    "Cell Density": [0.01, 0.3466], # <=
    "Angular Inertia": [0.01, 0.01],# in Vicsek the average of neighbor directions include the agent itself
    "Alignment Force": [1.0, 1.0],
    "Gradient Direction": [0.0, 0.0],
    "Alignment Range": [2.000000001, 10.0], # <=
    "Adhesion": [0.0, 0.0], #
    "Interaction Force": [0.0, 0.0],
    "Noise Intensity": [0.0, 0.7961], # <=
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
