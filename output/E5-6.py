"""
 - Basing on E4-1
 - Hyperparameter search:
     0      1    2   3
 K = 0.55, 1.2, 1.6, 2
     4     5    6    7
Cr = 0.3, 0.6, 0.7, 0.9
     8          9             10              11
   (0.3, 0.3)  (1.2, 0.3)    (1.2,  0.7)    (1.2, 0.9)
"""

# --------- Differential evolution parameters ---------
STARTING_REP_ID = 0
WHICH_ORDER_PARAM = 0 # 0: angular momentum, 1: alignment, 2: clustering,
# 3: CM_x, 4: CM_y, 5: radial distance (CF), 6: pw distance, 7: nn distance,
# 8: group migration, 9: radial distance (CM)
# Boundary conditions of the simulation arena
PERIODIC_BOUNDARY = True



DENSITIES = []  # the set of densities to run experiments with; empty -> var

NUM_GENERATION = 300
POPULATION_SIZE = 100
SCALING_PARAM = 0.8 # usually from (0,2] best 0.3
CROSSOVER_RATE = 0.7 # usually from (0,1) best 0.8
NUM_REPEATS = 5  # number of repeated runs for the same gene
SIGNIFICANT_RANGE = [1500, 2000]  # range of steps to average over the fitness

MAX_OR_MIN = "MAX"  # choose from {"MAX", "MIN"}


# --------- Feasible model parameter ranges in evolution ---------

PARAM_LIMITS = {
    "Gradient Intensity": [0.0, 0.0],
    "Cell Density": [0.01, 0.1], # matters only when DENSITIES set to empty
    "Angular Inertia": [0.0, 5.0],
    "Alignment Force": [0.0, 5.0],
    "Gradient Direction": [0.0, 2.0],
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
