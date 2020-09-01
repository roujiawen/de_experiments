
ABBR_DICT = {"Gradient Intensity": "GrdInt",
    "Cell Density": "Dens",
    "Angular Inertia": "Inert",
    "Alignment Force": "AForce",
    "Gradient Direction": "GrdDir",
    "Alignment Range": "ARange",
    "Adhesion": "Adhes",
    "Interaction Force": "IForce",
    "Noise Intensity": "Noise",
    "Velocity": "Veloc",
    "Interaction Range": "IRange"}

ORDER_PARAMS = ["ang", "ali", "clu"]
BOUND_CONDS = ["pb", "fb"]

PARAM_ORDER_DICT = {"Gradient Intensity": 0,
    "Cell Density": 10,
    "Angular Inertia": 1,
    "Alignment Force": 2,
    "Gradient Direction": 3,
    "Alignment Range": 4,
    "Adhesion": 5,
    "Interaction Force": 6,
    "Noise Intensity": 7,
    "Velocity": 8,
    "Interaction Range": 9}

PARAM_NAMES = ABBR_DICT.keys()


#WHICH ORDER PARAM
# 0: angular momentum, 1: alignment, 2: clustering,
# 3: CM_x, 4: CM_y, 5: radial distance (CF), 6: pw distance, 7: nn distance,
# 8: group migration, 9: radial distance (CM)
