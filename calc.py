import sys
from os import listdir
import json
import numpy as np


indiv_name = sys.argv[1]
exper_name = indiv_name.split("/")[0]

from importlib import import_module
SIGNIFICANT_RANGE = getattr(import_module("output.{}".format(exper_name)), "SIGNIFICANT_RANGE")

path = "output/" + indiv_name + ".json"

with open(path, "r") as input_file:
    json_obj = json.load(input_file)

[start, end] = SIGNIFICANT_RANGE

ang_mom = np.mean(json_obj["global_stats"][0][start:end])
pw_dist = np.mean(json_obj["global_stats"][1][start:end])
print "Angular Momentum:\t", ang_mom
print "Pairwise Distance:\t", pw_dist
print "Ratio:\t\t\t", ang_mom/pw_dist
