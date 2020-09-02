import sys
from os import listdir
exper_name = sys.argv[1]
if len(sys.argv) <= 2:
    take_num_gen = 20
else:
    take_num_gen = int(sys.argv[2])

path = "output/" + exper_name

all_names = listdir(path)
generations = list({int(name.split("_")[0][3:]) for name in all_names})
generations.sort(reverse=True)
print(generations[:take_num_gen])
