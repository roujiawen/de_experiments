import json
from collections import OrderedDict
import time

import numpy as np


# set matplotlib backend
import matplotlib
from sys import platform
if platform == "darwin":

    matplotlib.use('TkAgg')
else:
    matplotlib.use('agg')

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import c_code as c_model

from weave_compile import N_GLOBAL_STATS

def counts2slices(counts):
    """Convert a list of counts to cummulative slices.
    Example: [5, 1, 3] ==> [slice(0, 5), slice(5, 6), slice(6, 9)]
    """
    cumu = [sum(counts[:i]) for i in range(len(counts)+1)]
    slices = [slice(cumu[i-1], cumu[i]) for i in range(1, len(cumu))]
    return slices

def fit_into(x, a, b):
    return max(min(x, b), a)

def gene2params(gene):
    params = {
        "Alignment Range": gene["Alignment Range"],
        "Pinned Cells": ["none"] * 3,
        "Interaction Force": gene["Interaction Force"],
        "Gradient Intensity": [gene["Gradient Intensity"], 0.0, 0.0],
        "Cell Ratio": [1.0, 0.0, 0.0],
        "Alignment Force": gene["Alignment Force"],
        "Noise Intensity": gene["Noise Intensity"],
        "Angular Inertia": gene["Angular Inertia"],
        "Adhesion": [
            [np.exp(np.tan(fit_into(gene["Adhesion"], -0.99, 0.99)*np.pi/2)), 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]
        ],
        "Gradient Direction": [gene["Gradient Direction"], 0.0, 0.0],
        "Cell Density": gene["Cell Density"],
        "Velocity": [gene["Velocity"], 0., 0.],
        "Interaction Range": gene["Interaction Range"]
    }
    return params

def gen_internal_params(uprm, general_params):
    """Format user-provided parameters into internal parameters accepted
    by the C++ program. """
    field_size = general_params["Field Size"]
    core_radius = general_params["Core Radius"]
    scale_factor = general_params["Scale Factor"]
    # FORMATTING INTERNAL PARAMS
    # Particle core radius
    r0_x_2 = core_radius * 2
    # Calculate actual field size (scaled)
    xlim = ylim = field_size / float(scale_factor)
    # Calculate number of particles
    max_num_particles = (np.sqrt(3)/6.) * (xlim*ylim) / (core_radius**2)
    nop = int(uprm['Cell Density'] * max_num_particles)
    # Calculate force and alignment radii
    r1 = uprm['Interaction Range'] * core_radius
    ra = uprm['Alignment Range'] * core_radius
    # Just copying
    iner_coef = uprm['Angular Inertia']
    f0 = uprm['Interaction Force']
    fa = uprm['Alignment Force']
    noise_coef = uprm['Noise Intensity']
    # Change type
    v0 = np.array(uprm["Velocity"])
    beta = np.array(uprm["Adhesion"])
    # Pinned = 0 (none) or 1 (fixed)
    pinned = np.array([0 if x == "none" else 1
                       for x in uprm["Pinned Cells"]]).astype(np.int32)
    # Convert ratio to number of cells in each species
    ratios = uprm["Cell Ratio"][:2]
    cumu_ratios = [sum(ratios[:i+1]) for i in range(len(ratios))] + [1.0]
    cumu_n_per_species = [0] + [int(nop*each) for each in cumu_ratios]
    n_per_species = np.array(
        [cumu_n_per_species[i] - cumu_n_per_species[i - 1]
         for i in range(1, len(cumu_n_per_species))]).astype(np.int32)
    # Gradient from polar to cartesian
    grad_x = np.array(
        [np.cos(d*np.pi) * i for d, i in zip(
            uprm["Gradient Direction"], uprm["Gradient Intensity"])])
    grad_y = np.array(
        [np.sin(d*np.pi) * i for d, i in zip(
            uprm["Gradient Direction"], uprm["Gradient Intensity"])])

    # Effective number of particles (excluding pinned)
    eff_nop = float(
        np.sum([n_per_species[i] for i in range(len(n_per_species))
                if pinned[i] == 0]))

    names = [
        'nop', 'eff_nop', 'xlim', 'ylim',
        'r0_x_2', 'r1', 'ra',  # radii
        'iner_coef', 'f0', 'fa', 'noise_coef',  # strengths
        'v0', 'pinned', 'n_per_species', 'beta', 'grad_x', 'grad_y'  # arr
        ]
    internal_params = OrderedDict([(x, locals()[x]) for x in names])
    return internal_params

class Model(object):
    def __init__(self, gene, significant_range, which_order_param, general_params, num_repeats):
        self.which_order_param = which_order_param
        self.num_repeats = num_repeats
        self.general_params = general_params
        self.repeats = [Repeat(gene,
            np.random.RandomState((_+int(time.time()*1e6)) % 4294967296),
            which_order_param, general_params, significant_range) for _ in range(num_repeats)]
        self.gene = gene

    @property
    def fitness(self):
        return np.mean([_.fitness for _ in self.repeats])

    def save(self, name):
        best_repeat = np.argmax([_.fitness for _ in self.repeats])
        data = {
            "gene": self.gene,
            "general_params": self.general_params,
            "fitness": self.fitness,
            "global_stats": self.repeats[best_repeat].global_stats[self.which_order_param,:].tolist()
        }
        with open("{}.json".format(name), "w") as outfile:
            json.dump(data, outfile)
        self.plot(save=name)

    def plot(self, save=None):
        # smaller file size
        alpha = 1 #0.5
        velocity_trace = 0.3
        scale_factor = self.repeats[0].scale_factor
        colors = ["blue", "red", "green"]
        figsize = (3.3*(self.num_repeats-1)+3, 3.53)
        dpi = 100
        plt.figure(figsize=figsize, dpi=dpi)
        for subplot_id, rep in enumerate(self.repeats):
            # Prepare for making plots
            x, y, dir_x, dir_y = rep.pos_x, rep.pos_y, rep.dir_x, rep.dir_y
            n_per_species = rep.internal_params["n_per_species"]
            # Set up plot parameters
            ax = plt.subplot(1, len(self.repeats), subplot_id+1, adjustable='box', aspect=1)
            dots = (figsize[1]*dpi)**2
            circle_size = dots/100. * 3.14 * (0.0615 * scale_factor)**2
            # Plot particle system
            for k, sli in enumerate(counts2slices(n_per_species)):
                if velocity_trace > 0:
                    segs = []
                    for j in range(sli.start, sli.stop):
                        segs.append(
                            ((x[j], y[j]),
                             (x[j]-dir_x[j]*velocity_trace,
                              y[j]-dir_y[j]*velocity_trace)))
                    ln_coll = LineCollection(segs, colors=colors[k],
                                             linewidths=1, alpha=alpha)
                    ax.add_collection(ln_coll)
                ax.scatter(x[sli], y[sli], s=circle_size, color=colors[k],
                           linewidths=0, alpha=alpha)
            # Set plot limits
            adjusted_limit = 10. / scale_factor
            plt.xlim([0, adjusted_limit])
            plt.ylim([0, adjusted_limit])
            plt.axis("off")
            plt.title("fitness={}".format(round(rep.fitness, 4)))


        plt.subplots_adjust(left=0, bottom=0, right=1, top=0.85, hspace=0, wspace=0.1)
        plt.suptitle("overal fitness={}".format(round(self.fitness, 4)))

        if save is None:
            plt.show()
        else:
            plt.savefig("{}.png".format(save))
        plt.close()


class Repeat(object):
    def __init__(self, gene, rand_state, which_order_param, general_params, significant_range):
        self.gene=gene
        self.rand_state = rand_state
        self.which_order_param = which_order_param
        self.general_params = general_params
        self.significant_range = significant_range

    def init(self):
        general_params = self.general_params
        gene = self.gene
        self.user_params = gene2params(gene)

        self.scale_factor = general_params["Scale Factor"]
        self.internal_params = gen_internal_params(self.user_params, general_params)
        self.init_particles_state()

        # Periodic boundary settings
        if general_params["Periodic Boundary"] == False:
            self.tick = self.fb_tick
        else:
            self.tick = self.pb_tick

        self.global_stats = np.zeros([N_GLOBAL_STATS, self.significant_range[1]])
        # Run simulation
        self.tick(self.significant_range[1])
        self.fitness = self.calculate_fitness()

    def init_particles_state(self):
        """Initialize a system of particles given params."""
        iprm = self.internal_params
        nop, xlim, ylim = iprm['nop'], iprm['xlim'], iprm['ylim']

        # Randomize position
        pos_x = self.rand_state.rand(nop) * xlim
        pos_y = self.rand_state.rand(nop) * ylim

        # Randomize velocity
        theta = self.rand_state.rand(nop) * 2 * np.pi
        dir_x = np.cos(theta)
        dir_y = np.sin(theta)

        self.pos_x, self.pos_y, self.dir_x, self.dir_y = (
            pos_x, pos_y, dir_x, dir_y)

    def fb_tick(self, steps):
        """Run the simulation for a given number of steps under fixed
        boundary conditions."""
        global_stats_slice = np.zeros(N_GLOBAL_STATS * steps)
        c_model.fb_tick(
            *self.internal_params.values()
            + [self.pos_x, self.pos_y, self.dir_x, self.dir_y,
               global_stats_slice, steps])
        self.global_stats = global_stats_slice.reshape(N_GLOBAL_STATS, steps)

    def pb_tick(self, steps):
        """Run the simulation for a given number of steps under periodic
        boundary conditions."""
        global_stats_slice = np.zeros(N_GLOBAL_STATS * steps)
        c_model.pb_tick(
            *self.internal_params.values()
            + [self.pos_x, self.pos_y, self.dir_x, self.dir_y,
               global_stats_slice, steps])
        self.global_stats = global_stats_slice.reshape(N_GLOBAL_STATS, steps)

    def fb_tick_dyn(self, steps):
        global_stats_slice = np.zeros(N_GLOBAL_STATS * steps)
        c_model.fb_tick(
            *self.internal_params.values()
            + [self.pos_x, self.pos_y, self.dir_x, self.dir_y,
               global_stats_slice, steps])
        self.global_stats = np.hstack(
            [self.global_stats,
             global_stats_slice.reshape(N_GLOBAL_STATS, steps)])

    def pb_tick_dyn(self, steps):
        """Run the simulation for a given number of steps under periodic
        boundary conditions."""
        global_stats_slice = np.zeros(N_GLOBAL_STATS * steps)
        c_model.pb_tick(
            *self.internal_params.values()
            + [self.pos_x, self.pos_y, self.dir_x, self.dir_y,
               global_stats_slice, steps])
        self.global_stats = np.hstack(
            [self.global_stats,
             global_stats_slice.reshape(N_GLOBAL_STATS, steps)])

    def plot(self, save=None):
        alpha = 0.5
        velocity_trace = 0.3
        scale_factor = self.scale_factor
        # Prepare for making plots
        x, y, dir_x, dir_y = self.pos_x, self.pos_y, self.dir_x, self.dir_y
        n_per_species = self.internal_params["n_per_species"]
        colors = ["blue", "red", "green"]
        # Set up plot parameters
        figsize = (5, 5)
        dpi = 100
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)
        dots = (figsize[0]*dpi)**2
        circle_size = dots/100. * 3.14 * (0.0615 * scale_factor)**2
        # Plot particle system
        for k, sli in enumerate(counts2slices(n_per_species)):
            if velocity_trace > 0:
                segs = []
                for j in range(sli.start, sli.stop):
                    segs.append(
                        ((x[j], y[j]),
                         (x[j]-dir_x[j]*velocity_trace,
                          y[j]-dir_y[j]*velocity_trace)))
                ln_coll = LineCollection(segs, colors=colors[k],
                                         linewidths=1, alpha=alpha)
                ax.add_collection(ln_coll)
            ax.scatter(x[sli], y[sli], s=circle_size, color=colors[k],
                       linewidths=0, alpha=alpha)
        # Set plot limits
        adjusted_limit = 10. / scale_factor
        plt.xlim([0, adjusted_limit])
        plt.ylim([0, adjusted_limit])

        plt.subplots_adjust(left=0, bottom=0, right=1, top=0.95)
        plt.title("fitness={}".format(round(self.fitness, 4)))
        plt.axis("off")
        if save is None:
            plt.show()
        else:
            plt.savefig("{}.png".format(save))
        plt.close()

    def calculate_fitness(self):
        # numerator = np.mean(self.global_stats[self.which_order_param[0],
        #                   self.significant_range[0]:self.significant_range[1]])
        # denominator = np.mean(self.global_stats[self.which_order_param[1],
        #                   self.significant_range[0]:self.significant_range[1]])
        data_segment = self.global_stats[self.which_order_param,
                          self.significant_range[0]:self.significant_range[1]]
        return np.mean(data_segment)

    def save(self, name):
        data = {
            "gene": self.gene,
            "general_params": self.general_params,
            "fitness": self.fitness,
            "global_stats": self.global_stats.tolist()
        }
        with open("{}.json".format(name), "w") as outfile:
            json.dump(data, outfile)
        self.plot(save=name)


def main():
    # SIGNIFICANT_RANGE = [50, 60]
    # WHICH_ORDER_PARAM = 1
    # gene = {
    #     "Gradient Intensity": 0.,
    #     "Cell Density": 0.5,
    #     "Angular Inertia": 1.,
    #     "Alignment Force": 1.,
    #     "Gradient Direction": 0.,
    #     "Alignment Range": 10.,
    #     "Adhesion": -1.,
    #     "Interaction Force": 1.0,
    #     "Noise Intensity": 0.2,
    #     "Velocity": 0.03,
    #     "Interaction Range": 30,
    # }
    #
    # steps = SIGNIFICANT_RANGE[1]
    # test_model = Repeat(gene)
    # # Run model
    # test_model.tick(steps)
    # test_model.save("test_model")
    import sys
    genepath = sys.argv[1]
    with open(genepath, 'r') as infile:
        model_data = json.load(infile)
    model = Repeat()
    model.init(model_data["gene"])
    steps = int(sys.argv[2])
    model.tick(steps)
    model.plot()


if __name__ == "__main__":
    main()

# gene_order = ["Gradient Intensity", "Cell Density", "Angular Inertia",
# "Alignment Force", "Gradient Direction", "Alignment Range", "Adhesion",
# "Interaction Force", "Noise Intensity", "Velocity", "Interaction Range"]
# gene_code = {__:_ for _, __ in enumerate(gene_order)}
