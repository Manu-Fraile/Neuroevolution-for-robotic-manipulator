"""
Main algorithm
"""

from __future__ import print_function

from fitness_functions import FitnessFunctions
from random import random

import os
import pickle

import neat
import visualize

#time_const=1

fitnessFunctions = FitnessFunctions()
trajectory_paper=[[2.25,1.1,0.25],
                    [0.9,1.5,0.25],
                    [-0.85,1.14,2.22],
                    [-1.8,1.25,1.17],
                    [1.8,1.25,1.17],
                    [-1.25,-1.1,0.25],
                    [-2.25,-1.48,0.25],
                    [0.45,-1.14,2.22],
                    [0.8,-1.25,2.35],
                    [0.8,-1.25,-1.35]]

# Use the RN network phenotype
def eval_genome(genome, config):
    ###ATTENTION: TO ADD CHECK IF JOINT CAN DO THE ROTATION

    ###ATTENTION: think about that some joint can go from -600 to 600. That means that they can't rotate???

    net = neat.nn.RecurrentNetwork.create(genome, config)

    trajectory_points=trajectory_paper
    outputs = []
    for point in trajectory_points:
        outputs.append(net.activate(point))

    total_rotation = fitnessFunctions.evaluate_rotations(outputs)
    total_energy = fitnessFunctions.evaluate_energy(outputs)
    total_operation_time = fitnessFunctions.evaluate_operation_time(outputs)

    fitness = 1/total_rotation+1/total_energy+1/total_operation_time
    return fitness

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # Load the config file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-rnn')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    ########################################
    num_generation=20
    if 0:
        winner = pop.run(eval_genomes, num_generation)
    else:
        pe = neat.ParallelEvaluator(4, eval_genome)
        winner = pop.run(pe.evaluate,num_generation)

    # Save the winner.
    with open('results/winner-rnn', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    print(stats);
    if(0):

        visualize.plot_stats(stats, ylog=True, view=False, filename="results/rnn-fitness.svg")
        visualize.plot_species(stats, view=False, filename="results/rnn-speciation.svg")

        node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}

        visualize.draw_net(config, winner, view=False, node_names=node_names,
                           filename="results/winner-rnn.gv")
        visualize.draw_net(config, winner, view=False, node_names=node_names,
                           filename="results/winner-rnn-enabled.gv", show_disabled=False)
        visualize.draw_net(config, winner, view=False, node_names=node_names,
                           filename="results/winner-rnn-enabled-pruned.gv", show_disabled=False, prune_unused=True)


if __name__ == '__main__':
    run()
