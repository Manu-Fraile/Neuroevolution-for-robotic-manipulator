"""
Main algorithm
"""

from __future__ import print_function

import sys

from fitness_functions import FitnessFunctions
from random import random
import matplotlib.pyplot as plt
from statistics import mean
import math
import os
import pickle
import pandas as pd
import numpy as np

import neat
import visualize

type="rnn"
#time_const=1

fitnessFunctions = FitnessFunctions()
trajectory_paper = [[2.25, 1.1, 0.25],#
                    [0.9, 1.5, 0.25],
                    [-0.85, 1.14, 2.22],
                    [-1.8, 1.25, 1.17],
                    [1.8, 1.25, 1.17],
                    [-1.25, -1.1, 0.25],
                    [-2.25, -1.48, 0.25],
                    [0.45, -1.14, 2.22],
                    [0.8, -1.25, 2.35],
                    [0.8, -1.25, -1.35]]
fitnesses={"total":[],"rotation":[],"energy":[],"time":[],"accuracy":[]}

def getMultiplierForNormalization(_min, _max):
    return (_max - _min)/1 + (_min)

# Use the RN network phenotype
def eval_genome(genome, config, verbose=False):
    ###ATTENTION: TO ADD CHECK IF JOINT CAN DO THE ROTATION

    ###ATTENTION: think about that some joint can go from -600 to 600. That means that they can't rotate???
    if(type=="rnn"):
        net = neat.nn.RecurrentNetwork.create(genome, config)
    else:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
    data={}

    trajectory_points = trajectory_paper
    outputs = []
    rescaling_factors=np.array([
        getMultiplierForNormalization(math.radians(-180), math.radians(180)),
        getMultiplierForNormalization(math.radians(-70), math.radians(70)),
        getMultiplierForNormalization(math.radians(-28), math.radians(105)),
        getMultiplierForNormalization(math.radians(-300), math.radians(300)),
        getMultiplierForNormalization(math.radians(-120), math.radians(120)),
        getMultiplierForNormalization(math.radians(-300), math.radians(300))
    ])
    #(c) * (z - y) / (1) + y

    i=0
    for i in range(0,len(trajectory_points)):
        if(type=="rnn"):
            output=net.activate(trajectory_points[i])
        else:
            output=net.activate(trajectory_points[i]+trajectory_points[(i-1+10)%10])
        # RESCALING
        output=(np.array(output)*rescaling_factors).tolist()


        #for j in range(0,len(output)):
        #    output[j]=output[j]*2*math.pi
        outputs.append(output)
        #print(min(outputs))
        #print(max(outputs))
    if verbose:
        np_outputs=np.array(outputs)
        for i in range(0,6):
            data["outputs_"+str(i)]=np_outputs[:,i].tolist()
        total_rotation, data["rotation"] = fitnessFunctions.evaluate_rotations(outputs, verbose)
        total_energy, data["energy"] = fitnessFunctions.evaluate_energy(outputs, verbose)
        total_operation_time, data["operation"] = fitnessFunctions.evaluate_operation_time(outputs, verbose)
        total_accuracy, data["accuracy"] = fitnessFunctions.evaluate_position_accuracy(outputs, trajectory_points, verbose)
        data["total_rotation"]=total_rotation
        data["total_energy"]=total_energy
        data["total_operation_time"]=total_operation_time
        data["total_accuracy"]=total_accuracy
    else:
        total_rotation = fitnessFunctions.evaluate_rotations(outputs)
        total_energy = fitnessFunctions.evaluate_energy(outputs)
        total_operation_time = fitnessFunctions.evaluate_operation_time(outputs)
        total_accuracy = fitnessFunctions.evaluate_position_accuracy(outputs, trajectory_points)

    fitness = -(total_accuracy)#ACCURACY OPTIMAL
    #fitness = -(total_accuracy+20/100*total_operation_time)#TIME OPTIMAL
    #fitness = -(total_accuracy+20*total_energy)#ENERGY OPTIMAL
    #fitness = -(total_accuracy+20/100*total_rotation)#MINIMUM ROTATION
    #fitness = -(total_accuracy+5*total_energy+10*total_operation_time+5*total_rotation)#COMBINED CONTROL

    fitnesses["total"].append(-fitness)
    fitnesses["rotation"].append(total_rotation)
    fitnesses["energy"].append(total_energy)
    fitnesses["time"].append(total_operation_time)
    fitnesses["accuracy"].append(total_accuracy)
    if verbose:
        data["fitness"]=total_accuracy
        df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in data.items() ]))
        df.to_csv('tableResults.csv', index=False)
    return fitness


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # Load the config file
    local_dir = os.path.dirname(__file__)

    if(type=="rnn"):
        config_path = os.path.join(local_dir, 'config-ctrnn')
    else:
        config_path = os.path.join(local_dir, 'config-feed')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    ########################################
    num_generation=200
    pop_size=250#Remember that is defined in config file
    if 1:
        winner = pop.run(eval_genomes, num_generation)
    else:
        pe = neat.ParallelEvaluator(4, eval_genome)
        winner = pop.run(pe.evaluate, num_generation)

    # Save the winner.
    with open('results/winner-rnn', 'wb') as f:
        pickle.dump(winner, f)

    eval_genome(winner, config, True)
    print(stats);


    #plt.figure("NEAT (Population's average/std. dev and best fitness)")
    plt.figure("NEAT fitnesses")# (Population's average/std. dev and best fitness)")
    avg_fitnesses={"total":[],"rotation":[],"energy":[],"time":[],"accuracy":[]}
    for i in range(0, num_generation):
        avg_fitnesses["total"].append(np.mean(fitnesses["total"][i*pop_size:i*pop_size + pop_size-1]))
        avg_fitnesses["time"].append(np.mean(fitnesses["time"][i*pop_size:i*pop_size + pop_size-1]))
        avg_fitnesses["rotation"].append(np.mean(fitnesses["rotation"][i*pop_size:i*pop_size + pop_size-1]))
        avg_fitnesses["energy"].append(np.mean(fitnesses["energy"][i*pop_size:i*pop_size + pop_size-1]))
        avg_fitnesses["accuracy"].append(np.mean(fitnesses["accuracy"][i*pop_size:i*pop_size + pop_size-1]))
    #print(avg_fitnesses["accuracy"])
    #print(avg_fitnesses)
    #plt.plot([i for i in range(0, len(avg_fitnesses["total"]))], avg_fitnesses["total"], 'b-', label="fitness")
    plt.plot([i for i in range(0, len(avg_fitnesses["energy"]))], avg_fitnesses["energy"], color="#000000", label="energy", linestyle=':')
    #plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    #plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    #plt.plot(generation, best_fitness, 'r-', label="best")

    #plt.title("Population's average/std. dev and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")

    plt.savefig("energy.png")
    view=True
    if view:
        plt.show()
        #plt.close()
        fig = None

    plt.plot([i for i in range(0, len(avg_fitnesses["accuracy"]))], avg_fitnesses["accuracy"], color="#000000", linestyle=":", label="accuracy")
    plt.plot([i for i in range(0, len(avg_fitnesses["time"]))], avg_fitnesses["time"], color="#000000", linestyle="-", label="time")
    plt.plot([i for i in range(0, len(avg_fitnesses["rotation"]))], avg_fitnesses["rotation"], color="#000000",linestyle="--", label="rotation")
    #plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    #plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    #plt.plot(generation, best_fitness, 'r-', label="best")

    #plt.title("Population's average/std. dev and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")

    plt.savefig("accuracy_time_rotation.png")
    view=True
    if view:
        plt.show()
        #plt.close()
        fig = None


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
    print(sys.argv)
    if(len(sys.argv)>1):
        type=sys.argv[1]
    run()
