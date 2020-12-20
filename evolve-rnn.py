"""
Main algorithm
"""

from __future__ import print_function

import sys

from fitness_functions import FitnessFunctions
from random import random, choice
import matplotlib.pyplot as plt
from statistics import mean
import math
import os
import pickle
import pandas as pd
import numpy as np

import neat
import visualize

#################### PARAMETERS   ####################
type_optimization="accuracy"
t=["accuracy", "timeO", "rotationO", "energyO"]
type_optimization=t[0]

USE_FIXED_TRAJECTORY=True
N_POINTS_TRAJECTORY_TRAINING=30 #Number of point for evaluate the quality of each network during training
num_generation=600
pop_size=200 #Remember that is defined in config file

type="rnn"  #rnn or feed

#################### PARAMETERS   ####################
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

points=[]

if(USE_FIXED_TRAJECTORY==False):
    with open('points.csv', 'rb') as f:
        points = pickle.load(f)

def getMultiplierForNormalization(_min, _max):
    return (_max - _min)/1 + (_min)

# Use the RN network phenotype
def eval_genome(genome, config, verbose=False, fixed=-1, points_to_use=[]):

    if(type=="rnn"):
        net = neat.nn.RecurrentNetwork.create(genome, config)
    else:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
    data={}

    fixed_data=USE_FIXED_TRAJECTORY
    if(fixed>-1):
        fixed_data=fixed==0 #fixed==0 used fix trajectory, any other number use the not fixed

    trajectory_points=[]

    if(fixed_data):
        trajectory_points = trajectory_paper
    else:
        trajectory_points = points_to_use

        #trajectory_points=trajectory_points+trajectory_paper

    outputs = []
    rescaling_factors=np.array([
        getMultiplierForNormalization(math.radians(-180), math.radians(180)),
        getMultiplierForNormalization(math.radians(-70), math.radians(70)),
        getMultiplierForNormalization(math.radians(-28), math.radians(105)),
        getMultiplierForNormalization(math.radians(-300), math.radians(300)),
        getMultiplierForNormalization(math.radians(-120), math.radians(120)),
        getMultiplierForNormalization(math.radians(-300), math.radians(300))
    ])

    trajectory_len=len(trajectory_points)

    for i in range(0,trajectory_len):
        if(type=="rnn"):
            output=net.activate(trajectory_points[i])
        else:
            #output=net.activate(trajectory_points[i]+trajectory_points[(i-1+trajectory_len)%trajectory_len])
            output=net.activate(trajectory_points[i])

        # RESCALING
        output=(np.array(output)*rescaling_factors).tolist()
        outputs.append(output)

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

    if(type_optimization=="accuracy"):
        fitness = -(total_accuracy)#ACCURACY OPTIMAL
    elif(type_optimization=="timeO"):
        fitness = -(total_accuracy+20/150*total_operation_time)#TIME OPTIMAL
    elif(type_optimization=="rotationO"):
        fitness = -(total_accuracy+20/4000*total_rotation)#MINIMUM ROTATION
    elif(type_optimization=="energyO"):
        fitness = -(total_accuracy+20/20000*total_energy)#ENERGY OPTIMAL
    else:
        fitness = -(total_accuracy+5*total_energy+10*total_operation_time+5*total_rotation)#COMBINED CONTROL

    fitnesses["total"].append(-fitness)
    fitnesses["rotation"].append(total_rotation)
    fitnesses["energy"].append(total_energy)
    fitnesses["time"].append(total_operation_time)
    fitnesses["accuracy"].append(total_accuracy)
    if verbose:
        data["fitness"]=fitness
        df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in data.items() ]))
        df.to_csv('./result_algorithm/tableResults_'+type_optimization+'.csv', index=False)
    return fitness


def eval_genomes(genomes, config):
    points_to_use=[]
    if(USE_FIXED_TRAJECTORY==False):
        for i in range(0, N_POINTS_TRAJECTORY_TRAINING):
            points_to_use.append(choice(points)[1])
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config, points_to_use=points_to_use)


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

    if 1:
        winner = pop.run(eval_genomes, num_generation)
    else:
        pe = neat.ParallelEvaluator(4, eval_genome)
        winner = pop.run(pe.evaluate, num_generation)

    # Save the winner.
    with open('results/winner-rnn', 'wb') as f:
        pickle.dump(winner, f)

    eval_genome(winner, config, True, fixed=0)
    print(stats);


    #plt.figure("NEAT (Population's average/std. dev and best fitness)")
    #plt.rcParams["figure.figsize"] = [6,4]
    plt.rcParams.update({'font.size': 15})
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
    plt.plot([i for i in range(0, len(avg_fitnesses["energy"]))], avg_fitnesses["energy"], color="#000000", label="energy", linestyle='-')
    #plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    #plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    #plt.plot(generation, best_fitness, 'r-', label="best")

    #plt.title("Population's average/std. dev and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness", labelpad=0)
    plt.grid()
    plt.legend(loc="best")



    plt.savefig("energy_"+type_optimization+".png")
    view=True
    if view:
        plt.show()
        #plt.close()
        fig = None

    plt.plot([i for i in range(0, len(avg_fitnesses["accuracy"]))], avg_fitnesses["accuracy"], color="#808080", linestyle="-", label="accuracy",lw=0.9)
    plt.plot([i for i in range(0, len(avg_fitnesses["time"]))], avg_fitnesses["time"], color="#D3D3D3", linestyle="-", label="time",lw=0.9)
    plt.plot([i for i in range(0, len(avg_fitnesses["rotation"]))], avg_fitnesses["rotation"], color="#000000",linestyle="-", label="rotation",lw=0.9)

    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")

    plt.savefig("accuracy_time_rotation_"+type_optimization+".png")
    view=True
    if view:
        plt.show()
        fig = None


    if(1):
        visualize.plot_stats(stats, ylog=True, view=False, filename="results/rnn-fitness.svg")
        visualize.plot_species(stats, view=False, filename="results/rnn-speciation.svg")

        node_names = {0: 'omega', 1: 'theta', 2: 'psi', 3: 'fi', 4: 'ro', 5: 'epsilon', -3: 'x', -2: 'y', -1: 'z'}

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
    print(type)
    run()
