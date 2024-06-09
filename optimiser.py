import json
import math
import random
import time

import swarming_simulator
import numpy as np
import matplotlib.pyplot as plt
from instability_condition import criterion
from sympy import evalf
import multiprocessing

parameter_ranges = {
    "sigma": (0, 5.555e-6),
    "scale": (0, 2),
    "rho_max": (0, 28e8),
    "cushion": (0, 2e8),
    "beta_a": (0, 1.111e-7),
    "beta_r": (-1.111e-7, 1e-9),
    "alpha_a": (0, 15e8),
    "alpha_r": (0, 15e8),
    "D_a": (0, 1.111e-8),
    "D_r": (0, 1.111e-7),
    "gamma_a": (0, 1),
    "gamma_r": (0, 1e-1),
    "s_a": (0, 1e4),
    "s_r": (0, 1e3),
    "rho0": (1e6, 10e9),
}
individual_index=0
def create_individual(_):
    global individual_index
    criterion_satisfied = False
    #time it
    t = time.time()
    while not criterion_satisfied:
        individual = []
        for param, (min_val, max_val) in parameter_ranges.items():
                individual.append(random.uniform(min_val, max_val))
        #check if the individual satisfies the criterion
        solution = criterion(individual[0], individual[4], individual[5], individual[6], individual[7], individual[8], individual[9], individual[10], individual[11], individual[12], individual[13])
        #print(solution)
        if solution:
            criterion_satisfied = True
            #pick a random solution, then use it start and end as the range for the last parameter
            picked_solution = random.choice(solution)
            start = picked_solution.start
            end = picked_solution.end
            if end.is_infinite:
                end = 10e9
            individual[-1] = float(random.uniform(start, end).evalf())
    print(individual_index, " : ", individual)
    individual_index+=1
    endt = time.time()
    print("Time taken to generate individual: ", endt-t)
    return individual

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            min_val, max_val = parameter_ranges[list(parameter_ranges.keys())[i]]
            individual[i] = random.uniform(min_val, max_val)
    return individual


def objectives(gen, individual, i_):
    print("gen: " + str(gen) + ",ind: " + str(i_))
    parameter_dir = f"parameters_swarming_optimised{i_}.json"
    #save the individual inside of "paramters_swarming_optimisation.json"
    with open(parameter_dir, "w") as f:
        individual_dictionary = {}
        for i, (param, _) in enumerate(parameter_ranges.items()):
            individual_dictionary[param] = individual[i]
        json.dump(individual_dictionary, f)
    #run the swarming simulator
    #have a try catch block to catch any errors. in that case, time is set to infinity and clustering is set to 0
    print("Running swarming simulator: ", gen, i_, parameter_dir)
    clustering, time = swarming_simulator.run([gen, i_, parameter_dir])

    max_time = 500000*0.01
    #normalize time
    time = time/max_time
    #return the clustering and the time
    return [clustering, time]

def non_dominated_sorting_algorithm(values1, values2):
    S = [[] for _ in range(len(values1))]
    front = [[]]
    n = [0] * len(values1)
    rank = [0] * len(values1)

    for p in range(len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(len(values1)):
            if (values1[p] < values1[q] and values2[p] < values2[q]) or (values1[p] <= values1[q] and values2[p] < values2[q]) or (values1[p] < values1[q] and values2[p] <= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] < values1[p] and values2[q] < values2[p]) or (values1[q] <= values1[p] and values2[q] < values2[p]) or (values1[q] < values1[p] and values2[q] <= values2[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
    i = 0
    while front[i]:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i += 1
        front.append(Q)
    front.pop()
    return front

def index_locator(a, list):
    for i in range(len(list)):
        if list[i] == a:
            return i
    return -1

def sort_by_values(list1, values):
    sorted_list = []
    while len(sorted_list) != len(list1):
        if index_locator(min(values), values) in list1:
            sorted_list.append(index_locator(min(values), values))
        values[index_locator(min(values), values)] = math.inf
    return sorted_list

def crowding_distance(values1, values2, front):
    distance = [0] * len(front)
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = distance[-1] = float('inf')
    m1 = max(values1) - min(values1)
    m2 = max(values2) - min(values2)
    m1 = m1 if m1 != 0 else 1e-10
    m2 = m2 if m2 != 0 else 1e-10

    for k in range(1, len(front) - 1):
        distance[k] += (values1[sorted1[k + 1]] - values1[sorted1[k - 1]]) / m1
        distance[k] += (values2[sorted2[k + 1]] - values2[sorted2[k - 1]]) / m2
    return distance
def nsga2(population, max_gen, mutation_rate):
    gen_no = 0

    #parallelize the creation of the initial population
    with multiprocessing.Pool(processes=population) as pool:
        solution = pool.map(create_individual, [None]*population)
    #solution = [create_individual() for _ in range(population)]

    while gen_no < max_gen:

        #prepare parameters for multiprocessing
        params = [(gen_no, individual, ind) for ind, individual in enumerate(solution)]
        #run the objectives function in parallel
        with multiprocessing.Pool() as pool:
            objectives_values = pool.starmap(objectives, params)

        #objectives_values = [objectives(gen_no, individual, ind) for ind, individual in enumerate(solution)]
        print(objectives_values)
        objective1_values = [obj[0] for obj in objectives_values]
        objective2_values = [obj[1] for obj in objectives_values]
        non_dominated_sorted_solution = non_dominated_sorting_algorithm(objective1_values[:], objective2_values[:])

        print('Best Front for Generation:', gen_no)
        for values in non_dominated_sorted_solution[0]:
            print(np.array(solution[values]).round(3), end=" ")
        print("\n")

        crowding_distance_values = []
        for i in range(len(non_dominated_sorted_solution)):
            crowding_distance_values.append(
                crowding_distance(objective1_values[:], objective2_values[:], non_dominated_sorted_solution[i][:])
            )

        solution2 = solution[:]

        while len(solution2) < 2 * population:
            a1 = random.randint(0, population - 1)
            b1 = random.randint(0, population - 1)
            c1, c2 = crossover(solution[a1], solution[b1])
            mutate(c1, mutation_rate)
            mutate(c2, mutation_rate)
            solution2.append(c1)
            solution2.append(c2)

        #prepare parameters for multiprocessing
        params2 = [(gen_no, individual, ind) for ind, individual in enumerate(solution2)]
        #run the objectives function in parallel
        with multiprocessing.Pool() as pool:
            objectives_values2 = pool.starmap(objectives, params2)

        #objectives_values2 = [objectives(gen_no, individual, ind) for ind, individual in enumerate(solution)]
        objective1_values2 = [obj[0] for obj in objectives_values2]
        objective2_values2 = [obj[1] for obj in objectives_values2]
        non_dominated_sorted_solution2 = non_dominated_sorting_algorithm(objective1_values2[:], objective2_values2[:])

        crowding_distance_values2 = []
        for i in range(len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(
                crowding_distance(objective1_values2[:], objective2_values2[:], non_dominated_sorted_solution2[i][:])
            )

        new_solution = []
        for i in range(len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2_1 = [
                index_locator(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in range(len(non_dominated_sorted_solution2[i]))
            ]
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
            front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(len(non_dominated_sorted_solution2[i]))]
            front.reverse()
            for value in front:
                new_solution.append(value)
                if len(new_solution) == population:
                    break
            if len(new_solution) == population:
                break

        solution = [solution2[i] for i in new_solution]
        gen_no += 1

    return [objective1_values, objective2_values]

def non_dominating_curve_plotter(objective1_values, objective2_values):
    plt.figure(figsize=(15, 8))
    plt.xlabel('Objective 1: Percentage of agents in center (minimized)', fontsize=15)
    plt.ylabel('Objective 2: Ratio of second parameter to the sum of first two parameters (minimized)', fontsize=15)
    plt.scatter(objective2_values, objective1_values, c='red', s=25)
    plt.show()

# Parameters
population = 4
max_gen = 2
mutation_rate = 0.3

# Running NSGA-II
objective1_values, objective2_values = nsga2(population, max_gen, mutation_rate)
#save the two lists
np.save("objective1_values.npy", objective1_values)
np.save("objective2_values.npy", objective2_values)
# Plotting the non-dominated curve
non_dominating_curve_plotter(objective1_values, objective2_values)