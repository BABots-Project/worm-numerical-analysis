import math
import random

from optimiser import create_individual, objectives, crossover, mutate

def pareto_dominance(objective1, objective2):
    # Check for any objective where objective2 is strictly better than objective1
    condition1 = any(o2 < o1 for o1, o2 in zip(objective1, objective2))
    # Check for any objective where objective1 is strictly better than objective2
    condition2 = any(o1 < o2 for o1, o2 in zip(objective1, objective2))
    return condition1 and not condition2

def simulated_annealing(population, max_gen, mutation_rate):
    # Initial population
    current_population = [create_individual() for _ in range(population)]

    # Initial solution
    current_solution = current_population[0]
    current_objective = objectives(0, current_solution, 0)

    # Parameters for simulated annealing
    initial_temperature = 1000
    final_temperature = 0.01
    alpha = 0.9

    # Main loop
    for gen_no in range(max_gen):
        print("Generation:", gen_no)
        temperature = initial_temperature * (alpha ** gen_no)
        print("Temperature:", temperature)
        # Select two individuals from the population
        a1 = random.randint(0, population - 1)
        b1 = random.randint(0, population - 1)

        # Create a new solution
        new_solution, _ = crossover(current_population[a1], current_population[b1])
        mutate(new_solution, mutation_rate)

        # Calculate the objective of the new solution
        new_objective = objectives(gen_no, new_solution, 0)

        if pareto_dominance(new_objective, current_objective):
            current_solution = new_solution
            current_objective = new_objective
        # If the new solution is worse, accept it with a probability that decreases with temperature
        elif random.random() < math.exp(-1 / temperature):
            current_solution = new_solution
            current_objective = new_objective

        # If the temperature is low enough, stop the algorithm
        if temperature <= final_temperature:
            break
        print("Objective:", current_objective)
        print("Solution:", current_solution)
    return current_solution

pop_size = 10
max_gen = 100
mutation_rate = 0.1

solution = simulated_annealing(pop_size, max_gen, mutation_rate)
print(solution)
