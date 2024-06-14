import math
import multiprocessing
import os
import random
from datetime import datetime

import numpy as np
from matplotlib.colors import LogNorm
from numba import njit
from tqdm import tqdm

ODOR = True
ATTRACTIVE_PHEROMONE = False
REPULSIVE_PHEROMONE = False
V_RHO = False
N = 512
l = 0.02
dx = l / N
dxdx = dx ** 2
indices = np.array([[(j - 2) % N, (j - 1) % N, (j + 1) % N, (j + 2) % N] for j in range(N)])
dx12 = 12 * dx

parameter_ranges = {
    "rho0": (1e6, 10e9)
}

parameter_ranges_odor = {
    "sigma": (0, 1),
    "D": (0, 1),
    "beta": (0, 1),
    "alpha": (0, 1),
    "gamma": (0, 1)
}

parameter_ranges_pheromone_attractive = {
    "beta_a": (0, 1),
    "alpha_a": (0, 15e10),
    "gamma_a": (0, 1),
    "s_a": (0, 1e4),
    "D_a": (0, 1)
}

parameter_ranges_pheromone_repulsive = {
    "beta_r": (-1, 1e-9),
    "alpha_r": (0, 15e10),
    "D_r": (0, 1),
    "gamma_r": (0, 1e-1),
    "s_r": (0, 1e3),
}

parameter_ranges_vrho = {
    "scale": (0, 20),
    "rho_max": (0, 28e10),
    "cushion": (0, 2e10)
}


def save_individual_parameters(individual, directory):
    with open(directory + "/parameters.txt", "w") as f:
        f.write("rho0: " + str(individual[0]) + "\n")
        if ODOR:
            f.write("D: " + str(individual[1]) + "\n")
            f.write("beta: " + str(individual[2]) + "\n")
            f.write("alpha: " + str(individual[3]) + "\n")
            f.write("gamma: " + str(individual[4]) + "\n")
        if ATTRACTIVE_PHEROMONE:
            f.write("beta_a: " + str(individual[5]) + "\n")
            f.write("alpha_a: " + str(individual[6]) + "\n")
            f.write("gamma_a: " + str(individual[7]) + "\n")
            f.write("s_a: " + str(individual[8]) + "\n")
            f.write("D_a: " + str(individual[9]) + "\n")
        if REPULSIVE_PHEROMONE:
            f.write("beta_r: " + str(individual[10]) + "\n")
            f.write("alpha_r: " + str(individual[11]) + "\n")
            f.write("gamma_r: " + str(individual[12]) + "\n")
            f.write("s_r: " + str(individual[13]) + "\n")
            f.write("D_r: " + str(individual[14]) + "\n")
        if V_RHO:
            f.write("sigma: " + str(individual[15]) + "\n")
            f.write("scale: " + str(individual[16]) + "\n")
            f.write("rho_max: " + str(individual[17]) + "\n")
            f.write("cushion: " + str(individual[18]) + "\n")
        print("parameters saved in " + directory)


def initial_conditions_only_odor(rho0):
    #all the initial density is set to 0 besides at the initial cell (300, 380)
    rho = np.zeros((N, N))
    U = np.zeros((N, N))
    rho[280:320, 360:400] = (rho0 + rho0 * random.uniform(-0.01, 0.01))
    U[236:276, 236:276] = rho0
    t = 0
    step = 0
    return rho, U, t, step


def initial_conditions_odor_and_attractive_pheromone(rho0, s_a, gamma_a):
    rho = np.zeros((N, N))
    U = np.zeros((N, N))
    rho[280:320, 360:400] = (rho0 + rho0 * random.uniform(-0.01, 0.01))
    U[236:276, 236:276] = rho0
    Ua = np.zeros((N, N))
    Ua[236:276, 236:276] = s_a * gamma_a * rho0
    t = 0
    step = 0
    return rho, U, Ua, t, step


def initial_conditions_odor_and_attractive_and_repulsive_pheromone(rho0, s_a, gamma_a, s_r, gamma_r):
    rho = np.zeros((N, N))
    U = np.zeros((N, N))
    rho[280:320, 360:400] = (rho0 + rho0 * random.uniform(-0.01, 0.01))
    U[236:276, 236:276] = rho0
    Ua = np.zeros((N, N))
    Ua[236:276, 236:276] = s_a * gamma_a * rho0
    Ur = np.zeros((N, N))
    Ur[236:276, 236:276] = s_r * gamma_r * rho0
    t = 0
    step = 0
    return rho, U, Ua, Ur, t, step


@njit(fastmath=True)
def gradientX(in_array):
    out = np.zeros_like(in_array)

    for i in range(N):
        for j in range(N):
            # Compute shifted versions using pre-computed indices
            left_2 = in_array[i, indices[j][0]]
            left_1 = in_array[i, indices[j][1]]
            right_1 = in_array[i, indices[j][2]]
            right_2 = in_array[i, indices[j][3]]

            # Compute gradient
            out[i, j] = (-left_2 + 8 * left_1 - 8 * right_1 + right_2) / dx12

    return out


@njit(fastmath=True)
def gradientY(in_array):
    out = np.zeros_like(in_array)

    for i in range(N):
        for j in range(N):
            # Compute shifted versions
            up_2 = in_array[indices[i][0], j]
            up_1 = in_array[indices[i][1], j]
            down_1 = in_array[indices[i][2], j]
            down_2 = in_array[indices[i][3], j]

            # Compute gradient
            out[i, j] = (-up_2 + 8 * up_1 - 8 * down_1 + down_2) / dx12

    return out


@njit(fastmath=True)
def laplacian(in_array):
    out = np.zeros_like(in_array)

    for i in range(N):
        for j in range(N):
            # Compute shifted versions
            center = in_array[i, j]
            right = in_array[i, indices[j][2]]
            left = in_array[i, indices[j][1]]
            up = in_array[indices[i][1], j]
            down = in_array[indices[i][2], j]
            up_right = in_array[indices[i][1], indices[j][2]]
            up_left = in_array[indices[i][1], indices[j][1]]
            down_right = in_array[indices[i][2], indices[j][2]]
            down_left = in_array[indices[i][2], indices[j][1]]

            # Compute laplacian
            out[i, j] = -center + 0.20 * (right + left + up + down) + \
                        0.05 * (up_right + up_left + down_right + down_left)

    out /= dxdx
    return out

def save_final_plot(rho, directory):
    import matplotlib.pyplot as plt
    plt.imshow(rho, cmap="rainbow")
    plt.colorbar()
    #save as pdf
    plt.savefig(directory + "/final_plot.pdf")
    plt.close()

    #same with imshow with LogNorm cut off at 10^3
    plt.imshow(rho, cmap="rainbow", norm=LogNorm(vmin=1e3, vmax=rho.max()))
    plt.colorbar()
    #save as pdf
    plt.savefig(directory + "/final_plot_lognorm.pdf")
    plt.close()

def run(individual, gen, individual_index):
    rho0 = individual[0]
    if ODOR:
        sigma, D, beta, alpha, gamma = individual[1:6]
    if ATTRACTIVE_PHEROMONE:
        beta_a, alpha_a, gamma_a, s_a, D_a = individual[6:11]
    if REPULSIVE_PHEROMONE:
        beta_r, alpha_r, gamma_r, s_r, D_r = individual[11:16]
    if V_RHO:
        scale, rho_max, cushion = individual[16:19]
    dt = 0.01

    now = datetime.now()
    # Format as string
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Use this string in your directory name
    directory = f"../my_swarming_results_optimisation/sim_{now_str}/gen_{gen}/ind_{individual_index}"
    while os.path.isdir(directory):
        now = datetime.now()

        # Format as string
        now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        directory = f"../my_swarming_results_optimisation/sim_{now_str}/gen_{gen}/ind_{individual_index}"

    eps_min = 1e3
    step_max = 500000
    logging = True

    if not os.path.isdir(directory):
        os.makedirs(directory)

    #save parameters in the directory
    save_individual_parameters(individual, directory)
    # Main loop
    pbar = tqdm(total=step_max)
    success = False
    rho = np.zeros((N, N))
    U = np.zeros((N, N))
    Ua = np.zeros((N, N))
    Ur = np.zeros((N, N))
    if ODOR and not ATTRACTIVE_PHEROMONE and not REPULSIVE_PHEROMONE and not V_RHO:
        rho, U, t, step = initial_conditions_only_odor(rho0)
    elif ODOR and ATTRACTIVE_PHEROMONE and not REPULSIVE_PHEROMONE and not V_RHO:
        rho, U, Ua, t, step = initial_conditions_odor_and_attractive_pheromone(rho0, s_a, gamma_a)
    elif ODOR and ATTRACTIVE_PHEROMONE and REPULSIVE_PHEROMONE:
        rho, U, Ua, Ur, t, step = initial_conditions_odor_and_attractive_and_repulsive_pheromone(rho0, s_a, gamma_a,
                                                                                                 s_r, gamma_r)

    if V_RHO:
        sigma_times_scale = sigma * scale
    while step < step_max:

        gradient_x_rho = gradientX(rho)
        gradient_y_rho = gradientY(rho)
        V = np.zeros((N, N))
        if ODOR:
            V = -beta * np.log(alpha + U)
            dU = -gamma * U + D * laplacian(U)
            U = dU * dt + U
        if ATTRACTIVE_PHEROMONE:
            V -= beta_a * np.log(alpha_a + Ua)
            dUa = -gamma_a * Ua + D_a * laplacian(Ua) + s_a * rho
            Ua = dUa * dt + Ua
        if REPULSIVE_PHEROMONE:
            V -= beta_r * np.log(alpha_r + Ur)
            dUr = -gamma_r * Ur + D_r * laplacian(Ur) + s_r * rho
            Ur = dUr * dt + Ur
        if V_RHO:
            V += sigma_times_scale * (1 + np.tanh((rho - rho_max) / cushion))

        drho = gradientX(rho * gradientX(V) + sigma * gradient_x_rho) + gradientY(
            rho * gradientY(V) + sigma * gradient_y_rho)

        rho = (drho) * dt + rho
        #add small noise to rho where rho>0:
        #rho[rho>0] += rho[rho>0] * random.uniform(-0.001, 0.001)

        # check for non numerical values
        if np.any(np.isnan(rho)) or np.any(np.isnan(U)):
            Exception("Non numerical values")
            break

        # check for infinities
        if np.any(rho) > 10e16:
            Exception("Rho is too large")
            break

        # add 10e7 (100/mm^2 = 10e7/mm^2) to Ua in the target area
        #Ua[236:276, 236:276] += 10e9

        # force Ua and Ur positive
        U[U < 0] = 0
        rho[rho < 0] = 0

        t = t + dt

        eps = np.max(np.abs(drho))

        if eps > 10e16 or np.max(Ua) > 10e16:
            Exception("Rho is too large")
            break

        if logging and step % 1000 == 0:
            # save matrices every 10000 steps
            np.save(directory + f"/rho_{step}.npy", rho)
            np.save(directory + f"/U_{step}.npy", U)

        if eps < eps_min or step == step_max - 1:
            # save matrices every 10000 steps
            np.save(directory + f"/rho_{step}.npy", rho)
            np.save(directory + f"/U_{step}.npy", U)
            print("saving in " + directory + f"/rho_{step}.npy")
            success = True
            break
        pbar.update(1)
        step += 1
    print("eps: ", eps)

    if success:
        save_final_plot(rho, directory)
        c = -rho[236:276, 236:276].sum() / rho.sum()
        print(c)
    else:
        c = 0

    return c


def evaluate_fitness(individual, gen, individual_index):
    try:
        c = run(individual, gen, individual_index)
    except Exception as e:
        print(e)
        c = 0
    return c


def create_individual():
    individual = []
    for parameter, (lower, upper) in parameter_ranges.items():
        individual.append(random.uniform(lower, upper))
    if ODOR:
        for parameter, (lower, upper) in parameter_ranges_odor.items():
            individual.append(random.uniform(lower, upper))
    if ATTRACTIVE_PHEROMONE:
        for parameter, (lower, upper) in parameter_ranges_pheromone_attractive.items():
            individual.append(random.uniform(lower, upper))
    if REPULSIVE_PHEROMONE:
        for parameter, (lower, upper) in parameter_ranges_pheromone_repulsive.items():
            individual.append(random.uniform(lower, upper))
    if V_RHO:
        for parameter, (lower, upper) in parameter_ranges_vrho.items():
            individual.append(random.uniform(lower, upper))
    return individual


def tournament_selection(population, fitness_list):
    #select two random individuals
    individual1_index = random.randint(0, len(fitness_list) - 1)
    individual2_index = random.randint(0, len(fitness_list) - 1)
    while individual1_index == individual2_index:
        individual2_index = random.randint(0, len(fitness_list) - 1)
    #select the best individual
    if fitness_list[individual1_index] < fitness_list[individual2_index]:
        return population[individual1_index]
    else:
        return population[individual2_index]


def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        r = random.random()
        if r < mutation_rate:
            if i < 1:
                min_, max_ = parameter_ranges[list(parameter_ranges.keys())[i]]
            elif 1 <= i < 6:
                min_, max_ = parameter_ranges_odor[list(parameter_ranges_odor.keys())[i - 1]]

            elif 6 <= i < 11:
                min_, max_ = parameter_ranges_pheromone_attractive[list(parameter_ranges_pheromone_attractive.keys())[i - 6]]

            elif 11 <= i < 16:
                min_, max_ = parameter_ranges_pheromone_repulsive[list(parameter_ranges_pheromone_repulsive.keys())[i - 11]]
            try:
                #print("editing individual ", i, " with min: ", min_, " and max: ", max_, "gene: ", individual[i])
                individual[i] = random.uniform(min_, max_)
            except:
                print("error in mutation")
                print("individual: ", individual)
                print("i: ", i)
    return individual


def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def load_individual(directory):
    individual = []
    with open(directory, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                individual.append(float(line.split("=")[1]))
            elif 1 <= i < 6 and ODOR:
                individual.append(float(line.split("=")[1]))
            elif 6 <= i < 11 and ATTRACTIVE_PHEROMONE:
                individual.append(float(line.split("=")[1]))
            elif 11 <= i < 16 and REPULSIVE_PHEROMONE:
                individual.append(float(line.split("=")[1]))
    return individual


def GA(maxgen, popsize, mutation_rate, elitism=0):
    #create random population
    population = [create_individual() for _ in range(popsize - 1)]
    population.append(load_individual("current_best_swarming_individual.txt"))
    #print(population)
    best_individual_list = []
    best_fitness_list = []
    for gen in range(maxgen):
        print("generation: ", gen)
        #run evaluation in parallel
        if gen!=0:
            params = [(population[j], gen, j) for j in range(elitism, popsize)]
            with multiprocessing.Pool() as pool:
                new_fitness_list = pool.starmap(evaluate_fitness, params)
            fitness_list[elitism:] = new_fitness_list
        else:
            params = [(population[j], gen, j) for j in range(popsize)]
            with multiprocessing.Pool() as pool:
                fitness_list = pool.starmap(evaluate_fitness, params)


        #find the best individual, by selecting the index of the minimum fitness
        best_individual_index = fitness_list.index(min(fitness_list))
        best_individual = population[best_individual_index]
        best_fitness = fitness_list[best_individual_index]

        best_individual_list.append(best_individual)
        best_fitness_list.append(best_fitness)

        # sort population by the fitness
        population = [x for _, x in sorted(zip(fitness_list, population))]
        #sort fitness list
        fitness_list.sort()

        #create a new population
        new_population = population[:elitism]


        for j in range(elitism, popsize - elitism):
            #select two individuals
            parent1 = tournament_selection(population, fitness_list)
            parent2 = tournament_selection(population, fitness_list)
            #crossover
            child1, child2 = crossover(parent1, parent2)
            #mutate
            child = mutate(child1, mutation_rate)
            new_population.append(child)

        population = new_population
        print("best individual: ", best_individual, " best fitness: ", best_fitness)
    return best_individual_list, best_fitness_list


def saes(maxgen, popsize, lambda_min, lambda_max, alpha_0, c, m, elitism=0):
    # Create initial population with a random uniform distribution
    population = [create_individual() for _ in range(popsize - 1)]
    population.append(load_individual("current_best_swarming_individual.txt"))
    # Initial parameters
    t_e = 1.0
    alpha_e = alpha_0
    best_individual_list = []
    best_fitness_list = []
    fitness_history = []

    for gen in range(maxgen):
        print("generation: ", gen)

        if gen != 0:
            params = [(population[j], gen, j) for j in range(elitism, popsize)]
            with multiprocessing.Pool() as pool:
                new_fitness_list = pool.starmap(evaluate_fitness, params)
            fitness_list[elitism:] = new_fitness_list
        else:
            params = [(population[j], gen, j) for j in range(popsize)]
            with multiprocessing.Pool() as pool:
                fitness_list = pool.starmap(evaluate_fitness, params)

        # Record best fitness and individual
        best_individual_index = fitness_list.index(min(fitness_list))
        best_individual = population[best_individual_index]
        best_fitness = fitness_list[best_individual_index]

        best_individual_list.append(best_individual)
        best_fitness_list.append(best_fitness)
        fitness_history.append(best_fitness)
        # sort population by the fitness
        population = [x for _, x in sorted(zip(fitness_list, population))]
        # sort fitness list
        fitness_list.sort()
        # Calculate gamma and update t_e
        if gen >= 2:
            gamma_e = abs(fitness_history[-1] - fitness_history[-2])
            if gamma_e == 0:
                t_e = t_e
                alpha_e = alpha_0 * t_e
            else:
                recent_gammas = fitness_history[-(m + 1):-1] if gen >= m + 1 else fitness_history
                t_e = m / sum(abs(recent_gammas[i] - recent_gammas[i - 1]) for i in range(1, len(recent_gammas)))
                alpha_e = c * alpha_0 * t_e
        # Update lambda_e and alpha_e
        lambda_e = ((t_e + (lambda_min / lambda_max) ** 2) / (t_e + (lambda_min / lambda_max))) * lambda_max


        # Maintain elite individuals
        new_population = population[:elitism]

        # Create new offspring
        for _ in range(elitism, math.ceil(lambda_e) - elitism):
            parent1 = tournament_selection(population, fitness_list)
            parent2 = tournament_selection(population, fitness_list)
            child1, child2 = crossover(parent1, parent2)
            child = mutate(child1, alpha_e)
            new_population.append(child)

        # Ensure the population size is maintained
        if len(new_population) > popsize:
            new_population = new_population[:popsize]

        population = new_population
        print("best individual: ", best_individual, " best fitness: ", best_fitness)

    return best_individual_list, best_fitness_list

if __name__ == "__main__":
    maxgen = 50
    popsize = 20
    mutation_rate = 0.1
    elitism = 3
    lambda_min = 2
    lambda_max = 100
    c=2
    m=10
    best_individual_list, best_fitness_list = saes(maxgen, popsize, lambda_min, lambda_max, mutation_rate, c, m, elitism=elitism)#GA(maxgen, popsize, mutation_rate, elitism)
    print(best_individual_list)
    print(best_fitness_list)
    #save the best individual and its fitness
    with open("best_individual.txt", "w") as f:
        f.write("best individual: " + str(best_individual_list[-1]) + "\n")
        f.write("best fitness: " + str(best_fitness_list[-1]) + "\n")
    print("best individual: ", best_individual_list[-1])
    print("best fitness: ", best_fitness_list[-1])
    #run the best individual
    #run(best_individual_list[-1])
    #print("done")
