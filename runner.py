import sys

import numpy as np
import concurrent.futures
from varying_b_diffusion_simulator import Simulator
OPTIMAL_ODOR_PARAMETERS_FILE = "parameters/optimised_odor.txt"
OPTIMAL_PHEROMONES_PARAMETERS_FILE = "parameters/optimised_pheromones.txt"
BASE_PARAMETERS_FILE = "parameters/base_individual.txt"
mode = sys.argv[1]
if mode == "odor":
    PARAMETERS_FILE = OPTIMAL_ODOR_PARAMETERS_FILE
    ATTRACTANT = False
    REPELLENT = False
    V_RHO = False
else:
    PARAMETERS_FILE = OPTIMAL_PHEROMONES_PARAMETERS_FILE
    ATTRACTANT = True
    REPELLENT = True
    V_RHO = True
def updateParameters(textFileName):
    sep = "="
    with open(textFileName, 'r') as file:
            lines = file.readlines()
            sigma = float(lines[0].split(sep)[1])
            gamma = float(lines[1].split(sep)[1])
            beta = float(lines[2].split(sep)[1])
            alpha = float(lines[3].split(sep)[1])
            D = float(lines[4].split(sep)[1])
            if ATTRACTANT:
                beta_a = float(lines[5].split(sep)[1])
                alpha_a = float(lines[6].split(sep)[1])
                D_a = float(lines[7].split(sep)[1])
                gamma_a = float(lines[8].split(sep)[1])
                s_a = float(lines[9].split(sep)[1])
            else:
                beta_a = 0
                alpha_a = 0
                D_a = 0
                gamma_a = 0
                s_a = 0
            if REPELLENT:
                beta_r = float(lines[10].split(sep)[1])
                alpha_r = float(lines[11].split(sep)[1])
                D_r = float(lines[12].split(sep)[1])
                gamma_r = float(lines[13].split(sep)[1])
                s_r = float(lines[14].split(sep)[1])
            else:
                beta_r = 0
                alpha_r = 0
                D_r = 0
                gamma_r = 0
                s_r = 0
            if V_RHO:
                scale = float(lines[15].split(sep)[1])
                rho_max = float(lines[16].split(sep)[1])
                cushion = float(lines[17].split(sep)[1])
            else:
                scale = 0
                rho_max = 0
                cushion = 0
            return sigma, gamma, beta, alpha, D, beta_a, alpha_a, D_a, gamma_a, s_a, beta_r, alpha_r, D_r, gamma_r, s_r, scale, rho_max, cushion


parameter_file = PARAMETERS_FILE
sigma, gamma, beta, alpha, D, beta_a, alpha_a, D_a, gamma_a, s_a, beta_r, alpha_r, D_r, gamma_r, s_r, scale, rho_max, cushion = updateParameters(parameter_file)
with open(parameter_file, 'r') as file:
    lines = file.readlines()
    rho0 = float(lines[-1].split("=")[1])

'''b_start = 132
sim = Simulator(rho0, sigma, gamma, beta, alpha, D, D, beta_a, alpha_a, D_a, gamma_a, s_a, beta_r, alpha_r, D_r, gamma_r, s_r, scale, rho_max, cushion, mode, parameter_file, b_start, True)
print("rho0: ", rho0)
sim.run()
sys.exit()'''
def run_simulation(simulator):
    simulator.run()

b_starts = list(range(20, 256, 20))
max_d = 4.47*10**-9
min_d = 1.12*10**-9
delta_d = (max_d - min_d) / 10
d_list = list(np.arange(min_d, max_d, delta_d))
simulators = []

#parameter_file = PARAMETERS_FILE
for b_start in b_starts:
    for D_B in d_list:
        sim = Simulator(rho0, sigma, gamma, beta, alpha, D, D_B, beta_a, alpha_a, D_a, gamma_a, s_a, beta_r, alpha_r, D_r, gamma_r, s_r, scale, rho_max, cushion, mode, parameter_file, b_start)
        simulators.append(sim)

with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(run_simulation, simulators)
