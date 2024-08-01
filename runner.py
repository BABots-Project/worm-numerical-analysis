import numpy as np
import concurrent.futures
from real_swarming_sim import updateParameters
from varying_b_diffusion_simulator import Simulator, PARAMETERS_FILE

sigma, gamma, beta, alpha, D, beta_a, alpha_a, D_a, gamma_a, s_a, beta_r, alpha_r, D_r, gamma_r, s_r, scale, rho_max, cushion = updateParameters(PARAMETERS_FILE)
with open(PARAMETERS_FILE, 'r') as file:
    lines = file.readlines()
    rho0 = float(lines[-1].split("=")[1])

#sim = Simulator(rho0, sigma, gamma, beta, alpha, D, D, beta_a, alpha_a, D_a, gamma_a, s_a, beta_r, alpha_r, D_r, gamma_r, s_r, scale, rho_max, cushion)
#sim.run()
def run_simulation(simulator):
    simulator.run()

b_starts = list(range(20, 256, 20))
max_d = 4.47*10**-9
min_d = 1.12*10**-9
delta_d = (max_d - min_d) / 10
d_list = list(np.arange(min_d, max_d, delta_d))
simulators = []

for b_start in b_starts:
    for D_B in d_list:
        sim = Simulator(rho0, sigma, gamma, beta, alpha, D, D_B, beta_a, alpha_a, D_a, gamma_a, s_a, beta_r, alpha_r, D_r, gamma_r, s_r, scale, rho_max, cushion, b_start)
        simulators.append(sim)

with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(run_simulation, simulators)
