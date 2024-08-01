import multiprocessing
from datetime import datetime
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
from skimage.transform import resize
import os, sys
from tqdm import tqdm
from numba import jit, vectorize, float32, njit, stencil, prange

from GA import save_final_plot
from numerical_solver import divergence
from real_swarming_sim import plot_generic_heatmap
from swarming_simulator import gradientX, gradientY, laplacian, show

PARAMETERS_FILE = "parameters_real_swarming.txt"
l = 0.02
N = 512
dx = l / N
indices = np.array([[(j - 2) % N, (j - 1) % N, (j + 1) % N, (j + 2) % N] for j in range(N)])
dx12 = 12 * dx
dxdx = dx ** 2
dt = 0.01
STEP_MAX = 500000
A_CENTER = (300, 380)
CENTER_LOCATION = (N // 2, N // 2)
ATTRACTANT = True
REPELLENT = True
V_RHO = True
RHO_0_FROM_FILE = True
NUMBER_OF_SPOTS = 2
NUMBER_OF_SPAWNS = 1
LOGGING = True
EPSILON = 1e3


class Simulator:
    def __init__(self, rho0, sigma, gamma, beta, alpha, D, D_B, beta_a, alpha_a, D_a, gamma_a, s_a, beta_r, alpha_r,
                 D_r, gamma_r, s_r, scale, rho_max, cushion, b_start_y=132):
        self.rho0 = rho0
        self.sigma = sigma
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha
        self.D = D
        self.D_B = D_B
        self.beta_a = beta_a
        self.alpha_a = alpha_a
        self.D_a = D_a
        self.gamma_a = gamma_a
        self.s_a = s_a
        self.beta_r = beta_r
        self.alpha_r = alpha_r
        self.D_r = D_r
        self.gamma_r = gamma_r
        self.s_r = s_r
        self.scale = scale
        self.rho_max = rho_max
        self.cushion = cushion
        self.sigma_times_scale = self.sigma * self.scale
        self.b_start_y = b_start_y
        self.rho = np.zeros((N, N))
        self.a = np.zeros((N, N))  # spot A
        self.b = np.zeros((N, N))  # spot B
        self.ua = np.zeros((N, N))
        self.ur = np.zeros((N, N))
        self.directory = ""
        self.t = 0
        self.timestep = 0

    def create_directory(self):
        if self.D_B > 0:
            self.directory = "../my_swarming_results_moving_and_diffusing_B/b_start_" + str(
                self.b_start_y) + "/D_" + str(self.D_B)
        else:
            self.directory = "../my_swarming_results/sim_0"
            while os.path.isdir(self.directory):
                self.directory = "../my_swarming_results/sim_" + str(
                    int(self.directory.split("/")[-1].split("_")[-1]) + 1)

        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

    #simply initialise with 2 spots of odor and 1 spawn point.
    def initial_conditions(self):
        self.rho[CENTER_LOCATION[0] - 20:CENTER_LOCATION[0] + 20,
        CENTER_LOCATION[1] - 20:CENTER_LOCATION[1] + 20] = self.rho0 + self.rho0 * random.uniform(-0.01, 0.01)
        #for now i assume to use a varying D and a varying location for the B spot.
        self.b[280:320, self.b_start_y-20:self.b_start_y+20] = self.rho0
        self.a[A_CENTER[0] - 20:A_CENTER[0] + 20, A_CENTER[1] - 20:A_CENTER[1] + 20] = self.rho0
        self.ua = self.s_a * self.gamma_a * self.rho
        self.ur = self.s_r * self.gamma_r * self.rho
        #show a and b
        #plot_generic_heatmap(self.a, range(0, 513, 128), range(0, 513, 128), "x", "y", 'A', True)
        #plot_generic_heatmap(self.b, range(0, 513, 128), range(0, 513, 128), "x", "y", 'B', True)

    def save_parameters(self):
        with open(self.directory + "/" + PARAMETERS_FILE, "w") as f:
            f.write("rho0: " + str(self.rho0) + "\n")
            f.write("sigma: " + str(self.sigma) + "\n")
            f.write("gamma: " + str(self.gamma) + "\n")
            f.write("beta: " + str(self.beta) + "\n")
            f.write("alpha: " + str(self.alpha) + "\n")
            f.write("D: " + str(self.D) + "\n")
            f.write("D_B: " + str(self.D_B) + "\n")
            f.write("beta_a: " + str(self.beta_a) + "\n")
            f.write("alpha_a: " + str(self.alpha_a) + "\n")
            f.write("D_a: " + str(self.D_a) + "\n")
            f.write("gamma_a: " + str(self.gamma_a) + "\n")
            f.write("s_a: " + str(self.s_a) + "\n")
            f.write("beta_r: " + str(self.beta_r) + "\n")
            f.write("alpha_r: " + str(self.alpha_r) + "\n")
            f.write("D_r: " + str(self.D_r) + "\n")
            f.write("gamma_r: " + str(self.gamma_r) + "\n")
            f.write("s_r: " + str(self.s_r) + "\n")
            f.write("scale: " + str(self.scale) + "\n")
            f.write("rho_max: " + str(self.rho_max) + "\n")
            f.write("cushion: " + str(self.cushion) + "\n")

    def save_final_plot(self):
        fig, ax = plt.subplots()
        img = ax.imshow(self.rho, cmap="cool", interpolation='nearest')

        # Set ticks and tick labels
        xs = range(0, 513, 128)
        ys = xs
        ax.set_xticks(np.linspace(0, self.rho.shape[1] - 1, len(xs)))
        ax.set_xticklabels(xs)
        ax.set_yticks(np.linspace(0, self.rho.shape[0] - 1, len(ys)))
        ax.set_yticklabels(ys)
        ax.invert_yaxis()
        # Customize tick and label sizes
        ax.tick_params(labelsize=25)
        plt.xticks(rotation=45)
        ax.set_xlabel("x", fontsize=45)
        ax.set_ylabel("y", fontsize=45)

        # Add colorbar
        clb = fig.colorbar(img, ax=ax, location="top", orientation="horizontal", shrink=0.5, pad=0.01, fraction=0.046)
        clb.ax.set_title(r'density ($\rho$)', fontsize=30)
        clb.ax.tick_params(labelsize=25)

        # Set aspect ratio
        ratio = 1.0
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
        plt.savefig(self.directory + "/final_density.pdf")
        plt.close()

    def log_data(self):
        np.save(self.directory + "/rho_" + str(self.timestep), self.rho)
        np.save(self.directory + "/a_" + str(self.timestep), self.a)
        np.save(self.directory + "/b_" + str(self.timestep), self.b)
        np.save(self.directory + "/ua_" + str(self.timestep), self.ua)
        np.save(self.directory + "/ur_" + str(self.timestep), self.ur)

    def run(self):
        self.create_directory()
        self.initial_conditions()
        self.save_parameters()
        success = False
        pbar = tqdm(total=STEP_MAX)
        print("Starting simulation with b_start_y = " + str(self.b_start_y) + " and D_B = " + str(self.D_B))
        while self.timestep < STEP_MAX and not success:
            gradient_x_rho = gradientX(self.rho)
            gradient_y_rho = gradientY(self.rho)
            potential_odor = -self.beta * np.log(self.alpha + self.a + self.b)
            potential_attractant = -self.beta_a * np.log(self.alpha_a + self.ua)
            potential_repellent = -self.beta_r * np.log(self.alpha_r + self.ur)
            potential_squeeze = self.sigma_times_scale * (1 + np.tanh((self.rho - self.rho_max) / self.cushion))
            potential = potential_odor + potential_attractant + potential_repellent + potential_squeeze

            dua = -self.gamma_a * self.ua + self.D_a * laplacian(self.ua) + self.s_a * self.rho
            dur = -self.gamma_r * self.ur + self.D_r * laplacian(self.ur) + self.s_r * self.rho
            drho = gradientX(self.rho * gradientX(potential) + self.sigma * gradient_x_rho) + gradientY(
                self.rho * gradientY(potential) + self.sigma * gradient_y_rho)
            da = -self.gamma * self.a + self.D * laplacian(self.a)
            db = -self.gamma * self.b + self.D_B * laplacian(self.b)
            self.ua += dt * dua
            self.ur += dt * dur
            self.a += dt * da
            self.b += dt * db
            self.rho += dt * drho

            if LOGGING and self.timestep % 1000 == 0:
                self.log_data()

            self.t += dt
            self.timestep += 1
            eps = np.max(np.abs(drho))
            if eps < EPSILON:
                success = True

            if np.max(self.rho) > 10e16 or np.isnan(np.max(self.rho)) or np.max(self.rho) < 0.0:
                success = False
                break

            #force rho, ua, ur, a and b to be positive
            self.rho[self.rho < 0] = 0
            self.ua[self.ua < 0] = 0
            self.ur[self.ur < 0] = 0
            self.a[self.a < 0] = 0
            self.b[self.b < 0] = 0



            pbar.update(1)

        if success or self.timestep == STEP_MAX:
            self.save_final_plot()
            self.log_data()
            print("Simulation finished successfully.")
        else:
            print("Simulation failed.")
            show(self.rho, self.directory)

        pbar.close()