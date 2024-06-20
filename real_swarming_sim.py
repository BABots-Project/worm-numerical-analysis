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
from swarming_simulator import gradientX, gradientY, laplacian, show


def updateParameters(textFileName, sep='='):
    with open(textFileName, 'r') as file:
            lines = file.readlines()
            sigma = float(lines[0].split(sep)[1])
            gamma = float(lines[1].split(sep)[1])
            beta = float(lines[2].split(sep)[1])
            alpha = float(lines[3].split(sep)[1])
            D = float(lines[4].split(sep)[1])
            beta_a = float(lines[5].split(sep)[1])
            alpha_a = float(lines[6].split(sep)[1])
            D_a = float(lines[7].split(sep)[1])
            gamma_a = float(lines[8].split(sep)[1])
            s_a = float(lines[9].split(sep)[1])
            beta_r = float(lines[10].split(sep)[1])
            alpha_r = float(lines[11].split(sep)[1])
            D_r = float(lines[12].split(sep)[1])
            gamma_r = float(lines[13].split(sep)[1])
            s_r = float(lines[14].split(sep)[1])
            scale = float(lines[15].split(sep)[1])
            rho_max = float(lines[16].split(sep)[1])
            cushion = float(lines[17].split(sep)[1])
            return sigma, gamma, beta, alpha, D, beta_a, alpha_a, D_a, gamma_a, s_a, beta_r, alpha_r, D_r, gamma_r, s_r, scale, rho_max, cushion


def initial_conditions(rho0, gamma_a, s_a, gamma_r, s_r):
    #all the initial density is set to 0 besides at the initial cell (300, 380)
    rho = np.zeros((N, N))
    U = np.zeros((N, N))
    rho[280:320, 360:400] = (rho0 + rho0 * random.uniform(-0.01, 0.01))
    #rho[236:276, 236:276] = (rho0 + rho0 * random.uniform(-0.01, 0.01))
    #set the target pheromone equal to s_a*gamma_a*rho0 in the 40x40 cells in the middle of the environment
    U[236:276, 236:276] = rho0
    Ua = s_a * gamma_a * rho
    Ur = s_r * gamma_r * rho
    t = 0
    step = 0
    return rho, U, Ua, Ur, t, step

def save_parameters(directory):
    with open(directory + "/parameters.txt", "w") as file:
        file.write(f"sigma={sigma}\n")
        file.write(f"gamma={gamma}\n")
        file.write(f"beta={beta}\n")
        file.write(f"alpha={alpha}\n")
        file.write(f"D={D}\n")
        file.write(f"beta_a={beta_a}\n")
        file.write(f"alpha_a={alpha_a}\n")
        file.write(f"D_a={D_a}\n")
        file.write(f"gamma_a={gamma_a}\n")
        file.write(f"s_a={s_a}\n")
        file.write(f"beta_r={beta_r}\n")
        file.write(f"alpha_r={alpha_r}\n")
        file.write(f"D_r={D_r}\n")
        file.write(f"gamma_r={gamma_r}\n")
        file.write(f"s_r={s_r}\n")
        file.write(f"scale={scale}\n")
        file.write(f"rho_max={rho_max}\n")
        file.write(f"cushion={cushion}\n")

l = 0.02
N = 512
dx = l / N
indices = np.array([[(j - 2) % N, (j - 1) % N, (j + 1) % N, (j + 2) % N] for j in range(N)])
dx12 = 12 * dx
dxdx = dx ** 2
dt = 0.01

sigma, gamma, beta, alpha, D, beta_a, alpha_a, D_a, gamma_a, s_a, beta_r, alpha_r, D_r, gamma_r, s_r, scale, rho_max, cushion = updateParameters("parameters_real_swarming.txt")

def run(args=None):
    global sigma, gamma, beta, alpha, D, beta_a, alpha_a, D_a, gamma_a, s_a, beta_r, alpha_r, D_r, gamma_r, s_r, dt
    if args is None:
        directory = "../my_swarming_results/sim_1"
        while os.path.isdir(directory):
            directory = "../my_swarming_results/sim_" + str(int(directory.split("/")[-1].split("_")[-1]) + 1)
    else:
        gen = str(args[0])
        i = str(args[1])
        now = datetime.now()

        # Format as string
        now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

        # Use this string in your directory name
        directory = f"my_swarming_results_optimisation/sim_{now_str}/gen_{gen}/ind_{i}"
        while os.path.isdir(directory):
            now = datetime.now()

            # Format as string
            now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
            directory = f"my_swarming_results_optimisation/sim_{now_str}/gen_{gen}/ind_{i}"
    time_integration = "euler"
    print("using: ", time_integration)
    eps_min = 1e3
    rho0 = 120e6
    step_max = 500000
    logging = False
    make_video = False
    # Read parameters from text file
    rho_matrices = []
    if args is not None:
        parameter_dir = args[2]
        if not parameter_dir.endswith(".json"):
            sigma, gamma, dc, k0, D, Dc, kc = updateParameters(
                parameter_dir)

    rho, U, Ua, Ur, t, step = initial_conditions(rho0, gamma_a, s_a, gamma_r, s_r)

    if not os.path.isdir(directory):
        os.makedirs(directory)

    #save parameters in the directory
    save_parameters(directory)
    # Main loop
    pbar = tqdm(total=step_max)
    success = False
    sigma_times_scale = sigma * scale
    while step < step_max:

        if time_integration == "euler":

            gradient_x_rho = gradientX(rho)
            gradient_y_rho = gradientY(rho)
            #gradient_x_U = gradientX(U)
            #gradient_y_U = gradientY(U)
            V = -beta * np.log(alpha + U) - beta_a * np.log(alpha_a + Ua) - beta_r * np.log(alpha_r + Ur)# + sigma_times_scale*(1+np.tanh((rho - rho_max)/cushion))
            drho = gradientX(rho * gradientX(V) + sigma * gradient_x_rho) + gradientY(
                rho * gradientY(V) + sigma * gradient_y_rho)

            dU = -gamma * U + D * laplacian(U)
            dUa = -gamma_a * Ua + D_a * laplacian(Ua) + s_a * rho
            dUr = -gamma_r * Ur + D_r * laplacian(Ur) + s_r * rho

            rho = (drho) * dt + rho
            #add small noise to rho where rho>0:
            #rho[rho>0] += rho[rho>0] * random.uniform(-0.001, 0.001)
            U = dU * dt + U
            Ua = dUa * dt + Ua
            Ur = dUr * dt + Ur

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

        # do not let Ua and Ur explode:
        #U[U > 10e16] = 10e16


        t = t + dt

        eps = np.max(np.abs(drho))

        if eps>10e16 or np.max(Ua)>10e16:
            Exception("Rho is too large")
            break
        '''
        scaled_rho = 10e7 * rho
        comparison_a = Ua > scaled_rho
        comparison_r = Ur > scaled_rho

        if np.any(comparison_a) or np.any(comparison_r):
            break
        '''

        if logging and step % 100 == 0:
            if make_video:
                rho_matrices.append(rho)
            # save matrices every 10000 steps
            np.save(directory + f"/rho_{step}.npy", rho)
            np.save(directory + f"/U_{step}.npy", U)


        if eps < eps_min or step == step_max - 1:
            rho_matrices.append(rho)
            # save matrices every 10000 steps
            np.save(directory + f"/rho_{step}.npy", rho)
            np.save(directory + f"/U_{step}.npy", U)
            print("saving in " + directory + f"/rho_{step}.npy")
            success = True
            break
        pbar.update(1)
        step += 1
    print("eps: ", eps)

    if make_video:
        fig, ax = plt.subplots()
        if len(rho_matrices) == 0:
            rho_matrices.append(rho)
        cax = ax.imshow(rho_matrices[0], cmap='viridis', interpolation='nearest')

        def update(frame):
            cax.set_array(rho_matrices[frame])
            return [cax]

        ani = animation.FuncAnimation(fig, update, frames=len(rho_matrices), blit=True)
        ani.save('rho_evolution.mp4', writer='ffmpeg', fps=1, dpi=100)

    #show(rho, directory)
    #save in directory the pdf of the final density
        save_final_plot(rho, directory)

    if success:
        # from clustering2 import evaluate
        # c, kurt, bins, ys = evaluate(directory + f"/rho_{step}.npy", 50)
        c = -rho[236:276, 236:276].sum() / rho.sum()
        print(c)
    else:
        c = -1
        t = step_max * dt

    return c, t

if __name__ == "__main__":
    arg = sys.argv[1]
    if arg=="run":
        #instaead of running only 1 instance, run an instance
        run()
    #'''
    else:
        from clustering2 import evaluate
        directory=f"../my_swarming_results/sim_{int(arg)}"
        #directory = f"../{arg}"
        #step=2091
        #c, kurt, bins, ys = evaluate(directory + f"/rho_{step}.npy", 50)
        #print(c)
        #show(np.load(directory + f"/rho_{step}.npy"))
        video=True
        def build_video(dir_):
            rho_matrices = []
            #filtered_directory shall contained only the files with "rho" in their name and that end with ".npy"
            filtered_directory = [file for file in os.listdir(dir_) if file.endswith(".npy") and "rho" in file]
            sorted_directory = sorted(filtered_directory, key=lambda x: int(x.split("_")[-1].split(".")[0]))

            for file in sorted_directory:
                if file.endswith(".npy") and "rho" in file:
                    rho_matrices.append(np.load(dir_ + "/" + file))

            c = rho_matrices[-1][236:276, 236:276].sum() / rho_matrices[-1].sum()
            fig, ax = plt.subplots()
            #set minimum value of all rho matrices to 10e3
            for i in range(len(rho_matrices)):
                rho_matrices[i][rho_matrices[i] < 10e3] = 10e3
            cax = ax.imshow(rho_matrices[0], cmap='rainbow', interpolation='nearest', norm=LogNorm(vmin=10e3, vmax=np.max(rho_matrices[0])))

            def update(frame):
                cax.set_array(rho_matrices[frame])
                return [cax]
            #show(rho_matrices[-1])
            ani = animation.FuncAnimation(fig, update, frames=len(rho_matrices), blit=True)
            ani.save('rho_evolution.mp4', writer='ffmpeg', fps=20, dpi=100)
            #get the sum of the densities inside of the target area over the total sum of the densities

            print(c)
        if video:
            build_video(directory)
        else:
            j=499999
            m = np.load(directory + f"/rho_{j}.npy")
            c = m[236:276, 236:276].sum() / m.sum()
            #print(c)
            #show(m, directory)

            for j in [0, 250000, 499999]:
                m = np.load(directory + f"/rho_{j}.npy")
                c = m[236:276, 236:276].sum()/m.sum()
                print(c)
                show(m, directory)
        # #'''