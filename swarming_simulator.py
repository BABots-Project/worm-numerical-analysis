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
import concurrent.futures

'''
This file represents the simulation of a swarm of BABOTs that are trying to reach a target. 
The target is a 40x40 square in the middle of the simulation space. 
The environment is a 512x512 square of length 20x20mm.
The BABOTs are placed in a specific square outside of the target area at coordinates (300, 380).
Parameters for the simulation are specified in the parameters_swarming.txt file.
All units are expressed in standard units (meters, seconds, etc.)
'''


def updateParameters(textFileName, sep='='):
    with open(textFileName, 'r') as file:
        if textFileName.endswith(".json"):
            import json
            data = json.load(file)
            sigma = data["sigma"]
            scale = data["scale"]
            rho_max = data["rho_max"]
            cushion = data["cushion"]
            beta_a = data["beta_a"]
            beta_r = data["beta_r"]
            alfa_a = data["alpha_a"]
            alfa_r = data["alpha_r"]
            D_a = data["D_a"]
            D_r = data["D_r"]
            gamma_a = data["gamma_a"]
            gamma_r = data["gamma_r"]
            s_a = data["s_a"]
            s_r = data["s_r"]
            rho0 = data["rho0"]
            return sigma, scale, rho_max, cushion, beta_a, beta_r, alfa_a, alfa_r, D_a, D_r, gamma_a, gamma_r, s_a, s_r, rho0
        else:
            lines = file.readlines()
            sigma = float(lines[0].split(sep)[1])
            scale = float(lines[1].split(sep)[1])
            rho_max = float(lines[2].split(sep)[1])
            cushion = float(lines[3].split(sep)[1])
            dt = float(lines[4].split(sep)[1])
            beta_a = float(lines[5].split(sep)[1])
            beta_r = float(lines[6].split(sep)[1])
            alfa_a = float(lines[7].split(sep)[1])
            alfa_r = float(lines[8].split(sep)[1])
            D_a = float(lines[9].split(sep)[1])
            D_r = float(lines[10].split(sep)[1])
            gamma_a = float(lines[11].split(sep)[1])
            gamma_r = float(lines[12].split(sep)[1])
            s_a = float(lines[13].split(sep)[1])
            s_r = float(lines[14].split(sep)[1])

            return sigma, scale, rho_max, cushion, dt, beta_a, beta_r, alfa_a, alfa_r, D_a, D_r, gamma_a, gamma_r, s_a, s_r


l = 0.02
N = 512
dx = l / N
indices = np.array([[(j - 2) % N, (j - 1) % N, (j + 1) % N, (j + 2) % N] for j in range(N)])
dx12 = 12 * dx
dxdx = dx ** 2

sigma, scale, rho_max, cushion, dt, beta_a, beta_r, alfa_a, alfa_r, D_a, D_r, gamma_a, gamma_r, s_a, s_r = updateParameters(
    'parameters_swarming.txt')


#utility function to show a matrix using im.show()
def show(matrix, directory):
    #invert y axis to have the origin in the bottom left corner
    matrix = np.flipud(matrix)
    #set values that are less than 10e3 to 10e3
    matrix[matrix < 10e3] = 10e3
    plt.imshow(matrix, cmap='rainbow', interpolation='nearest', norm=LogNorm(vmin=10e3, vmax=np.max(matrix)))
    #plt.imshow(matrix, cmap='rainbow', interpolation='nearest')
    plt.gca().invert_yaxis()
    #plt.colorbar(label='rho')
    #set the label sizes and tick sizes
    plt.tick_params(axis='both', which='major', labelsize=50)
    plt.tick_params(axis='both', which='minor', labelsize=48)
    plt.xlabel('x', fontsize=60)
    plt.ylabel('y', fontsize=60)
    #plt.tight_layout()
    #same for colorbar
    cbar = plt.colorbar()
    cbar.set_label(r'$\rho$', fontsize=60)
    cbar.ax.tick_params(labelsize=60)

    #set colorbar label size

    plt.savefig(directory + "/final_density.pdf")
    plt.show()


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


def initial_conditions(rho0, s_a, gamma_a, s_r, gamma_r):
    #all the initial density is set to 0 besides at the initial cell (300, 380)
    rho = np.zeros((N, N))
    rho[280:320, 360:400] = (rho0 + rho0 * random.uniform(-0.01, 0.01))
    #rho[236:276, 236:276] = (rho0 + rho0 * random.uniform(-0.01, 0.01))
    Ua = s_a * gamma_a * rho
    Ur = s_r * gamma_r * rho
    #set the target pheromone equal to s_a*gamma_a*rho0 in the 40x40 cells in the middle of the environment
    Ua[236:276, 236:276] = s_a * gamma_a * rho0
    t = 0
    step = 0
    return rho, Ua, Ur, t, step


def derivatives(rho, Ua, Ur):
    Vrho = sigma * scale * (1 + np.tanh((rho - rho_max) / cushion))
    V_u_a = -beta_a * np.log(alfa_a + Ua)
    V_u_r = -beta_r * np.log(alfa_r + Ur)
    V_phi = V_u_r + V_u_a + Vrho
    gradient_x_rho = gradientX(rho)
    gradient_y_rho = gradientY(rho)
    drho = gradientX(rho * gradientX(V_phi) + sigma * gradient_x_rho) + gradientY(
        rho * gradientY(V_phi) + sigma * gradient_y_rho)
    dUa = -gamma_a * Ua + D_a * laplacian(Ua) + s_a * rho
    dUr = -gamma_r * Ur + D_r * laplacian(Ur) + s_r * rho
    return drho, dUa, dUr


#args = [gen, i, parameter_dir]
def run(args=None):
    global sigma, scale, rho_max, cushion, dt, beta_a, beta_r, alfa_a, alfa_r, D_a, D_r, gamma_a, gamma_r, s_a, s_r
    if args is None:
        directory = "../swarming_results/sim_1"
        while os.path.isdir(directory):
            directory = "../swarming_results/sim_" + str(int(directory.split("/")[-1].split("_")[-1]) + 1)
    else:
        gen = str(args[0])
        i = str(args[1])
        now = datetime.now()

        # Format as string
        now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

        # Use this string in your directory name
        directory = f"swarming_results_optimisation/sim_{now_str}/gen_{gen}/ind_{i}"
        while os.path.isdir(directory):
            now = datetime.now()

            # Format as string
            now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
            directory =  f"swarming_results_optimisation/sim_{now_str}/gen_{gen}/ind_{i}"
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
            sigma, scale, rho_max, cushion, dt, beta_a, beta_r, alfa_a, alfa_r, D_a, D_r, gamma_a, gamma_r, s_a, s_r = updateParameters(
                parameter_dir)
        else:
            dt = 0.01
            sigma, scale, rho_max, cushion, beta_a, beta_r, alfa_a, alfa_r, D_a, D_r, gamma_a, gamma_r, s_a, s_r, rho0 = updateParameters(
                parameter_dir, ":")

    rho, Ua, Ur, t, step = initial_conditions(rho0, s_a, gamma_a, s_r, gamma_r)

    if not os.path.isdir(directory):
        os.makedirs(directory)
    # Main loop
    pbar = tqdm(total=step_max)
    success = False
    sigma_times_scale = sigma * scale
    while step < step_max:

        if time_integration == "euler":

            Vrho = sigma_times_scale * (1 + np.tanh((rho - rho_max) / cushion))

            if np.max(alfa_a + Ua) > 10e17 or np.min(alfa_a + Ua) < 0:
                print("oh oh:")
                print(np.max(alfa_a + Ua))
                print(np.min(alfa_a + Ua))
                Exception("Ua is negative or too large")
                break

            '''
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_V_u_a = executor.submit(lambda: -beta_a * np.log(alfa_a + Ua))
                future_V_u_r = executor.submit(lambda: -beta_r * np.log(alfa_r + Ur))
                future_gradient_x_rho = executor.submit(lambda: gradientX(rho))
                future_gradient_y_rho = executor.submit(lambda: gradientY(rho))

                V_u_a = future_V_u_a.result()
                V_u_r = future_V_u_r.result()
                gradient_x_rho = future_gradient_x_rho.result()
                gradient_y_rho = future_gradient_y_rho.result()'''

            V_u_a = -beta_a * np.log(alfa_a + Ua)
            V_u_r = -beta_r * np.log(alfa_r + Ur)
            V_phi = V_u_r + V_u_a + Vrho

            gradient_x_rho = gradientX(rho)
            gradient_y_rho = gradientY(rho)
            '''with concurrent.futures.ThreadPoolExecutor() as executor:
                future_drho = executor.submit(lambda: gradientX(rho * gradientX(V_phi) + sigma * gradient_x_rho) + gradientY(rho * gradientY(V_phi) + sigma * gradient_y_rho))
                future_dUa = executor.submit(lambda: -gamma_a * Ua + D_a * laplacian(Ua) + s_a * rho)
                future_dUr = executor.submit(lambda: -gamma_r * Ur + D_r * laplacian(Ur) + s_r * rho)

                drho = future_drho.result()
                dUa = future_dUa.result()
                dUr = future_dUr.result()'''
            drho = gradientX(rho * gradientX(V_phi) + sigma * gradient_x_rho) + gradientY(
                rho * gradientY(V_phi) + sigma * gradient_y_rho)

            dUa = -gamma_a * Ua + D_a * laplacian(Ua) + s_a * rho

            dUr = -gamma_r * Ur + D_r * laplacian(Ur) + s_r * rho
            rho = (drho) * dt + rho
            Ua = dUa * dt + Ua
            Ur = dUr * dt + Ur

            #check for non numerical values
            if np.any(np.isnan(rho)) or np.any(np.isnan(Ua)) or np.any(np.isnan(Ur)):
                Exception("Non numerical values")
                break

            #check for infinities
            if np.any(rho) > 10e16:
                Exception("Rho is too large")
                break

        else:  # Runge-Kutta 4th order
            # Calculate k1
            drho1, dUa1, dUr1 = derivatives(rho, Ua, Ur)
            k1_rho = dt * drho1
            k1_Ua = dt * dUa1
            k1_Ur = dt * dUr1

            # Calculate k2
            drho2, dUa2, dUr2 = derivatives(rho + 0.5 * k1_rho, Ua + 0.5 * k1_Ua, Ur + 0.5 * k1_Ur)
            k2_rho = dt * drho2
            k2_Ua = dt * dUa2
            k2_Ur = dt * dUr2

            # Calculate k3
            drho3, dUa3, dUr3 = derivatives(rho + 0.5 * k2_rho, Ua + 0.5 * k2_Ua, Ur + 0.5 * k2_Ur)
            k3_rho = dt * drho3
            k3_Ua = dt * dUa3
            k3_Ur = dt * dUr3

            # Calculate k4
            drho4, dUa4, dUr4 = derivatives(rho + k3_rho, Ua + k3_Ua, Ur + k3_Ur)
            k4_rho = dt * drho4
            k4_Ua = dt * dUa4
            k4_Ur = dt * dUr4

            # Update the variables
            drho = (k1_rho + 2 * k2_rho + 2 * k3_rho + k4_rho) / 6
            rho += drho
            Ua += (k1_Ua + 2 * k2_Ua + 2 * k3_Ua + k4_Ua) / 6
            Ur += (k1_Ur + 2 * k2_Ur + 2 * k3_Ur + k4_Ur) / 6

        #add 10e7 (100/mm^2 = 10e7/mm^2) to Ua in the target area
        Ua[236:276, 236:276] += 10e9

        #force Ua and Ur positive
        Ua[Ua < 0] = 0
        Ur[Ur < 0] = 0
        rho[rho < 0] = 0

        #do not let Ua and Ur explode:
        Ua[Ua > 10e16] = 10e16
        Ur[Ur > 10e16] = 10e16

        t = t + dt

        eps = np.max(np.abs(drho))
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
            #save matrices every 10000 steps
            np.save(directory + f"/rho_{step}.npy", rho)
            np.save(directory + f"/Ua_{step}.npy", Ua)
            np.save(directory + f"/Ur_{step}.npy", Ur)

        if eps < eps_min or step == step_max - 1:
            rho_matrices.append(rho)
            # save matrices every 10000 steps
            np.save(directory + f"/rho_{step}.npy", rho)
            np.save(directory + f"/Ua_{step}.npy", Ua)
            np.save(directory + f"/Ur_{step}.npy", Ur)
            print("saving in "+ directory + f"/rho_{step}.npy")
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
        ani.save('rho_evolution_oxy_40.mp4', writer='ffmpeg', fps=1, dpi=100)

    #show(rho)
    if success:
        #from clustering2 import evaluate
        #c, kurt, bins, ys = evaluate(directory + f"/rho_{step}.npy", 50)
        c = rho[236:276, 236:276].sum() / rho.sum()
        print(c)
    else:
        c=0
        t = step_max*dt
    return c, t


#main([0, 0, "parameters_swarming_optimised.json"])

#load matrix from swarming_results/sim_1/rho_{step}.npy and show it
def show_matrix(dir_):
    rho = np.load(dir_)
    show(rho)

#show_matrix()
if __name__ == "__main__":
    arg = sys.argv[1]
    if arg=="run":
        run()
    #'''
    else:
        from clustering2 import evaluate
        directory=f"../swarming_results/sim_{int(arg)}"
        #step=2091
        #c, kurt, bins, ys = evaluate(directory + f"/rho_{step}.npy", 50)
        #print(c)
        #show(np.load(directory + f"/rho_{step}.npy"))

        def build_video(dir_):
            rho_matrices = []
            for file in os.listdir(dir_):
                if file.endswith(".npy") and "rho" in file:
                    rho_matrices.append(np.load(dir_ + "/" + file))
            fig, ax = plt.subplots()
            cax = ax.imshow(rho_matrices[0], cmap='magma', interpolation='nearest')

            def update(frame):
                cax.set_array(rho_matrices[frame])
                return [cax]
            #show(rho_matrices[-1])
            ani = animation.FuncAnimation(fig, update, frames=len(rho_matrices), blit=True)
            ani.save('rho_evolution_oxy_40.mp4', writer='ffmpeg', fps=20, dpi=100)
            #get the sum of the densities inside of the target area over the total sum of the densities
            c = rho_matrices[-1][236:276, 236:276].sum()/rho_matrices[-1].sum()
            print(c)
        build_video(directory)#'''