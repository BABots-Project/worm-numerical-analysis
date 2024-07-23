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
from swarming_simulator import gradientX, gradientY, laplacian, show

PARAMETERS_FILE = "best_odor.txt"

l = 0.02
N = 512
dx = l / N
indices = np.array([[(j - 2) % N, (j - 1) % N, (j + 1) % N, (j + 2) % N] for j in range(N)])
dx12 = 12 * dx
dxdx = dx ** 2
dt = 0.01
STEP_MAX = 500000

ATTRACTANT = False
REPELLENT = False
V_RHO = False
RHO_0_FROM_FILE = True
NUMBER_OF_SPOTS = 2
NUMBER_OF_SPAWNS = 1
SIZE_OF_A = 160 #TOP RIGHT -- sizes are 2x (total size of spawn)
SIZE_OF_B = 160 #BOTTOM LEFT
SIZE_OF_SPAWN = 160 #BOTTOM RIGHT + TOP LEFT

def get_clustering_metric(rho):
    if NUMBER_OF_SPAWNS == 1:
        c = rho[236:276, 236:276].sum() / rho[-1].sum()
    else:
        top_right_index_x1, top_right_index_x2, top_right_index_y1, top_right_index_y2 = get_indices_top_right_quadrant(
            SIZE_OF_A)
        bottom_left_index_x1, bottom_left_index_x2, bottom_left_index_y1, bottom_left_index_y2 = get_indices_bottom_left_quadrant(
            SIZE_OF_B)
        c = -rho[top_right_index_x1:top_right_index_x2,
             top_right_index_y1:top_right_index_y2].sum() / rho.sum() - rho[bottom_left_index_x1:bottom_left_index_x2,bottom_left_index_y1:bottom_left_index_y2].sum() / \
            rho.sum()

    return c

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

def get_indices_top_left_quadrant(size_of_spawn):
    #start by defining the center point of the top-right quadrant:
    top_left_center = int(N/4), int(N/4)
    #define the area centered at the top-right quadrant center + the size of the spawn
    index_spawn_x1 = int(top_left_center[0] - size_of_spawn/2) #N/4 - N/4 = 0 | N/4-N/8 = N/8
    index_spawn_x2 = int(top_left_center[0] + size_of_spawn/2) #3N/4 + N/4 = N |
    index_spawn_y1 = int(top_left_center[1] - size_of_spawn/2) #3N/4 - N/4 = N/2
    index_spawn_y2 = int(top_left_center[1] + size_of_spawn/2) #3N/4 + N/4 = 5N/4
    return index_spawn_x1, index_spawn_x2, index_spawn_y1, index_spawn_y2

def get_indices_bottom_right_quadrant(size_of_spawn):
    #start by defining the center point of the top-right quadrant:
    bottom_right_center = int(3*N/4), int(3*N/4)
    #define the area centered at the top-right quadrant center + the size of the spawn
    index_spawn_x1 = int(bottom_right_center[0] - size_of_spawn/2) #N/4 - N/4 = 0 | N/4-N/8 = N/8
    index_spawn_x2 = int(bottom_right_center[0] + size_of_spawn/2) #3N/4 + N/4 = N |
    index_spawn_y1 = int(bottom_right_center[1] - size_of_spawn/2) #3N/4 - N/4 = N/2
    index_spawn_y2 = int(bottom_right_center[1] + size_of_spawn/2) #3N/4 + N/4 = 5N/4
    return index_spawn_x1, index_spawn_x2, index_spawn_y1, index_spawn_y2

def get_indices_top_right_quadrant(size_of_a):
    #start by defining the center point of the top-right quadrant:
    top_right_center = int(N/4), int(3*N/4)
    #define the area centered at the top-right quadrant center + the size of the spawn
    index_spawn_x1 = int(top_right_center[0] - size_of_a/2) #N/4 - N/4 = 0 | N/4-N/8 = N/8
    index_spawn_x2 = int(top_right_center[0] + size_of_a/2) #3N/4 + N/4 = N |
    index_spawn_y1 = int(top_right_center[1] - size_of_a/2) #3N/4 - N/4 = N/2
    index_spawn_y2 = int(top_right_center[1] + size_of_a/2) #3N/4 + N/4 = 5N/4
    return index_spawn_x1, index_spawn_x2, index_spawn_y1, index_spawn_y2

def get_indices_bottom_left_quadrant(size_of_b):
    #start by defining the center point of the top-right quadrant:
    bottom_left_center = int(3*N/4), int(N/4)
    #define the area centered at the top-right quadrant center + the size of the spawn
    index_spawn_x1 = int(bottom_left_center[0] - size_of_b/2) #N/4 - N/4 = 0 | N/4-N/8 = N/8
    index_spawn_x2 = int(bottom_left_center[0] + size_of_b/2) #3N/4 + N/4 = N |
    index_spawn_y1 = int(bottom_left_center[1] - size_of_b/2) #3N/4 - N/4 = N/2
    index_spawn_y2 = int(bottom_left_center[1] + size_of_b/2) #3N/4 + N/4 = 5N/4
    return index_spawn_x1, index_spawn_x2, index_spawn_y1, index_spawn_y2

def initial_conditions_2_spots(rho0, size_of_spawn, size_of_a, size_of_b, ca0, cb0, gamma_a, s_a, gamma_r, s_r):
    rho = np.zeros((N, N))
    c = np.zeros((N, N))

    index_b_x1, index_b_x2, index_b_y1, index_b_y2 = get_indices_bottom_right_quadrant(size_of_spawn)
    rho[index_b_x1:index_b_x2, index_b_y1:index_b_y2] = rho0 * np.random.uniform(0.99, 1.01, (size_of_spawn, size_of_spawn))

    if NUMBER_OF_SPAWNS>1 and False:
        index_a_x1, index_a_x2, index_a_y1, index_a_y2 = get_indices_top_left_quadrant(size_of_spawn)
        rho[index_a_x1:index_a_x2, index_a_y1:index_a_y2] = rho0 * np.random.uniform(0.99, 1.01, (size_of_spawn, size_of_spawn))

    u_a = s_a * gamma_a * rho
    u_r = s_r * gamma_r * rho

    index_spawn_x1, index_spawn_x2, index_spawn_y1,index_spawn_y2 = get_indices_top_right_quadrant(size_of_a)
    c[index_spawn_x1:index_spawn_x2, index_spawn_y1:index_spawn_y2] = ca0 * np.random.uniform(0.99, 1.01, (size_of_a, size_of_a))

    index_spawn_x1, index_spawn_x2, index_spawn_y1, index_spawn_y2 = get_indices_bottom_left_quadrant(size_of_b)
    c[index_spawn_x1:index_spawn_x2, index_spawn_y1:index_spawn_y2] = cb0 * np.random.uniform(0.99, 1.01, (size_of_b, size_of_b))

    return rho, c, u_a, u_r, 0, 0

def initial_conditions_from_center(rho0, gamma_a, s_a, gamma_r, s_r, b_start_y=132):
    #place worms in the center of the arena in a 40x40 square
    rho = np.zeros((N, N))
    U = np.zeros((N, N))
    rho[236:276, 236:276] = (rho0 + rho0 * random.uniform(-0.01, 0.01))
    #place U
    #spot A; fixed
    U[280:320, 360:400] = rho0
    #sopt B; moving only on y
    U[280:320, b_start_y-20:b_start_y+20] = rho0
    #show(U, ".")

    u_a = s_a * gamma_a * rho
    u_r = s_r * gamma_r * rho

    return rho, U, u_a, u_r, 0, 0

def save_parameters(directory, rho0):
    with open(directory + "/parameters.txt", "w") as file:
        file.write(f"rho0={rho0}\n")
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

sigma, gamma, beta, alpha, D, beta_a, alpha_a, D_a, gamma_a, s_a, beta_r, alpha_r, D_r, gamma_r, s_r, scale, rho_max, cushion = updateParameters(PARAMETERS_FILE)

def run(params, args=None):
    b_start, custom_D = params
    print("starting simulation with b_start=", b_start)
    global sigma, gamma, beta, alpha, D, beta_a, alpha_a, D_a, gamma_a, s_a, beta_r, alpha_r, D_r, gamma_r, s_r, dt
    if args is None:
        directory = "../my_swarming_results/distance/b_start_" + str(b_start) + "/D_" + str(custom_D)
        #while os.path.isdir(directory):
        #    directory = "../my_swarming_results/sim_" + str(int(directory.split("/")[-1].split("_")[-1]) + 1)
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
    rho0 = 55417298.26084167
    if RHO_0_FROM_FILE:
        with open(PARAMETERS_FILE, 'r') as file:
            lines = file.readlines()
            rho0 = float(lines[-1].split("=")[1])
    #rho0 = 55417298.26084167
    step_max = STEP_MAX
    logging = True
    make_video = False
    # Read parameters from text file
    rho_matrices = []
    if args is not None:
        parameter_dir = args[2]
        if not parameter_dir.endswith(".json"):
            sigma, gamma, dc, k0, D, Dc, kc = updateParameters(
                parameter_dir)
    if NUMBER_OF_SPOTS == 1:
        rho, U, Ua, Ur, t, step = initial_conditions(rho0, gamma_a, s_a, gamma_r, s_r)
    else:
        size_of_spawn = SIZE_OF_SPAWN
        size_of_a = SIZE_OF_A
        size_of_b = SIZE_OF_B
        ca0 = rho0
        cb0 = rho0
        rho, U, Ua, Ur, t, step = initial_conditions_2_spots(rho0, size_of_spawn, size_of_a, size_of_b, ca0, cb0, gamma_a, s_a, gamma_r, s_r)

    rho, U, Ua, Ur, t, step = initial_conditions_from_center(rho0, gamma_a, s_a, gamma_r, s_r, b_start)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    #save parameters in the directory
    save_parameters(directory, rho0)
    # Main loop
    pbar = tqdm(total=step_max)
    success = False
    sigma_times_scale = sigma * scale

    D = custom_D
    while step < step_max:

        if time_integration == "euler":

            gradient_x_rho = gradientX(rho)
            gradient_y_rho = gradientY(rho)
            #gradient_x_U = gradientX(U)
            #gradient_y_U = gradientY(U)
            V = -beta * np.log(alpha + U)# - beta_a * np.log(alpha_a + Ua) - beta_r * np.log(alpha_r + Ur) + sigma_times_scale*(1+np.tanh((rho - rho_max)/cushion))
            if ATTRACTANT:
                V-= beta_a * np.log(alpha_a + Ua)
                dUa = -gamma_a * Ua + D_a * laplacian(Ua) + s_a * rho
                Ua = dUa * dt + Ua
            if REPELLENT:
                V-= beta_r * np.log(alpha_r + Ur)
                dUr = -gamma_r * Ur + D_r * laplacian(Ur) + s_r * rho
            if V_RHO:
                V+= sigma_times_scale*(1+np.tanh((rho - rho_max)/cushion))
                Ur = dUr * dt + Ur
            drho = gradientX(rho * gradientX(V) + sigma * gradient_x_rho) + gradientY(
                rho * gradientY(V) + sigma * gradient_y_rho)

            dU = -gamma * U + D * laplacian(U)



            rho = (drho) * dt + rho
            #add small noise to rho where rho>0:
            #rho[rho>0] += rho[rho>0] * random.uniform(-0.001, 0.001)
            U = dU * dt + U



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

        if eps>10e16 or np.max(rho)>10e16:
            Exception("Rho is too large")
            break
        '''
        scaled_rho = 10e7 * rho
        comparison_a = Ua > scaled_rho
        comparison_r = Ur > scaled_rho

        if np.any(comparison_a) or np.any(comparison_r):
            break
        '''

        if logging and step % 10000 == 0:
            if make_video:
                rho_matrices.append(rho)
            # save matrices every 10000 steps
            np.save(directory + f"/rho_{step}.npy", rho)
            np.save(directory + f"/U_{step}.npy", U)
            if ATTRACTANT:
                np.save(directory + f"/Ua_{step}.npy", Ua)
            if REPELLENT:
                np.save(directory + f"/Ur_{step}.npy", Ur)



        if eps < eps_min or step == step_max - 1:
            rho_matrices.append(rho)
            # save matrices every 10000 steps
            np.save(directory + f"/rho_{step}.npy", rho)
            np.save(directory + f"/U_{step}.npy", U)
            if ATTRACTANT:
                np.save(directory + f"/Ua_{step}.npy", Ua)
            if REPELLENT:
                np.save(directory + f"/Ur_{step}.npy", Ur)
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
        ani.save('rho_evolution_oxy_40.mp4', writer='ffmpeg', fps=1, dpi=100)

    #show(rho, directory)
    #save in directory the pdf of the final density
    save_final_plot(rho, directory)

    if success:
        # from clustering2 import evaluate
        # c, kurt, bins, ys = evaluate(directory + f"/rho_{step}.npy", 50)
        c = get_clustering_metric(rho)
        print(c)
    else:
        c = -1
        t = step_max * dt

    return c, t

def evalute_two_spots():
    c_a_list = []
    c_b_list = []
    for b_start in range(20, 256, 20):
        # laod matrix from ../my_swarming_results/distance/b_start_{b_start}/rho_499999.npy
        rho = np.load(f"../my_swarming_results/distance/b_start_{b_start}/rho_499999.npy")
        c_a = rho[280:320, 360:400].sum() / rho.sum()
        c_b = rho[280:320, b_start - 20:b_start + 20].sum() / rho.sum()
        c_a_list.append(c_a)
        c_b_list.append(c_b)
    plt.plot(list(range(20, 256, 20)), c_a_list, label="A")
    plt.plot(list(range(20, 256, 20)), c_b_list, label="B")
    plt.legend()
    plt.show()


def evalute_two_spots_diffusion(b_start):
    c_a_list = []
    c_b_list = []
    max_d = 4.47 * 10 ** -9
    min_d = 1.12 * 10 ** -9
    delta_d = (max_d - min_d) / 10
    d_list = list(np.arange(min_d, max_d, delta_d))
    for custom_D in d_list:
        # laod matrix from ../my_swarming_results/distance/b_start_{b_start}/rho_499999.npy
        rho = np.load(f"../decision_making/2spots/b_start_{b_start}/D_{custom_D}/rho_499999.npy")
        c_a = rho[280:320, 360:400].sum() / rho.sum()
        c_b = rho[280:320, b_start - 20:b_start + 20].sum() / rho.sum()
        c_a_list.append(c_a)
        c_b_list.append(c_b)
    plt.plot(d_list, c_a_list, label="A")
    plt.plot(d_list, c_b_list, label="B")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #evalute_two_spots_diffusion(int(sys.argv[1]))


    #sys.exit()
    arg = sys.argv[1]
    if arg=="run":

        b_starts = list(range(20, 256, 20))
        max_d = 4.47*10**-9
        min_d = 1.12*10**-9
        delta_d = (max_d - min_d) / 10
        d_list = list(np.arange(min_d, max_d, delta_d))
        params = [(b_start, d) for b_start in b_starts for d in d_list]
        print(params)
        with multiprocessing.Pool() as pool:
            results = pool.map(run, params)
        #run()
    #
    else:
        from clustering2 import evaluate, create_matrix_from_tsv

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
            c = get_clustering_metric(rho_matrices[-1])
            c1 = rho_matrices[-1][280:320, 360:400].sum() / rho_matrices[-1].sum()
            c2 = rho_matrices[-1][280:320, 360-265:400-265].sum() / rho_matrices[-1].sum()
            print(c1)
            print(c2)
            fig, ax = plt.subplots()
            #set minimum value of all rho matrices to 10e3
            #for i in range(len(rho_matrices)):
            #    rho_matrices[i][rho_matrices[i] < 10e3] = 10e3
            colors = ['#FFFCF4', '#E6A76C', '#D8524E', '#8F1D4D', '#3B0A3A']

            # Create a list of positions from 0 to 1
            positions = [0.0, 0.25, 0.5, 0.75, 1.0]
            import matplotlib.colors as mcolors
            from matplotlib.colors import LinearSegmentedColormap
            # Create a colormap object
            cmap = mcolors.LinearSegmentedColormap.from_list('custom_magma', list(zip(positions, colors)))
            vmin = np.min(rho_matrices[0])+10**-5
            cax = ax.imshow(rho_matrices[0], cmap='magma', interpolation='nearest')#, norm=LogNorm(vmin=vmin, vmax=np.max(rho_matrices[0])))
            fig.patch.set_facecolor('#FFFCF4')
            ax.set_facecolor('#FFFCF4')
            #remap x and y ticks to go from 0 to 10
            ax.set_xticks(np.linspace(0, 512, 11))
            ax.set_yticks(np.linspace(0, 512, 11))
            ax.set_xticklabels(np.linspace(0, 20, 11))
            ax.set_yticklabels(np.linspace(0, 20, 11))
            #reverse y axis
            ax.invert_yaxis()
            def update(frame):
                cax.set_array(rho_matrices[frame])
                return [cax]
            #show(rho_matrices[-1])
            ani = animation.FuncAnimation(fig, update, frames=len(rho_matrices), blit=True)
            ani.save(dir_+'/0rho_evolution.mp4', writer='ffmpeg', fps=20, dpi=100)
            #get the sum of the densities inside of the target area over the total sum of the densities

            print(c)
        if video:
            build_video(directory)
        else:
            j=499
            m = np.load(directory + f"/rho_{j}.npy")
            c = get_clustering_metric(m)
            #print(c)
            show(m, directory)

            for j in [0, 250000, 499999]:
                m = np.load(directory + f"/rho_{j}.npy")
                c = get_clustering_metric(m)
                print(c)
                show(m, directory)
        # #'''
