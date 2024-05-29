import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import os, sys
from tqdm import tqdm
from numba import jit, vectorize, float32, njit, stencil, prange
import cProfile

N=512
l=0.01
dx=l/N
indices = np.array([[(j - 2)%N, (j - 1)%N, (j + 1)%N, (j + 2)%N] for j in range(N)])
dx12 = 12 * dx
dxdx = dx**2

@njit(fastmath=True)
def fourth_order_derivative(in_array):
    out = np.zeros_like(in_array)

    for i in range(N):
        for j in range(N):
            # Compute shifted versions
            left = in_array[i, (j-1)%N]
            right = in_array[i, (j+1)%N]
            up = in_array[(i-1)%N, j]
            down = in_array[(i+1)%N, j]
            left_2 = in_array[i, (j-2)%N]
            right_2 = in_array[i, (j+2)%N]
            up_2 = in_array[(i-2)%N, j]
            down_2 = in_array[(i+2)%N, j]
            up_left = in_array[(i-1)%N, (j-1)%N]
            up_right = in_array[(i-1)%N, (j+1)%N]
            down_left = in_array[(i+1)%N, (j-1)%N]
            down_right = in_array[(i+1)%N, (j+1)%N]
            
            # Compute fourth-order derivative
            out[i, j] = (20 * in_array[i, j] - 8 * left - 8 * right - 8 * up - 8 * down +
                         left_2 + right_2 + up_2 + down_2 +
                         2 * up_left + 2 * up_right + 2 * down_left + 2 * down_right) / (dx**4)
            
    return out


''''@njit(fastmath=True)
def gradientX(in_array, dx):
    out = np.zeros_like(in_array)
    N, N = in_array.shape
    dx12 = 12 * dx
    for i in range(N):
        for j in range(N):
            # Compute shifted versions
            left_2 = in_array[i, (j-2)%N]
            left_1 = in_array[i, (j-1)%N]
            right_1 = in_array[i, (j+1)%N]
            right_2 = in_array[i, (j+2)%N]
            
            # Compute gradient
            out[i, j] = (-left_2 + 8 * left_1 - 8 * right_1 + right_2) / dx12
            
    return out'''


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


def initial_conditions(wormDensity, Oam, f, kc, n, rho0,s_a, gamma_a, s_r, gamma_r, eta):
    W = (1-eta)*(rho0 * np.ones((n, n)) + rho0 * (np.random.rand(n, n) - 0.5) / 50)
    O = np.ones((n, n)) * (Oam - (kc / f) * rho0)
    rho = eta*(rho0 * np.ones((n, n)) + rho0 * (np.random.rand(n, n) - 0.5) / 50)
    Ua =s_a*gamma_a*rho
    Ur = s_r*gamma_r*rho
    fig = plt.figure(figsize=(15, 9))
    t = 0
    step = 0
    return W, O, rho, Ua, Ur, t, step

def visualizeGrid(O, W, visualizationPeriod, step, L=None):
    if step % visualizationPeriod == 0:
        plt.subplot(1, 2, 1)
        plt.imshow(resize(O / np.max(O), (O.shape[0]*5, O.shape[1]*5)))
        plt.colorbar()
        plt.title('Oxygen')
        
        plt.subplot(1, 2, 2)
        plt.imshow(resize(W / np.max(W), (W.shape[0]*5, W.shape[1]*5)))
        plt.colorbar()
        plt.title(f'Worms Total Worms: {np.sum(W)}')
        
        plt.pause(0.001)
        
        if L:
            plt.savefig(L)  # Save the figure if the filename is provided
        else:
            plt.show()

def save_matrix_to_tsv(matrix, dx, filename):
    """
    Save a matrix to a tab-separated values (TSV) file with format: x, y, value.

    Parameters:
        matrix (numpy.ndarray): Matrix to save.
        dx (float): Grid spacing.
        filename (str): Name of the TSV file to save.
    """
    with open(filename, 'w') as file:
        # Write data
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                x = i * dx
                y = j * dx
                value = matrix[i, j]
                file.write(f'{x}\t{y}\t{value}\n')

def updateParameters(textFileName):
    with open(textFileName, 'r') as file:
        lines = file.readlines()
        sigma = float(lines[0].split('=')[1])
        scale = float(lines[1].split('=')[1])
        rho_max = float(lines[2].split('=')[1])
        cushion = float(lines[3].split('=')[1])
        dt = float(lines[4].split('=')[1])
        K = float(lines[5].split('=')[1])
        beta_a = float(lines[6].split('=')[1])
        beta_r = float(lines[7].split('=')[1])
        alfa_a = float(lines[8].split('=')[1])
        alfa_r = float(lines[9].split('=')[1])
        D_a = float(lines[10].split('=')[1])
        D_r = float(lines[11].split('=')[1])
        gamma_a = float(lines[12].split('=')[1])
        gamma_r = float(lines[13].split('=')[1])
        s_a = float(lines[14].split('=')[1])
        s_r = float(lines[15].split('=')[1])
        Do = float(lines[16].split('=')[1])
        Oam = float(lines[17].split('=')[1])
        f = float(lines[18].split('=')[1])
        kc = float(lines[19].split('=')[1])
        
    return sigma, scale, rho_max, cushion, dt, K, beta_a, beta_r, alfa_a, alfa_r, D_a, D_r, gamma_a, gamma_r, s_a, s_r, Do, Oam, f, kc

@njit(fastmath=True)
def pdeSolver(h, oxy_params, n, rho, U_a, U_r, dt, dx, sigma, scale, rho_max, cushion, beta_a, alfa_a, beta_r, alfa_r, D_a, D_r, gamma_a, gamma_r, s_a, s_r, t, K):
    W, O, Do, kc, f, Oam, a, b, c, T = oxy_params
    V_O = a * O ** 2 + b * O + c
    dVdO = 2 * a * O + b
    D = V_O ** 2 / T
    beta = -V_O * dVdO / T
    
    Vrho = sigma*scale*(1+np.tanh((rho-rho_max)/cushion))
    alfa_a=alfa_a*np.ones((n,n))
    alfa_r=alfa_r*np.ones((n,n))
    
    V_u_a = -beta_a*np.log(alfa_a+U_a)
    V_u_r = -beta_r*np.log(alfa_r+U_r)
    V_phi = Vrho+V_u_r+V_u_a
    non_local_rho = K * fourth_order_derivative(rho)
    diffusion = gradientX(D * gradientX(rho)) + gradientY(D * gradientY(rho))
    taxis = gradientX(beta * rho * gradientX(O)) + gradientY(beta * rho * gradientY(O))
    drho_oxygen = diffusion - taxis - non_local_rho

    drho_pheromone = gradientX(rho*gradientX(V_phi)+sigma*gradientX(rho))+gradientY(rho*gradientY(V_phi)+sigma*gradientY(rho))

    drho = h * drho_pheromone + (1-h) * drho_oxygen / 100
    dU_a = -gamma_a*U_a+D_a*laplacian(U_a)+s_a*rho
    dU_r = -gamma_r*U_r+D_r*laplacian(U_r)+s_r*rho
    rho = (drho-non_local_rho) * dt + rho
    U_a = dU_a * dt + U_a
    U_r = dU_r * dt + U_r
    #W = (diffusion - taxis - non_local) * dt + W
    O = (Do * laplacian(O) - kc * (rho) * (O / (O + 0.0001)) + f * (Oam - O)) * dt + O
    t = t + dt
    
    return W, O, rho, U_a, U_r, t

@njit(fastmath=True)
def bulk_gradientX(gradient_x_O_1, gradient_x_O_2, gradient_x_phi):
    '''
    :param gradient_x_O_1: first argument of gradient of O in the x direction
    :param gradient_x_O_2: second argument of gradient of O in the x direction
    :param gradient_x_phi: argument of gradient of phi in the x direction
    :return: out1,out2,out3: gradients of the first argument, second argument and third argument of the function in the x direction
    basically allows to compute the gradient of a matrix of arrays, leveraging the parallelization of the calculation for the second iteration of the gradient
    '''

    out1 = np.zeros_like(gradient_x_O_1)
    out2 = np.zeros_like(gradient_x_O_2)
    out3 = np.zeros_like(gradient_x_phi)

    for i in range(N):
        for j in range(N):
            # Compute shifted versions using pre-computed indices
            left_2O1 = gradient_x_O_1[i, indices[j][0]]
            left_1O1 = gradient_x_O_1[i, indices[j][1]]
            right_1O1 = gradient_x_O_1[i, indices[j][2]]
            right_2O1 = gradient_x_O_1[i, indices[j][3]]

            left_2O2 = gradient_x_O_2[i, indices[j][0]]
            left_1O2 = gradient_x_O_2[i, indices[j][1]]
            right_1O2 = gradient_x_O_2[i, indices[j][2]]
            right_2O2 = gradient_x_O_2[i, indices[j][3]]

            left_2phi = gradient_x_phi[i, indices[j][0]]
            left_1phi = gradient_x_phi[i, indices[j][1]]
            right_1phi = gradient_x_phi[i, indices[j][2]]
            right_2phi = gradient_x_phi[i, indices[j][3]]

            # Compute gradient
            out1[i, j] = (-left_2O1 + 8 * left_1O1 - 8 * right_1O1 + right_2O1) / dx12
            out2[i, j] = (-left_2O2 + 8 * left_1O2 - 8 * right_1O2 + right_2O2) / dx12
            out3[i, j] = (-left_2phi + 8 * left_1phi - 8 * right_1phi + right_2phi) / dx12

    return out1, out2, out3


@njit(fastmath=True)
def bulk_gradientY(gradient_y_O_1, gradient_y_O_2, gradient_y_phi):
    '''
    :param gradient_y_O_1: first argument of gradient of O in the y direction
    :param gradient_y_O_2: second argument of gradient of O in the y direction
    :param gradient_y_phi: argument of gradient of phi in the y direction
    :return: out1,out2,out3: gradients of the first argument, second argument and third argument of the function in the y direction
    basically allows to compute the gradient of a matrix of arrays, leveraging the parallelization of the calculation for the second iteration of the gradient
    '''
    out1 = np.zeros_like(gradient_y_O_1)
    out2 = np.zeros_like(gradient_y_O_2)
    out3 = np.zeros_like(gradient_y_phi)

    for i in range(N):
        for j in range(N):
            # Compute shifted versions
            up_2O1 = gradient_y_O_1[indices[i][0], j]
            up_1O1 = gradient_y_O_1[indices[i][1], j]
            down_1O1 = gradient_y_O_1[indices[i][2], j]
            down_2O1 = gradient_y_O_1[indices[i][3], j]

            up_2O2 = gradient_y_O_2[indices[i][0], j]
            up_1O2 = gradient_y_O_2[indices[i][1], j]
            down_1O2 = gradient_y_O_2[indices[i][2], j]
            down_2O2 = gradient_y_O_2[indices[i][3], j]

            up_2phi = gradient_y_phi[indices[i][0], j]
            up_1phi = gradient_y_phi[indices[i][1], j]
            down_1phi = gradient_y_phi[indices[i][2], j]
            down_2phi = gradient_y_phi[indices[i][3], j]

            # Compute gradient
            out1[i, j] = (-up_2O1 + 8 * up_1O1 - 8 * down_1O1 + down_2O1) / dx12
            out2[i, j] = (-up_2O2 + 8 * up_1O2 - 8 * down_1O2 + down_2O2) / dx12
            out3[i, j] = (-up_2phi + 8 * up_1phi - 8 * down_1phi + down_2phi) / dx12

    return out1, out2, out3

@njit(fastmath=True)
def bulk_laplacian(laplacian_Ur, laplacian_Ua, laplacian_O):
    '''
    :param laplacian_Ur: laplacian of Ur
    :param laplacian_Ua: laplacian of Ua
    :param laplacian_O: laplacian of O
    :return: out1,out2,out3: laplacians of the first argument, second argument and third argument of the function
    basically allows to compute the laplacian of a matrix of arrays, leveraging the parallelization of the calculation for the second iteration of the laplacian
    '''
    out1 = np.zeros_like(laplacian_Ur)
    out2 = np.zeros_like(laplacian_Ua)
    out3 = np.zeros_like(laplacian_O)

    for i in range(N):
        for j in range(N):
            # Compute shifted versions
            centerUr = laplacian_Ur[i, j]
            rightUr = laplacian_Ur[i, indices[j][2]]
            leftUr = laplacian_Ur[i, indices[j][1]]
            upUr = laplacian_Ur[indices[i][1], j]
            downUr = laplacian_Ur[indices[i][2], j]
            up_rightUr = laplacian_Ur[indices[i][1], indices[j][2]]
            up_leftUr = laplacian_Ur[indices[i][1], indices[j][1]]
            down_rightUr = laplacian_Ur[indices[i][2], indices[j][2]]
            down_leftUr = laplacian_Ur[indices[i][2], indices[j][1]]

            # Compute laplacian
            out1[i, j] = -centerUr + 0.20 * (rightUr + leftUr + upUr + downUr) + \
                        0.05 * (up_rightUr + up_leftUr + down_rightUr + down_leftUr)

            # Compute shifted versions
            centerUa = laplacian_Ua[i, j]
            rightUa = laplacian_Ua[i, indices[j][2]]
            leftUa = laplacian_Ua[i, indices[j][1]]
            upUa = laplacian_Ua[indices[i][1], j]
            downUa = laplacian_Ua[indices[i][2], j]
            up_rightUa = laplacian_Ua[indices[i][1], indices[j][2]]
            up_leftUa = laplacian_Ua[indices[i][1], indices[j][1]]
            down_rightUa = laplacian_Ua[indices[i][2], indices[j][2]]
            down_leftUa = laplacian_Ua[indices[i][2], indices[j][1]]

            # Compute laplacian
            out2[i, j] = -centerUa + 0.20 * (rightUa + leftUa + upUa + downUa) + \
                        0.05 * (up_rightUa + up_leftUa + down_rightUa + down_leftUa)

            # Compute shifted versions
            centerO = laplacian_O[i, j]
            rightO = laplacian_O[i, indices[j][2]]
            leftO = laplacian_O[i, indices[j][1]]
            upO = laplacian_O[indices[i][1], j]
            downO = laplacian_O[indices[i][2], j]
            up_rightO = laplacian_O[indices[i][1], indices[j][2]]
            up_leftO = laplacian_O[indices[i][1], indices[j][1]]
            down_rightO = laplacian_O[indices[i][2], indices[j][2]]
            down_leftO = laplacian_O[indices[i][2], indices[j][1]]

            # Compute laplacian
            out3[i, j] = -centerO + 0.20 * (rightO + leftO + upO + downO) + \
                        0.05 * (up_rightO + up_leftO + down_rightO + down_leftO)

    out1 /= dxdx
    out2 /= dxdx
    out3 /= dxdx
    return out1, out2, out3


def main():
    eps_min = 10e-3
    wormDensity = 30e6
    rho0=65e6 #ACTUAL INITIAL DENSITY
    NHeigth = 0.01  # Simulation window size
    gridSize = 512 # Number of bins on the simulation window
    dx = NHeigth / gridSize  # Size of a bin
    step_max=50000
    h=float(sys.argv[1].replace(",","."))
    eta = 1
    # Read parameters from text file
    sigma, scale, rho_max, cushion, dt, K, beta_a, beta_r, alfa_a, alfa_r, D_a, D_r, gamma_a, gamma_r, s_a, s_r, Do, Oam, f, kc = updateParameters('parameters_hetero.txt')
    #print("dt ", dt)
    # Initialization
    W, O ,rho, U_a, U_r, t, step = initial_conditions(wormDensity, Oam, f, kc, gridSize, rho0,s_a, gamma_a, s_r, gamma_r, eta)
    a = 1.89e-2
    b = -3.98e-3
    c = 2.25e-4
    n=gridSize
    T = 0.5  
    L = []
    if not os.path.isdir(f"time_test/results_hetero/rho0_{rho0}_h_{h}/"):
        os.makedirs(f"time_test/results_hetero/rho0_{rho0}_h_{h}/")
    # Main loop
    pbar = tqdm(total=step_max)
    rho_matrices = []
    eps_list=[]
    avg_eps_list=[]
    while step<step_max:
        V_O = a * O ** 2 + b * O + c

        dVdO = 2 * a * O + b
        D = V_O ** 2 / T
        beta = -V_O * dVdO / T
        
        Vrho = sigma*scale*(1+np.tanh((rho-rho_max)/cushion))

        if np.max(alfa_a + U_a) > 10e11 or np.min(alfa_a + U_a) < 0:
            print("oh oh:")
            print(np.max(alfa_a+U_a))
            print(np.min(alfa_a+U_a))
            break

        V_u_a = -beta_a*np.log(alfa_a+U_a)
        V_u_r = -beta_r*np.log(alfa_r+U_r)
        V_phi = V_u_r+V_u_a+Vrho
        non_local_rho = K * fourth_order_derivative(rho)
        gradient_x_rho = gradientX(rho)
        gradient_y_rho = gradientY(rho)



        diffusion = gradientX(D * gradient_x_rho) + gradientY(D * gradient_y_rho)
        taxis = gradientX(beta * rho * gradientX(O)) + gradientY(beta * rho * gradientY(O))
        drho_oxygen = diffusion - taxis - non_local_rho

        drho_pheromone = gradientX(rho*gradientX(V_phi)+sigma*gradient_x_rho)+gradientY(rho*gradientY(V_phi)+sigma*gradient_y_rho)

        drho = h * drho_pheromone + (1-h) * drho_oxygen #/ 10

        #drho = gradientX((h*sigma+(1-h)*D)*gradient_x_rho + h*rho*gradientX(V_phi)-(1-h)*beta*rho*gradientX(O))+gradientY((h*sigma+(1-h)*D)*gradient_y_rho + h*rho*gradientY(V_phi)-(1-h)*beta*rho*gradientY(O))

        dU_a = -gamma_a*U_a+D_a*laplacian(U_a)+s_a*rho
        dU_r = -gamma_r*U_r+D_r*laplacian(U_r)+s_r*rho
        rho = (drho) * dt + rho
        U_a = dU_a * dt + U_a
        U_r = dU_r * dt + U_r
        #W = (diffusion - taxis - non_local) * dt + W
        O = (Do * laplacian(O) - kc * (rho) * (O / (O + 0.0001)) + f * (Oam - O)) * dt + O
        t = t + dt

        
        if step%10000==0:
            filenameRho = f"time_test/results_hetero/rho0_{rho0}_h_{h}/rhot_{step}.txt"
            #filenameW = "results_hetero/Wt_"+str(step)+".txt"
            save_matrix_to_tsv(rho, dx, filenameRho)


        eps = np.max(np.abs(drho))
        avg_eps= np.average(np.abs(drho))
        eps_list.append(eps)
        avg_eps_list.append(avg_eps)
        if eps<eps_min:
            break

        
        pbar.update(1)
        #if step%100:
        #    rho_matrices.append(rho.copy())
        scaled_rho = 10e6*rho 
        comparison_a = U_a>scaled_rho
        comparison_r = U_r > scaled_rho
        step += 1
        if np.any(comparison_a) or np.any(comparison_r):
            break
    rho_matrices.append(rho) 
    print("eps: ", eps)
    filenameRho = f"time_test/results_hetero/rho0_{rho0}_h_{h}/rhot_{step}.txt"
    
    save_matrix_to_tsv(rho, dx, filenameRho)
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # Create a figure and axis for the plot
    #fig, ax = plt.subplots()
    #x_values = range(step)

    # Create the plot
    #plt.figure(figsize=(10, 6))

    # Plot eps_list
    #plt.plot(x_values, eps_list, label='Eps List', color='blue')

    # Plot avg_eps_list
    #plt.plot(x_values, avg_eps_list, label='Avg Eps List', color='red') 
    # Initialize the plot with the first frame
    #cax = ax.imshow(rho_matrices[0], cmap='viridis', interpolation='nearest')
    #plt.show()
    # Function to update the plot for each frame
    def update(frame):
        cax.set_array(rho_matrices[frame])
        return [cax]

    # Create the animation
    #ani = animation.FuncAnimation(fig, update, frames=len(rho_matrices), blit=True)

    # Save the animation as a video file or display it
    #ani.save('rho_evolution.mp4', writer='ffmpeg', fps=10, dpi=100)
    L = visualizeGrid(O, rho, 1000, step, L)
    #tpbar.close()

def main_ext():
    wormDensity = 30e6
    rho0=65e6 #ACTUAL INITIAL DENSITY
    NHeigth = 0.01  # Simulation window size
    gridSize = 512 # Number of bins on the simulation window
    dx = NHeigth / gridSize  # Size of a bin
    step_max=500000
    h=float(sys.argv[1].replace(",","."))
    eta = 1
    # Read parameters from text file
    sigma, scale, rho_max, cushion, dt, K, beta_a, beta_r, alfa_a, alfa_r, D_a, D_r, gamma_a, gamma_r, s_a, s_r, Do, Oam, f, kc = updateParameters('parameters_hetero.txt')
    print("dt ", dt)
    # Initialization
    W, O ,rho, U_a, U_r, t, step = initial_conditions(wormDensity, Oam, f, kc, gridSize, rho0,s_a, gamma_a, s_r, gamma_r, eta)
    a = 1.89e-2
    b = -3.98e-3
    c = 2.25e-4
    n=gridSize
    T = 0.5  
    L = []
    if not os.path.isdir(f"results_hetero/rho0_{rho0}_h_{h}/"):
        os.makedirs(f"results_hetero/rho0_{rho0}_h_{h}/")
    # Main loop
    pbar = tqdm(total=step_max)
    while step<step_max:
        oxy_params = W, O, Do, kc, f, Oam, a, b, c, T
        W, O, rho, U_a, U_r, t = pdeSolver(h, oxy_params, n, rho, U_a, U_r, dt, dx, sigma, scale, rho_max, cushion, beta_a, alfa_a, beta_r, alfa_r, D_a, D_r, gamma_a, gamma_r, s_a, s_r, t, K)
        if step%1000==0:
            filenameRho = f"results_hetero/rho0_{rho0}_h_{h}/rhot_{step}.txt"
            #filenameW = "results_hetero/Wt_"+str(step)+".txt"
            filenameO = f"results_hetero/rho0_{rho0}_h_{h}/Ot_{step}.txt"
            filenameUa = f"results_hetero/rho0_{rho0}_h_{h}/Uat_{step}.txt"
            filenameUr = f"results_hetero/rho0_{rho0}_h_{h}/Urt_{step}.txt"
            save_matrix_to_tsv(rho, dx, filenameRho)
            save_matrix_to_tsv(U_a, dx, filenameUa)
            save_matrix_to_tsv(U_r, dx, filenameUr)
            #save_matrix_to_tsv(W, dx, filenameW)
            save_matrix_to_tsv(O, dx, filenameO)
        step += 1
        pbar.update(1)
        

    filenameRho = f"results_hetero/rho0_{rho0}_h_{h}/rhot_{step}.txt"
    filenameW = f"results_hetero/rho0_{rho0}_h_{h}/Wt_{step}.txt"
    filenameO = f"results_hetero/rho0_{rho0}_h_{h}/Ot_{step}.txt"
    filenameUa = f"results_hetero/rho0_{rho0}_h_{h}/Uat_{step}.txt"
    filenameUr = f"results_hetero/rho0_{rho0}_h_{h}/Urt_{step}.txt"
    save_matrix_to_tsv(rho, dx, filenameRho)
    save_matrix_to_tsv(U_a, dx, filenameUa)
    save_matrix_to_tsv(U_r, dx, filenameUr)
    save_matrix_to_tsv(W, dx, filenameW)
    save_matrix_to_tsv(O, dx, filenameO)
    L = visualizeGrid(O, rho, 1000, step, L)
    pbar.close()
main()