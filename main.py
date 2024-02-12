# the following code allows to numerically solve the keller and sagel equations applied to the case of C. elegans
import time


from numba import jit, cuda
# equations are taken from the paper: "Dynamics of pattern formation and emergence of swarming in Caenorhabditis elegans"

import numpy as np
import matplotlib.pyplot as plt

def solve_PDE(W, O, a, b, c, tau, D0, f, kc, Om, D, L, dt, t_start, dest,l,N, t_end=600):
    timestep = 0
    cell_size_1d = l/N
    cell_size_2d = l**2/N**2
    while True:
        Wold = W
        Oold = O

        dV = 2 * a * O + b
        ddV = 2 * a
        V = a * O ** 2 + b * O + c
        # check type of V and dV

        Dw = 1 / (2 * tau) * V * V

        beta = 1 / (2 * tau) * V * dV
        # nablaW = np.dot(D, W) + np.dot(W, D)
        # nablaO = np.dot(D, O) + np.dot(O, D)
        laplacianO = (np.dot(L, O) + np.dot(O, L))/cell_size_2d
        laplacianW = (np.dot(L, W) + np.dot(W, L))/cell_size_2d
        # nablaDw = np.dot(D, Dw) + np.dot(Dw, D)
        # nablaBeta= np.dot(D, beta) + np.dot(beta, D)
        W_x = np.dot(W, D)/cell_size_1d
        W_y = np.dot(D, W)/cell_size_1d
        O_y = np.dot(D, O)/cell_size_1d
        O_x = np.dot(O, D)/cell_size_1d
        '''beta_y = np.dot(D, beta)
        beta_x = np.dot(beta, D)
        Dw_y = np.dot(D, Dw)
        Dw_x = np.dot(Dw, D)
        beta_x = 1/(2*tau)*(dV**2 + V*ddV*W_x)
        beta_y = 1/(2*tau)*(dV**2 + V*ddV*W_y)
        Dw_x = V/tau*dV*O_x
        Dw_y = V/tau*dV*O_y'''
        # dW_term = Dw * nablaW + beta * W * nablaO
        # dW = np.dot(D, Dw*nablaW) + np.dot(Dw*nablaW, D) + np.dot(D, beta*W*nablaO) + np.dot(beta*W*nablaO, D)
        # dW =1/tau * (V* dV * nablaO* nablaW) + (Dw*laplacianW) + 1/(2*tau)*(dV**2 + ddV*V)* nablaO+beta* (nablaW* nablaO+W*laplacianO)
        # dW = nablaDw*nablaW + Dw*laplacianW + nablaBeta*W*nablaO + beta*(nablaW*nablaO+W*laplacianO)
        # dW = Dw_x*W_x + Dw_y*W_y + Dw*laplacianW + beta_x*W*O_x + beta_y*W*O_y + beta*(W_x*O_x+W_y*O_y+W*laplacianO)
        dW = V / tau * dV * (O_x * W_x + O_y * W_y) + Dw * laplacianW + 1 / (2 * tau) * (
                (dV ** 2 + V * ddV) * (O_x * W * O_x + O_y * W * O_y)) + beta * (W_x * O_x + W_y * O_y + W * laplacianO)
        dO = D0 * laplacianO + f * (Om - O) - kc * W

        W += dW * dt
        O += dO * dt
        #check that O is between 0 and 0.21
        #O[O<0]=0
        #O[O>0.21]=0.21
        #check W is positive
        #W[W<0]=0
        #save W as a .csv inside of dest+"plots/W_timestep.csv every 500 timesteps"
        #if timestep % 1000 == 0:
        #    np.savetxt(dest+"plots/W_"+str(timestep)+".csv", W, delimiter=",")

        t = time.time()
        if t_start + t_end < t:
            break
        if timestep % 1000 == 0:
            print("timestep and dW max: " + str(timestep) + " " + str(np.abs(dW).max()))
        if timestep>1 and (np.abs(dW).max() < 10 ** (-6)):
            print("Converged")
            print(np.abs((W - Wold)).max())
            print(np.abs((O - Oold)).max())
            print("non zero dW values: "+str(dW[dW!=0]))
            break
        '''left_term = W * beta * kc
        right_term = f * Dw
        if  (left_term > right_term).all():
            print("Swarming")
            break
        else:
            print("W beta kc: ", left_term)
            print("f Dw: ", right_term)
            #print where exactly it is not satisfied
            for i in range(128):
                for j in range(128):
                    if left_term[i,j] <= right_term[i,j]:
                        print("W beta kc: ", left_term[i,j])
                        print("f Dw: ", right_term[i,j])
                        print("i,j: ", i,j)
            break'''
        #print("dW max: " + str(dW.max()))
        #print("dO max: " + str(dO.max()))
        #print("timestep: " + str(timestep))
        timestep += 1
    return W, O, timestep

if __name__ == "__main__":
    # let N be the sample size
    dest = "results/run69/"
    #check if folder exists, if not create it
    import os
    if not os.path.exists(dest):
        os.makedirs(dest)
    if not os.path.exists(dest+"plots/"):
        os.makedirs(dest+"plots/")
    N = 128
    l = 20
    h = l / N
    dt = 0.01
    # let l be the size of the grid in cm

    W0_vector = [(10 ** 2, i * 10 ** 2) for i in range(20, 121, 20)]
    Oam_vector = [0.042 * (i + 1) for i in range(0, 5)]

    #W0_vector = [(23 ** 6, 120 *10** 6)]
    #Oam_vector = [0.21]
    #W0_vector = [(1*10 ** 2, 90 *10** 2)]
    W0_vector = [(1, 90)]
    Oam_vector = [0.042 * 5]
    #W_low, W_high = 1 * 10 ** (6), 20 * 10 ** (6)
    # let D be the matrix that represents the divergence operator
    D = np.zeros((N, N))
    # we use the central difference approximation for the divergence operator, with periodic boundary conditions
    for i in range(N):
        D[i, (i + 1) % N] = 1 / (2 * h)
        D[i, (i - 1) % N] = -1 / (2 * h)

    # let L be the matrix that represents the laplacian operator
    L = np.zeros((N, N))
    # we use the central difference approximation for the laplacian operator, with periodic boundary conditions
    for i in range(N):
        L[i, i] = -2 / (2*h) ** 2
        L[i, (i + 1) % N] = 1 / (2*h) ** 2
        L[i, (i - 1) % N] = 1 / (2*h) ** 2

    # let the velocity parameters be defined as follows
    a = 1.89 * 10
    b = -3.98
    c = 2.25 * 10 ** (-1)
    # tumbling rate
    tau = 0.5
    # oxygen diffusion coefficient
    D0 = 2 * 10 ** (-1)
    # oxygen penetration rate
    f = 0.65
    # oxygen consumption rate by worms and bacteria
    kc = 7.3 * 10 ** (-4)
    #kc = 2 * 10**(-4)
    # ambient oxygen level from 0 to .21



    n_worms = 24000

    # initialize W as a matrix with uniform distribution and additional nois
    for (W_low, W_high) in W0_vector:
        for Oam in Oam_vector:
            print("solving for (W_high(0), O(0)): ", W_high, Oam)

            #W = l * l / (N * N) * (np.random.uniform(W_low, W_high, (N, N)) + np.random.normal(0, 1 * 10 ** (1), (N, N)))
            #W = l * l / (N * N) * (W_high*np.ones((N,N)) + np.random.normal(0, W_high, (N, N)))
            W=np.random.uniform(W_low, W_high, (N, N))+np.random.normal(0, W_low*0.01, (N, N))
            #W = W_high * np.ones((N, N)) + np.random.normal(0, W_high * 0.01, (N, N))
            #initialize W as a matrix with 48000 worms and additional noise
            #W = 1 / (N * N) * (n_worms * np.ones((N, N)) + np.random.uniform(-n_worms,n_worms,(N, N)))
            # set negative values to 0
            W[W < 0] = 0
            n_worms = np.sum(W)/N**2

            print("initial number of worms: ", n_worms)
            #O = O0 * np.ones((N, N)) + np.random.normal(0, 1 * 10 ** (-2), (N, N))
            #O = Oam *(np.ones((N,N))+ np.random.normal(0,0.05,(N, N)))
            O = Oam*np.ones((N,N))-kc/f*W
            #O = Oam *(np.ones((N,N)))

            #set negative values to 0
            O[O<0]=0
            # save initial values of W and O
            np.save(dest + "W0_W0_" + str(round(W_high, 1)) + "O0_" + str(Oam)[2:5], W)
            np.save(dest + "O0_W0_" + str(round(W_high, 1)) + "O0_" + str(Oam)[2:5], O)

            im = plt.imshow(W, cmap='hot', interpolation='nearest', animated=True, vmin=0)
            cbar = plt.colorbar(im)
            plt.savefig(dest + "W0_W0_" + str(round(W_high, 1)) + "O0_" + str(Oam)[2:5] + ".png")
            cbar.set_label('Worm density at time 0')
            plt.show()

            # plot O in a 2D grid
            plt.imshow(O)
            plt.savefig(dest + "O0_W0_" + str(round(W_high, 1)) + "O0_" + str(Oam)[2:5] + ".png")
            plt.show()
            V = a * O ** 2 + b * O + c
            dV = 2 * a * O + b
            # verify instability criterion for all points: W beta kc > f Dw
            beta = 1 / (2 * tau) * V * (2 * a * O + b)
            Dw = 1 / (2 * tau) * V ** 2

            # show where it is not satisfied
            print("W beta kc: ", W * beta * kc)
            print("f Dw: ", f * Dw)
            #check CFL condition
            print("CFL condition: ", np.max(V)*dt/h**2)
            if np.max(V)*dt/h**2>0.5:
                print("CFL condition not satisfied")
                #break
            # check that it doesn't exceed 10 mins
            #kc = 2*f*np.max(Dw)/(np.min(W*beta))
            print("kc = ", kc)
            print("instability criterion: ", (W * beta * kc > f * Dw).all())
            t_start = time.time()
            W, O, timestep = solve_PDE(W, O, a, b, c, tau, D0, f, kc, Oam, D, L, dt, t_start,dest,l,N, 600)

            # plot W in a 2D grid
            im = plt.imshow(W, cmap='hot', interpolation='nearest', animated=True, vmin=0)
            cbar = plt.colorbar(im)
            cbar.set_label('Worm density at time 10 mins')
            plt.savefig(dest + "Wtmax_W0_" + str(round(W_high, 1)) + "O0_" + str(Oam)[2:5] + ".png")

            plt.show()
            # plot O in a 2D grid
            plt.imshow(O)
            plt.savefig(dest + "Otmax_W0_" + str(round(W_high, 1)) + "O0_" + str(Oam)[2:5] + ".png")
            plt.show()

            # save the final values of W and O
            np.save(dest + "Wtmax_W0_" + str(round(W_high, 1)) + "O0_" + str(Oam)[2:5], W)
            np.save(dest + "Otmax_W0_" + str(round(W_high, 1)) + "O0_" + str(Oam)[2:5], O)
