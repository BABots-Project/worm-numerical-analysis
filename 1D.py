import time

# equations are taken from the paper: "Dynamics of pattern formation and emergence of swarming in Caenorhabditis elegans"

import numpy as np
import matplotlib.pyplot as plt

def solve_PDE(W, O, a, b, c, tau, D0, f, kc, Om, D, L, dt, t_start, dest, t_end=600):
    timestep = 0
    while True:
        Wold = W
        Oold = O

        dV = 2 * a * O + b
        ddV = 2 * a
        V = a * O ** 2 + b * O + c
        #check type of V and dV

        Dw = 1/(2 * tau)*V*V
        beta = 1/(2*tau)*V*dV
        nablaW = np.dot(D, W)
        nablaO = np.dot(D, O)
        laplacianO = np.dot(L, O)
        laplacianW = np.dot(L, W)
        nablaDw = np.dot(D, Dw)
        nablaBeta= np.dot(D, beta)
        #dW_term = Dw * nablaW + beta * W * nablaO
        #dW = np.dot(D, Dw*nablaW) + np.dot(Dw*nablaW, D) + np.dot(D, beta*W*nablaO) + np.dot(beta*W*nablaO, D)
        #dW =1/tau * (V* dV * nablaO* nablaW) + (Dw*laplacianW) + 1/(2*tau)*(dV**2 + ddV*V)* nablaO+beta* (nablaW* nablaO+W*laplacianO)
        dW = nablaDw*nablaW + Dw*laplacianW + nablaBeta*W*nablaO + beta*(nablaW*nablaO+W*laplacianO)
        dO = D0 * laplacianO + f * (Om - O) - kc * W

        W += dW * dt
        O += dO * dt

        #save W as a .csv inside of dest+"plots/W_timestep.csv every 500 timesteps"
        if timestep % 1000 == 0:
            np.savetxt(dest+"plots/W_"+str(timestep)+".csv", W, delimiter=",")

        t = time.time()
        if t_start + t_end < t:
            break
        if timestep % 100 == 0:
            print("timestep and dW max: " + str(timestep) + " " + str(np.abs(dW).max()))
        if timestep>1 and (np.abs(dW).max() < 10 ** (-6)):
            print("Converged")
            print(np.abs((W - Wold)).max())
            print(np.abs((O - Oold)).max())
            print("non zero dW values: "+str(dW[dW!=0]))
            break

        #print("dW max: " + str(dW.max()))
        #print("dO max: " + str(dO.max()))
        #print("timestep: " + str(timestep))
        timestep += 1
    return W, O, timestep

# let N be the sample size
dest = "results/run44_1d/"
#check if folder exists, if not create it
import os
if not os.path.exists(dest):
    os.makedirs(dest)
if not os.path.exists(dest+"plots/"):
    os.makedirs(dest+"plots/")
N = 128
h = 1 / N
dt = 0.05

W0_vector = [(10 ** 4, i * 10 ** 4) for i in range(20, 121, 20)]
Oam_vector = [0.042 * (i + 1) for i in range(0, 5)]

#W0_vector = [(23 ** 6, 120 *10** 6)]
#Oam_vector = [0.21]
W0_vector = [(10*10 ** 4, 120 *10** 4)]
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
a = 1.90 * 10 ** (-2)
b = -3.98 * 10 ** (-3)
c = 2.25 * 10 ** (-4)
# tumbling rate
tau = 0.5
# oxygen diffusion coefficient
D0 = 2 * 10 ** (-5)
# oxygen penetration rate
f = 0.65
# oxygen consumption rate by worms and bacteria
kc = 7.3 * 10 ** (-10)
# ambient oxygen level from 0 to .21

# let l be the size of the grid in cm
l = 2

n_worms = 48000

# initialize W as a matrix with uniform distribution and additional nois
for (W_low, W_high) in W0_vector:
    for Oam in Oam_vector:
        print("solving for (W_high(0), O(0)): ", W_high, Oam)

        W = l * l / (N * N) * (np.random.uniform(W_low, W_high, (N, N)) + np.random.normal(0, 1 * 10 ** (4), (N, N)))

        #initialize W as a matrix with 48000 worms and additional noise
        W = 1 / (N) * (n_worms * np.ones((N, 1)) + np.random.uniform(-n_worms,n_worms,(N, 1)))
        n_worms = np.sum(W)
        print("initial number of worms: ", n_worms)
        #O = O0 * np.ones((N, N)) + np.random.normal(0, 1 * 10 ** (-2), (N, N))
        O = Oam * np.random.normal(0,0.05,(N, 1))
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
        #check CFL condition
        print("CFL condition: ", np.max(V)*dt/h**2)
        if np.max(V)*dt/h**2>0.5:
            print("CFL condition not satisfied")
            break
        # check that it doesn't exceed 10 mins
        t_start = time.time()
        W, O, timestep = solve_PDE(W, O, a, b, c, tau, D0, f, kc, Oam, D, L, dt, t_start,dest, 1200)

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