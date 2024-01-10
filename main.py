# the following code allows to numerically solve the keller and sagel equations applied to the case of C. elegans
import time

# equations are taken from the paper: "Dynamics of pattern formation and emergence of swarming in Caenorhabditis elegans"

import numpy as np
import matplotlib.pyplot as plt
from solver import solve_PDE

# let N be the sample size
dest = "results/run4/"
N = 128
h = 1 / N
dt = 0.01
W0_vector = [(10 ** 6, i * 10 ** 6) for i in range(20, 121, 20)]
O0_vector = [0.042 * (i + 1) for i in range(0, 5)]
Oam_vector = [0.042 * (i + 1) for i in range(0, 5)]
#W0_vector = [(10 ** 6, 90 *10** 6)]
#O0_vector = [0.042 * 5]
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
D0 = 2 * 10 ** (-9)
# oxygen penetration rate
f = 0.65
# oxygen consumption rate by worms and bacteria
kc = 7.3 * 10 ** (-10)
# ambient oxygen level from 0 to .21

# let l be the size of the grid in cm
l = 2 * 10 ** (-2)

# let V be the vector of the velocity of worms, where V[i] is the velocity at the i-th point of the grid

# initialize W as a matrix with uniform distribution and additional nois
for (W_low, W_high) in W0_vector:
    for Oam in Oam_vector:
        print("solving for (W_high(0), O(0)): ", W_high, Oam)

        W = l * l / (N * N) * (np.random.uniform(W_low, W_high, (N, N)) + np.random.normal(0, 1 * 10 ** (6), (N, N)))
        #O = O0 * np.ones((N, N)) + np.random.normal(0, 1 * 10 ** (-2), (N, N))
        O = Oam * np.ones((N, N))
        # save initial values of W and O
        np.save(dest + "W0_W0_" + str(round(W_high, 1)) + "O0_" + str(Oam)[2:5], W)
        np.save(dest + "O0_W0_" + str(round(W_high, 1)) + "O0_" + str(Oam)[2:5], O)

        im = plt.imshow(W, cmap='hot', interpolation='nearest', animated=True)
        cbar = plt.colorbar(im)
        plt.savefig(dest + "W0_W0_" + str(round(W_high, 1)) + "O0_" + str(Oam)[2:5] + ".png")
        cbar.set_label('Worm density at time 0')
        plt.show()

        # plot O in a 2D grid
        plt.imshow(O)
        plt.savefig(dest + "O0_W0_" + str(round(W_high, 1)) + "O0_" + str(Oam)[2:5] + ".png")
        plt.show()

        # check that it doesn't exceed 10 mins
        t_start = time.time()
        W, O, timestep = solve_PDE(W, O, a, b, c, tau, D0, f, kc, Oam, D, L, dt, t_start)

        # plot W in a 2D grid
        im = plt.imshow(W, cmap='hot', interpolation='nearest', animated=True)
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
