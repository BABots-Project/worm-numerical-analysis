import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Animator:
    def __init__(self, W, O, a, b, c, tau, D0, f, kc, Oam, D, L, dt, t_start, t_end=600):
        self.W = W
        self.O = O
        self.a = a
        self.b = b
        self.c = c
        self.tau = tau
        self.D0 = D0
        self.f = f
        self.kc = kc
        self.Oam = Oam
        self.D = D
        self.L = L
        self.dt = dt
        self.t_start = t_start
        self.t_end = t_end

        self.timestep = 0

        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.W, cmap='hot', interpolation='nearest', animated=True)
        self.cbar = self.fig.colorbar(self.im)
        self.cbar.set_label('Worm density at time 0')
        self.ax.set_title('timestep: ' + str(self.timestep))
        self.anim = FuncAnimation(self.fig, self.update, interval=1, blit=True)

    def update(self, i):
        self.timestep += 1
        dV = 2 * self.a * self.O + self.b
        V = self.a * self.O ** 2 + self.b * self.O + self.c
        Dw = V ** 2 / (2 * self.tau)
        beta = V / (2 * self.tau) * dV
        nablaW = np.dot(self.D, self.W) + np.dot(self.W, self.D)
        nablaO = np.dot(self.D, self.O) + np.dot(self.O, self.D)
        laplacianO = np.dot(self.L, self.O) + np.dot(self.O, self.L)
        dW_term = Dw * nablaW + beta * self.W * nablaO
        dW = np.dot(self.D, dW_term) + np.dot(dW_term, self.D)
        dO = self.D0 * laplacianO + self.f * (self.Oam - self.O) - self.kc * self.W

        self.W += dW * self.dt
        self.O += dO
        self.im.set_array(self.W)
        self.ax.set_title('timestep: ' + str(self.timestep))

        return [self.im]
    def animate(self):
        animation = FuncAnimation(self.fig, self.update, frames=2000, interval=1, cache_frame_data=False, blit=True)
        animation.save('animation.gif', writer='imagemagick', fps=60)
        plt.show()




def solve_PDE_attractant(U,rho,D,L,dt,t_max=600):
    gamma = 0.01
    s = 0.01
    beta = 1.111 * 10**(-5)
    sigma = 5.555 * 10 ** (-6)
    scale = 2
    rho_max = 28000
    cushion = 2000
    alfa = 1500
    d=1*10**(-6)
    t_start = time.time()
    timestep = 0
    while True:
        nablarho = np.dot(D,rho)+np.dot(rho,D)
        Vrho = sigma*scale*(1+np.tanh((rho-rho_max)/cushion))

        #check if alfa+U is nan
        if np.isnan(alfa+U).any():
            break
        Vu = -beta*np.log(alfa+U)
        V = Vu + Vrho
        nablaV = np.dot(D,V)+np.dot(V,D)
        laplacianU = np.dot(L,U)+np.dot(U,L)
        laplacianV = np.dot(L,V)+np.dot(V,L)
        laplacianrho = np.dot(L,rho)+np.dot(rho,L)
        dU = -gamma*U+d*laplacianU+s*rho
        drho = nablarho*nablaV+rho*laplacianV+sigma*laplacianrho

        U += dU*dt
        rho += drho*dt

        t = time.time()
        if t_start + t_max < t:
            break
        timestep += 1
    print("timestep: "+str(timestep))
    return U,rho


dest = "results/run43_diffusion/"
#check if folder exists, if not create it
import os
if not os.path.exists(dest):
    os.makedirs(dest)
N = 128
h = 1 / N
dt = 0.01
l=10**(-2)
# let D be the matrix that represents the divergence operator
D = np.zeros((N, N))
# we use the central difference approximation for the divergence operator, with periodic boundary conditions
for i in range(N):
    D[i, (i + 1) % N] = 1 / (2*h)
    D[i, (i - 1) % N] = -1 / (2*h)

# let L be the matrix that represents the laplacian operator
L = np.zeros((N, N))
# we use the central difference approximation for the laplacian operator, with periodic boundary conditions
for i in range(N):
    L[i, i] = -2 / (2*h) ** 2
    L[i, (i + 1) % N] = 1 / (2*h) ** 2
    L[i, (i - 1) % N] = 1 / (2*h) ** 2

rho = (np.random.uniform(0, 9000, (N,N)) + np.random.normal(0, 90, (N, N)))/(N*N)
U = np.random.normal(0,1,(N,N))
plt.imshow(U, cmap='hot', interpolation='nearest', animated=True, vmin=0)
cbar = plt.colorbar()
cbar.set_label('U(0)')
plt.show()
plt.imshow(rho, cmap='hot', interpolation='nearest', animated=True, vmin=0)
cbar = plt.colorbar()
cbar.set_label('rho(0)')
plt.show()
#save matrices
np.save(dest + "U0", U)
np.save(dest + "rho0", rho)
rho, U = solve_PDE_attractant(U,rho,D,L,dt)
plt.imshow(U, cmap='hot', interpolation='nearest', animated=True, vmin=0)
cbar = plt.colorbar()
cbar.set_label('U(tmax)')
plt.show()
plt.imshow(rho, cmap='hot', interpolation='nearest', animated=True, vmin=0)
cbar = plt.colorbar()
cbar.set_label('rho(tmax)')
plt.show()
#save matrices
np.save(dest + "Utmax", U)
np.save(dest + "rhotmax", rho)