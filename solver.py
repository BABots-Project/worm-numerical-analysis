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

def solve_PDE(W, O, a, b, c, tau, D0, f, kc, Om, D, L, dt, t_start, t_end=600):
    timestep= 0
    #animator = Animator(W, O, a, b, c, tau, D0, f, kc, Om, D, L, dt, t_start, t_end)
    #animator.animate()
    #return W, O, timestep
    while True:
        Wold = W
        Oold = O
        dV = 2 * a * O + b
        ddV = 2 * a
        V = a * O ** 2 + b * O + c
        Dw = V ** 2 / (2 * tau)
        beta = V / (2 * tau) * dV
        nablaW = np.dot(D, W) + np.dot(W, D)
        nablaO = np.dot(D, O) + np.dot(O, D)
        laplacianO = np.dot(L, O) + np.dot(O, L)
        dW_term = Dw * nablaW + beta * W * nablaO
        #dW = np.dot(D, Dw*nablaW) + np.dot(Dw*nablaW, D) + np.dot(D, beta*W*nablaO) + np.dot(beta*W*nablaO, D)
        dW = V/tau*dV*(np.dot(D, O)+np.dot(O,D)) + Dw*(np.dot(L,W)+np.dot(W,L)) + 1/(2*tau)*((dV**2*(np.dot(D,O)+np.dot(O,D)))+V*ddV*(np.dot(D,O)+np.dot(O,D)))*W*(np.dot(D,O)+np.dot(O,D)) + beta*(np.dot(D,W)+np.dot(W,D))*(np.dot(D,O)+np.dot(O,D)) + beta*W*(np.dot(L,O)+np.dot(O,L))
        dO = D0 * laplacianO + f * (Om - O) - kc * W

        W += dW * dt
        O += dO + dt


        t = time.time()
        if t_start + t_end < t:
            break
        if timestep>1 and (np.abs(dW).max() < 10 ** (-14) or np.abs(dO).max()< 10 ** (-14)):
            print("Converged")
            print(np.abs((W - Wold)).max())
            print(np.abs((O - Oold)).max())
            print("non zero dW values: "+str(dW[dW!=0]))
            #break
        if (W * beta * kc).all() > (f * Dw).all():
            print("Swarming")
            break
        #print("dW max: " + str(dW.max()))
        #print("dO max: " + str(dO.max()))
        #print("timestep: " + str(timestep))
        timestep += 1
    return W, O, timestep