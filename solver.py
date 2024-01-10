import time

import numpy as np


def solve_PDE(W, O, a, b, c, tau, D0, f, kc, Om, D, L, dt, t_start, t_end=600):
    timestep= 0
    while True:
        Wold = W
        Oold = O
        dV = 2 * a * O + b
        V = a * O ** 2 + b * O + c
        Dw = V ** 2 / (2 * tau)
        beta = V / (2 * tau) * dV
        nablaW = np.dot(D, W) + np.dot(W, D)
        nablaO = np.dot(D, O) + np.dot(O, D)
        laplacianO = np.dot(L, O) + np.dot(O, L)

        dW = np.dot(D, Dw * nablaW + beta * W * nablaO) + np.dot(Dw * nablaW + beta * W * nablaO, D)
        dO = D0 * laplacianO + f * (Om - O) - kc * W

        W += dW * dt
        O += dO + dt

        timestep += 1
        t = time.time()
        if t_start + t_end < t:
            break
        if dW.max() < 10 ** (-6) and dO.max() < 10 ** (-6):
            print("Converged")
            print((W - Wold).max())
            print((O - Oold).max())
            break
        if (W * beta * kc).all() > (f * Dw).all():
            print("Swarming")
            break

    return W, O, timestep
