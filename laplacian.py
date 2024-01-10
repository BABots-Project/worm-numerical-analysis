# Description: This file contains the function to generate the Laplacian matrix
import numpy as np

def laplacian_matrix_2d(Nx, Ny, hx, hy):
    D2 = np.zeros((Nx * Ny, Nx * Ny))

    for i in range(Nx):
        for j in range(Ny):
            idx = i * Ny + j

            D2[idx, idx] = -2 / hx**2 - 2 / hy**2
            D2[idx, (i + 1) % Nx * Ny + j] = 1 / hx**2  # Right neighbor
            D2[idx, (i - 1) % Nx * Ny + j] = 1 / hx**2  # Left neighbor
            D2[idx, i * Ny + (j + 1) % Ny] = 1 / hy**2  # Upper neighbor
            D2[idx, i * Ny + (j - 1) % Ny] = 1 / hy**2  # Lower neighbor

    return D2