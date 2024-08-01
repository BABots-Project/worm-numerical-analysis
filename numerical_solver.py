import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from numba import njit

N=512
l=0.02
dx=l/N
indices = np.array([[(j - 2)%N, (j - 1)%N, (j + 1)%N, (j + 2)%N] for j in range(N)])
dx12 = 12 * dx
dxdx = dx**2

def divergence(in_array):
    out = gradientX(in_array) + gradientY(in_array)
    return out


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

def show(matrix):
    #invert y axis to have the origin in the bottom left corner
    matrix = np.flipud(matrix)
    #set values that are less than 10e3 to 10e3
    #matrix[matrix < 10e3] = 10e3
    #plt.imshow(matrix, cmap='rainbow', interpolation='nearest', norm=LogNorm(vmin=1e-12, vmax=np.max(matrix)))
    plt.imshow(matrix, cmap='rainbow', interpolation='nearest')
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

    #plt.savefig(directory + "/final_density.pdf")
    plt.show()