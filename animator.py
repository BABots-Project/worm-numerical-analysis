import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#from swarming_simulator import show

rho_matrices = []

dir = "swarming_simulator/sim_3/"

#load all the matrices that start with rho_{step}.npy for step between 0 and 500000 with step 10000
for step in range(10000, 500001, 10000):
    #check if the file exists, if not, break
    try:
        with open(dir + f"rho_{step}.npy", 'r') as f:
            pass
    except FileNotFoundError:
        break
    rho = np.load(dir + f"rho_{step}.npy")
    rho_matrices.append(rho)

vmin = 0
vmax = np.max(rho_matrices)

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Initialize the plot with the first matrix
cax = ax.matshow(rho_matrices[0], vmin=vmin, vmax=vmax, cmap='viridis')
fig.colorbar(cax)

# Function to update the plot for each frame
def update(frame):
    cax.set_data(rho_matrices[frame])
    return cax,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(rho_matrices), blit=True)

# Save the animation to a file
ani.save('rho_evolution_oxy_40.mp4', writer='ffmpeg')

#plt.show()
#show(rho_matrices[-1])