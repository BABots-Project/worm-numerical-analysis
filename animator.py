import random

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import os
dest = "results/run46_AA/"
# Generate file paths, taking all files in dest+"plots/" that end with .csv
file_paths = [dest + "plots/" + file for file in os.listdir(dest + "plots/") if file.endswith(".csv")]

# Create a figure and axis for the heatmap
fig, ax = plt.subplots()

# Function to update the heatmap for each frame
def update(frame):
    ax.clear()

    # Read the CSV file for the current frame
    file_path = file_paths[frame]
    data = pd.read_csv(file_path)

    # Assuming your CSV file has a matrix-like structure
    heatmap = ax.imshow(data.values, cmap='viridis', vmin=0)

    # Customize the plot as needed (e.g., add colorbar, labels, etc.)
    timestep = file_paths[frame].split("/")[-1].split('.')[0].split('_')[-1]
    plt.title(f"timestep: {timestep}")

# Create an animation
animation = FuncAnimation(fig, update, frames=len(file_paths), interval=1000)

# Save the animation as a video file (e.g., MP4)
animation_file = "heatmap_animation" +str(random.randint(0,10000)) +".mp4"
animation.save(animation_file, writer='ffmpeg', fps=1)

# Display the animation (optional)
plt.show()