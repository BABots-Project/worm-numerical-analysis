import os

from PIL import Image

# List to store heatmap images
heatmap_images = []
current_directory = os.getcwd()
folder = current_directory+"/run4/"
O_vector = [0.042 * (i + 1) for i in range(0, 5)]
W_vector = [i * 10 ** 6 for i in range(120, 19, -20)]


for W in W_vector:
    for O in O_vector:
        heatmap_path = folder + f"Wtmax_W0_{W}O0_"+str(O)[2:5]+".png"
        heatmap_images.append(Image.open(heatmap_path))

# Create a new blank image for the grid
grid_size = (5, 5)
grid_width = heatmap_images[0].width * grid_size[0]
grid_height = heatmap_images[0].height * grid_size[1]
grid_image = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

# Paste each heatmap onto the grid
for i in range(len(heatmap_images)):
    row = i // grid_size[0]
    col = i % grid_size[0]
    x_offset = col * heatmap_images[i].width
    y_offset = row * heatmap_images[i].height
    grid_image.paste(heatmap_images[i], (x_offset, y_offset))

# Save the final grid image
grid_image.save(folder+"grid_image.png")