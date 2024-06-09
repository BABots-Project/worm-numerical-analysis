import csv
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
from math import sqrt, floor
import sys,os
from tqdm import tqdm
from scipy.stats import kurtosis
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

def save_graph(graph, filename):
    with open(filename, 'wb') as f:
        pickle.dump(graph, f)

def load_graph(filename):
    with open(filename, 'rb') as f:
        graph = pickle.load(f)
    return graph
'''
def create_matrix_from_tsv(tsv_file):
    # Read data from tsv file
    data = np.loadtxt(tsv_file, delimiter='\t')
    n = int(sqrt(len(data)))
    # Extract x, y, and f(x, y) values
    x_values = data[:, 0]
    y_values = data[:, 1]
    f_values = data[:, 2]

    # Normalize x, y coordinates to range [0, 511]
    x_values_normalized = ((x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))) * (n-1)
    y_values_normalized = ((y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))) * (n-1)

    # Round normalized coordinates to integers
    x_indices = np.round(x_values_normalized).astype(int)
    y_indices = np.round(y_values_normalized).astype(int)

    # Initialize matrix
    matrix = np.zeros((n,n))

    # Fill matrix with data points
    matrix[x_indices, y_indices] = f_values

    return matrix
    '''

def create_matrix_from_tsv(tsv_file):
    # Read data from tsv file
    data = np.loadtxt(tsv_file, delimiter='\t')

    if data.ndim == 1 or data.shape[1] == 1:
        # Case with just f values
        f_values = data.flatten()
        
        # Calculate the number of rows and columns in the matrix
        n = int(sqrt(len(f_values)))
        
        # Check if f_values can form a perfect square matrix
        if n * n != len(f_values):
            raise ValueError("The number of f values does not form a perfect square.")
        
        # Create grid for x and y values
        l = n
        delta = l / n
        x_values = np.arange(0, l, delta)
        y_values = np.arange(0, l, delta)
        
        # Create meshgrid and flatten
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        x_values = x_grid.flatten()
        y_values = y_grid.flatten()
    else:
        # Case with x, y, f values
        x_values = data[:, 0]
        y_values = data[:, 1]
        f_values = data[:, 2]
        
        # Calculate the number of rows and columns in the matrix
        n = int(sqrt(len(f_values)))
    
    # Normalize x, y coordinates to range [0, n-1]
    x_values_normalized = ((x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))) * (n-1)
    y_values_normalized = ((y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))) * (n-1)

    # Round normalized coordinates to integers
    x_indices = np.round(x_values_normalized).astype(int)
    y_indices = np.round(y_values_normalized).astype(int)

    # Initialize matrix
    matrix = np.zeros((n, n))

    # Fill matrix with data points
    for i in range(len(f_values)):
        matrix[x_indices[i], y_indices[i]] = f_values[i]

    return matrix


def create_matrix_default(n, default_value):

    # Initialize matrix
    matrix = np.ones((n,n))*default_value

    return matrix

def get_neighbours(matrix, point, i, j, avg):
    neighbours = []
    nrows, ncols = matrix.shape

    # Define the indices for neighboring cells including diagonals
    indices = [
        (i-1, j), (i+1, j),  # Upper and lower neighbors
        (i, j-1), (i, j+1),  # Left and right neighbors
        (i-1, j-1), (i-1, j+1),  # Upper left and upper right neighbors
        (i+1, j-1), (i+1, j+1),  # Lower left and lower right neighbors
    ]

    # Handle periodic boundary conditions
    for ii, jj in indices:
        if ii < 0:
            ii = nrows - 1
        elif ii >= nrows:
            ii = 0
        if jj < 0:
            jj = ncols - 1
        elif jj >= ncols:
            jj = 0

        # Check if the neighbor meets the criteria
        if matrix[ii, jj] < avg and point < avg:
            neighbours.append((ii, jj))
        elif matrix[ii, jj] >= avg and point >= avg:
            neighbours.append((ii, jj))

    return neighbours


def intersect(points, neighbours):
    # Convert the lists of tuples into sets of tuples
    points_set = set(points)
    neighbours_set = set(neighbours)
    
    # Find the intersection of the sets
    intersection = points_set.intersection(neighbours_set)
    
    # Convert the intersection set back to a list of tuples
    intersection_list = list(intersection)
    
    return intersection_list

def get_unique_values(dictionary):
    # Collect all the values into a set
    unique_values = set(dictionary.values())
    
    # Return the count of unique values
    return list(unique_values)


def dfs(dic, start, start_i, start_j, visited, matrix, avg):
    stack = [(start_i, start_j)]
    id_ = dic[(start_i,start_j)]
    while stack:
        current = stack.pop()

        if visited[current[0],current[1]]==0:
            visited[current[0],current[1]]=1
            
            
            dic[current]=id_
            neighbours = get_neighbours(matrix, matrix[current[0], current[1]], current[0], current[1], avg)
            for neighbor in neighbours:
                stack.append(neighbor)

def plot_heatmap(densityMatrix, ranges_=""):
    fig, ax = plt.subplots()
    plt.gca().set_aspect('equal')
    img = ax.imshow(densityMatrix, cmap="magma", aspect='auto')
    if ranges_:
        ax.set_xticklabels(ranges_[0])
        ax.set_yticklabels(ranges_[1])

    ratio = 1.0

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
    plt.show()

def dict_to_matrix(dictionary, n):
    matrix = np.zeros((n, n))  # Initialize matrix with zeros
    
    for (x, y), value in dictionary.items():
        if x < n and y < n:  # Check if the indices are within the bounds of the matrix
            matrix[x, y] = value
    
    return matrix

def plot_distribution(matrix, b, show_plot=False, n1=1):
    n = len(matrix)
    minValue = np.min(matrix)
    maxValue = np.max(matrix)
    ys = np.zeros(b+1)

    delta = (maxValue-minValue)/b
    for i in range(n):
        for j in range(n):
            value = matrix[i, j]

            bin_index = floor((value - minValue) / delta)
            ys[bin_index] += 1

    bins = np.linspace(minValue, maxValue, b+1)+ delta / 2
    kurt = kurtosis(matrix.flatten())
    #print("kurtosis: ", kurt)
    if show_plot:
        fig, ax = plt.subplots()

        plt.bar(bins/1e6, ys / np.sum(ys), width=delta/1e6, align='edge', edgecolor='black')
        #plt.title('Distribution of Values in Matrix')
        #plt.xlabel('Value Range')
        plt.ylabel('Frequency', fontsize=30)
        ratio = 1.0

        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right-x_left)/(_ylow-y_high))*ratio)
        ax.tick_params(labelsize=21)
        plt.xlabel('$W_T$ [worm/mm$^2$]', fontsize=30)
        plt.tight_layout()
        plt.savefig(f"plots/fig2/math_{n1}wmm_histo.pdf", format="pdf", dpi=90)

    return kurt, bins, ys

    
def build_dic(matrix,data,b):
    avg = np.average(matrix)
    ids=[]
    p=[]
    p_id = {}
    last_id = 0

    visited = create_matrix_default(len(matrix), 0)
    for i,line in enumerate(matrix):
        for j,point in enumerate(line):

            if visited[i,j]==0:

                p_id[(i,j)]=last_id
                last_id+=1


                dfs(p_id, point, i, j, visited, matrix, avg)



    #save_graph(p_id, "dictionary_"+data.replace(".", "").split("/")[-1]+".pkl")
    '''pbar.close()
    plot_heatmap(matrix)
    plot_heatmap(visited)
    p_id_matrix = dict_to_matrix(p_id, len(matrix))
    plot_heatmap(p_id_matrix)
    plt.show()'''
    kurt, bins, ys = plot_distribution(matrix, b)
    return p_id, kurt, bins, ys

def evaluate(data, b):
    if data.endswith(".tsv"):
        matrix = create_matrix_from_tsv(data)
    else:
        matrix = np.load(data)
    avg = np.average(matrix)

    #matrix = matrix[236:276, 236:276]
    filename = "dictionary_"+data.replace(".", "").split("/")[-1]+".pkl"
    if True:#not os.path.isfile(filename) and False:

        #print("building...")
        p_id, kurt, bins, ys = build_dic(matrix, data, b)
    else:
        p_id, kurt = load_graph(filename)
    #plot_heatmap(dict_to_matrix(p_id, len(matrix)))
    m=dict_to_matrix(p_id, len(matrix))
    #restrict m to 236:276, 236:276
    #m = m[236:276, 236:276]
    unique_ids = get_unique_values(p_id)
    id_index = 0
    ids_under_avg = []
    ids_over_avg = []
    sum_of_densities_above_avg = 0
    sum_of_densities_below_avg = 0

    for k in p_id:
        current_id = unique_ids[id_index]
        if p_id[k] == current_id:
            i,j = k
            if matrix[i,j]<avg:
                ids_under_avg.append(current_id)
                #print("adding id: ", current_id)
            else:
                ids_over_avg.append(current_id)
            id_index+=1
            if id_index>=len(unique_ids):
                break
    mask = np.isin(m, ids_over_avg)
    filtered_matrix = np.where(mask, m, np.nan)
    #plot_heatmap(filtered_matrix)
    #print("clusters above: ", ids_over_avg)
    #print("clusters under: ", ids_under_avg)
    for k in p_id:
        i,j = k
        if matrix[i,j]>=avg:
            sum_of_densities_above_avg+=matrix[i,j]
    c=sum_of_densities_above_avg/np.sum(matrix)
    #print("clustering metric: ", c)
    #build_dendrogram(dict_to_matrix(p_id, len(matrix)), ids_over_avg)
    #plt.show()
    return c, kurt, bins, ys

def plot_distance_boxplot(distance_matrix):
    #upper_tri_indices = np.triu_indices(distance_matrix.shape[0], k=1)
    #upper_tri_distances = distance_matrix[upper_tri_indices].flatten()

    plt.boxplot(distance_matrix)
    plt.xlabel('Upper Triangular Distances')
    plt.ylabel('Distance')
    plt.title('Boxplot of Upper Triangular Distances')
    plt.show()

def get_toroidal_distance(a,b):
    a = a*20/511
    b = b*20/511
    dx = abs(a[0]-b[0])
    dy = abs(a[1]-b[1])

    if dx>10:
        dx = 20-dx
    if dy>10:
        dy = 20-dy

    return sqrt(dx**2+dy**2)

def compute_distance(cluster1, cluster2):
    # Compute the distance between two clusters based on their minimum distance
    # Find the minimum distance between all pairs of points in cluster1 and cluster2
    min_distance = np.inf
    for i in cluster1:
        for j in cluster2:
            distance = get_toroidal_distance(i,j)
            if distance < min_distance:
                min_distance = distance
    return min_distance

def build_dendrogram(matrix, id_whitelist):

    clusters = [np.where(matrix==x) for x in id_whitelist]

    # Compute the distance matrix
    n = len(clusters)

    print("n of clusters: ", n)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            distance_matrix[i, j] = compute_distance(clusters[i], clusters[j])
            distance_matrix[j, i] = distance_matrix[i, j]

    #plot_distance_boxplot(distance_matrix)
    #return distance_matrix

    # Build dendrogram
    condensed_dist_matrix = squareform(distance_matrix)
    linkage_matrix = hierarchy.linkage(distance_matrix)

    # Plot dendrogram
    plt.figure(figsize=(10, 5))
    hierarchy.dendrogram(linkage_matrix)#, truncate_mode="level", p=7)
    plt.title('Dendrogram')
    plt.xlabel('Cluster Index')
    plt.ylabel('Distance')
    plt.show()

def bulk_kurtosis_and_densities(ns, runs, b, colors, show=False):
    kurtosis_matrix = np.zeros((runs, (len(ns))))
    clustering_matrix = np.zeros((runs, (len(ns))))
    avg_densities = np.zeros((len(ns), b+1))
    avg_densities_matrices = []
    bin_list = np.zeros((len(ns), b+1))
    for i,n in enumerate(ns):
        print("W = ", n)
        #data = "results_oxygen_all/W_"+str(n)+"000000.0_n_512_tmax_50000/Wt_50000.txt"
        data = "results_pheromone/rho0_"+str(n)+"000000.0_n_512_tmax_500000/rhot_500000.txt"
        #data = sys.argv[1]
        data_center = data.split("/")[-2]
        data_list = [data.split("/")[-3]+"/"+data_center+str(i)+"/"+data.split("/")[-1] for i in range(1,runs+1)]

        density_list = []
        for j,data_ in enumerate(data_list):
            c, kurt, bins, ys = evaluate(data_, b)
            kurtosis_matrix[j,i]=kurt
            clustering_matrix[j,i] = c
            matrix = create_matrix_from_tsv(data_)
            density_list.append(matrix)
            #save_graph(bins, "distribution_rho0_"+str(n)+"_i_"+str(j)+".pkl")
        avg_density = np.average(density_list)
        if show==True:
            kurt, bins, ys = plot_distribution(matrix, b)
            avg_densities[i] = ys

    #save_graph(kurtosis_matrix, "oxygen_kurtosis_matrix.pkl")
    #save_graph(clustering_matrix, "oxygen_clustering_matrix.pkl")
    save_graph(kurtosis_matrix, "pheromone_kurtosis_matrix.pkl")
    save_graph(clustering_matrix, "pheromone_clustering_matrix.pkl")
    if show==True:
        fig, axes = plt.subplots(1, 5, figsize=(15, 4))  # 1 row, 5 columns

        # Plotting data on each subplot

        for i in range(len(ns)):
            data1 = avg_densities[i]
            axes[i].bar(range(len(data1)), data1, color=colors[i])
            axes[i].set_title('W='+str(ns[i]))

        plt.tight_layout()
        plt.show()

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)



def plot_heatmap_accurate(densityMatrix, x, y):
    fig, ax = plt.subplots()
    plt.gca().set_aspect('equal')
    img = ax.imshow(densityMatrix, cmap="magma", aspect='auto')
    ax.set_xticks(range(0, len(x)), [str(round(x_,2)) for x_ in x])
    ax.set_xlabel("h")
    ax.set_yticks(range(0, len(y)), [str(round(y_,2)) for y_ in y])
    ax.set_ylabel("$W_0$")
    ratio = 1.0

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
    plt.tight_layout()
    plt.show()


def plot_grid_of_heatmaps(ns, hs):
    # Number of rows and columns in the grid
    num_rows = len(ns)
    num_cols = len(hs)

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))

    # Cycle through the parameters and plot each heatmap
    for i, n in enumerate(ns):
        for j, h in enumerate(hs):
            ax = axes[i, j]
            data = f"results_hetero/rho0_{n}000000.0_h_{round(h,2)}/rhot_500000.txt"
            matrix = create_matrix_from_tsv(data)
            img = ax.imshow(matrix, cmap="magma", aspect='auto')

            ratio = 1.0

            x_left, x_right = ax.get_xlim()
            y_low, y_high = ax.get_ylim()
            ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

    
    # Adjust layout
    plt.tight_layout()
    plt.show()



'''
ns = list(range(40,90,5))
runs = 10
b = 50 #bins for the histograms 
colors = ["blue", "red", "green", "orange", "purple", "black", "yellow", "brown", "gray", "olive"] #colors for histograms
#colors = colors[:5]
#data = sys.argv[1]
distance_matrix = []
tmax=50000
for i, n in enumerate(ns):
# Construct the filename
    data = f"results_oxygen_all/W_{n}000000.0_n_512_tmax_500001/Wt_{tmax}.txt"

    matrix = create_matrix_from_tsv(data)
    avg = np.average(matrix)
    p_id, kurt, bins, ys = build_dic(matrix, data, b)
    unique_ids = get_unique_values(p_id)
    id_index = 0
    ids_under_avg = []
    ids_over_avg = []
    sum_of_densities_above_avg = 0
    sum_of_densities_below_avg = 0

    for k in p_id:
        current_id = unique_ids[id_index]
        if p_id[k] == current_id:
            i,j = k
            if matrix[i,j]<avg:
                ids_under_avg.append(current_id)
                #print("adding id: ", current_id)
            else:
                ids_over_avg.append(current_id)
            id_index+=1
            if id_index>=len(unique_ids):
                break
    print("before: ", len(ids_over_avg))
    #plot_heatmap(matrix)
    dist_m = build_dendrogram(dict_to_matrix(p_id, len(matrix)), list(dict.fromkeys(ids_over_avg)))
    distance_matrix.append(dist_m.flatten())

plot_distance_boxplot(distance_matrix)

'''

'''
ns = range(40,81,10)
b=50
runs=5
colors = ["blue", "red", "green", "orange", "purple", "black", "yellow", "brown", "gray", "olive"]
bulk_kurtosis_and_densities(ns, runs, b, colors, False)
'''


'''ns = [40, 65, 80]

# Define the fixed value for t
t = 50000
#evaluate("test.txt", b)
fig, axs = plt.subplots(len(ns), 1, figsize=(4, 18), sharex=True)

# Iterate over each value of n
for i, n in enumerate(ns):
    # Construct the filename
    data = f"results_oxygen_all/W_{n}000000.0_n_512_tmax_500001/Wt_{t}.txt"
    
    # Call evaluate function to get ys and bins
    c, kurt, bins, ys = evaluate(data, b)
    ys_normalized = ys / np.sum(ys)
    # Plot the histogram
    bins = bins/1e6
    delta = bins[1] - bins[0]  # Calculate the bin width
    axs[i].bar(bins, ys_normalized, width=delta, align='edge', edgecolor='black')
    #axs[i].set_title(f'Distribution of Values for $W_0$ = {n}/$mm^2$')
    axs[i].set_ylabel('Frequency')
    ratio = 1.0
    x_left, x_right = axs[i].get_xlim()
    y_low, y_high = axs[i].get_ylim()
    axs[i].set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
    axs[i].margins(x=0,y=0)
# Set common labels and show plot
plt.xlabel('')
plt.tight_layout()
plt.show()
'''


'''
tmax=500000
b=50
n=80
tlist = range(1000, tmax+1, 1000)
c_list = []
for t in range(1000, tmax+1, 1000):
    # Construct the filename
    data = f"results_pheromone/rho0_{n}000000.0_n_512_tmax_5000001/rhot_{t}.txt"
    
    # Call evaluate function to get ys and bins
    c, kurt, bins, ys = evaluate(data, b)
    c_list.append(c)
save_graph(c_list, "clustering_list.pkl")
plt.plot(tlist, c_list, color='blue', marker='o', linestyle='-')
plt.xlabel('File Number')
plt.ylabel('Percentage Above Average Density')
plt.title('Percentage Above Average Density for Each File')
plt.grid(True)
plt.show()
'''

'''
import sys
b=50
n=int(sys.argv[1])
t=50000
data = f"results_oxygen_all/W_{n}000000.0_n_512_tmax_500001/Wt_{t}.txt"
matrix = create_matrix_from_tsv(data)
plot_distribution(matrix, b, show_plot=True, n1=n)
'''

'''
b=50
t=500000
ns = range(40,85,5)
hs = np.arange(0, 1.0, 0.1)

clustering_matrix = np.zeros((len(ns), len(hs)))
kurtosis_matrix = np.zeros((len(ns), len(hs)))

for i,n in enumerate(ns):
    for j,h in enumerate(hs):
        print("n, h", n, h)
        
        if h<0.001:
            data = f"results_oxygen_all/W_{n}000000.0_n_512_tmax_500001/Wt_50000.txt"

        elif h>0.91:
            data = f"results_pheromone/rho0_{n}000000.0_n_512_tmax_5000001/rhot_{t}.txt"

        else:
            data = f"results_hetero/rho0_{n}000000.0_h_{round(h,2)}/rhot_{t}.txt"
                
        #data = f"results_hetero_/results_hetero/rho0_{n}000000.0_h_{round(h,2)}/rhot_{t}.txt"
        c, kurt, bins, ys = evaluate(data,b)
        clustering_matrix[i,j]=c
        kurtosis_matrix[i,j]=kurt


save_graph(clustering_matrix, "hetero_clustering_matrix.pkl")
save_graph(kurtosis_matrix, "hetero_kurtosis_matrix.pkl")
ranges_=[hs, ns]
plot_heatmap(clustering_matrix, ranges_)
plot_heatmap(kurtosis_matrix, ranges_)
'''

'''ns = range(40,85,5)
hs = np.arange(0, 1.0, 0.1)
ranges_=[[round(x,2) for x in hs], ns]
clustering_matrix = load_graph("hetero_clustering_matrix.pkl")
plot_heatmap(clustering_matrix, ranges_)'''

'''
h=0.2
n=80
t=500000
data = f"results_hetero/rho0_{n}000000.0_h_{round(h,2)}/rhot_{t}.txt"
#m =create_matrix_from_tsv(f"results_hetero/rho0_{n}000000.0_h_{round(h,2)}/rhot_{t}.txt")

#plot_heatmap(m)
evaluate(data, 50)
'''

'''
ns = range(40,85,5)
hs = np.arange(0.1, 1.0, 0.1)

plot_grid_of_heatmaps(ns, hs)
p = "hetero_clustering_matrix.pkl"
m = load_graph(p)
plot_heatmap_accurate(m, hs, ns)
'''


#calculate clustering metric on rho

