#compute the size and position of clusters of worms, that is, neighboring points with worm density above a certain threshold
import numpy as np
from matplotlib import pyplot as plt


def get_clusters_size_and_position(W, threshold):
    clusters=[] #matrix of clusters, each cluster is a list of points
    clusters_size=[] #vector of the size of each cluster

    above_threshold_points = np.where(W>threshold)
    above_threshold_points = np.array(above_threshold_points).T
    checked_points = np.zeros((W.shape[0], W.shape[1]))
    for point in above_threshold_points:
        print("Point: "+str(point))
        if checked_points[point[0], point[1]]==0:
            connected_points = [point]
            checked_points[point[0], point[1]]=1
            for other_point in above_threshold_points:
                if np.linalg.norm(point-other_point)==1:
                    connected_points.append(other_point)
                    checked_points[other_point[0], other_point[1]] = 1

            clusters.append(connected_points)
            clusters_size.append(len(connected_points))

    return clusters_size, clusters

def get_std(W):
    return np.std(W)

def get_entropy(W):
    #select only positive values
    Wnotzero = W[W!=0]
    return -np.sum(Wnotzero * np.log2(Wnotzero))

target = "results/run52_AA/"
W0_vector = [(120 *10** 4, 90* 10 ** 2)]
O0_vector = [0.21001]
for (W_low, W_high) in W0_vector:
    for O0 in O0_vector:
        print("(W_high(0), O(0)): ", W_high, O0)
        W_0 = np.load(target + "W0_W0_" + str(round(W_high, 1)) + "O0_" + str(O0)[2:5]+".npy")
        W_tmax = np.load(target + "Wtmax_W0_" + str(round(W_high, 1)) + "O0_" + str(O0)[2:5]+".npy")
        im = plt.imshow(W_tmax, cmap='hot_r', interpolation='nearest', animated=True, vmin=0)
        cbar = plt.colorbar(im)
        cbar.set_label('Worm density at time tmax')
        plt.savefig(target + "Wtmax_W0_" + str(round(W_high, 1)) + "O0_" + str(O0)[2:5] + ".pdf", format='pdf')
        plt.show()
        normalized_W_0 = W_0/np.max(W_0)
        normalized_W_tmax = W_tmax/np.max(W_tmax)
        O_tmax = np.load(target + "Otmax_W0_" + str(round(W_high, 1)) + "O0_" + str(O0)[2:5]+".npy")
        print("average W0: "+str(np.average(W_0)))
        print("average W_tmax: "+str(np.average(W_tmax)))
        print("negative W values: "+str((W_tmax[W_tmax<0])))
        print("std(W0): "+str(get_std(W_0)))
        print("std(W_tmax): "+str(get_std(W_tmax)))
        print("entropy(W0): "+str(get_entropy(normalized_W_0)))
        print("entropy(W_tmax): "+str(get_entropy(normalized_W_tmax)))
        print("oxy at tmax: "+str(np.min(O_tmax)))
        print("initial number of worms: "+str(np.sum(W_0)))
        print("number of worms: "+str(np.sum(W_tmax)))

target = "simulation_results/"
W = np.loadtxt(target + "countAgent.csv",delimiter=",", dtype=str)
W = W.astype(float)
print(W)
print("average W: "+str(np.average(W)))
print("std(W): "+str(get_std(W)))
normalized_W = W/np.max(W)
print("entropy(W): "+str(get_entropy(normalized_W)))
#im = plt.imshow(W, cmap='hot', interpolation='nearest', animated=True)
#cbar = plt.colorbar(im)
#cbar.set_label('Worm density at time tmax')
#plt.show()