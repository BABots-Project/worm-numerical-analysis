#compute the size and position of clusters of worms, that is, neighboring points with worm density above a certain threshold
import numpy as np


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


target = "results/run4/"
W0_vector = [(10 ** 6, i * 10 ** 6) for i in range(20, 121, 20)]
O0_vector = [0.042 * (i + 1) for i in range(0, 5)]
for (W_low, W_high) in W0_vector:
    for O0 in O0_vector:
        print("(W_high(0), O(0)): ", W_high, O0)
        W_0 = np.load(target + "W0_W0_" + str(round(W_high, 1)) + "O0_" + str(O0)[2:5]+".npy")
        W_tmax = np.load(target + "Wtmax_W0_" + str(round(W_high, 1)) + "O0_" + str(O0)[2:5]+".npy")
        print("average W0: "+str(np.average(W_0)))
        print("average W_tmax: "+str(np.average(W_tmax)))
        print("std(W0): "+str(get_std(W_0)))
        print("std(W_tmax): "+str(get_std(W_tmax)))
