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

def check_instability_criterion(W, O, a, b, c, tau, f, kc, Oam):
    V = a * O ** 2 + b * O + c
    dV = 2 * a * O + b
    # verify instability criterion for all points: W beta kc > f Dw
    beta = 1 / (2 * tau) * V * (2 * a * O + b)
    Dw = 1 / (2 * tau) * V ** 2
    print("W beta kc: ", W * beta * kc)
    print("f Dw: ", f * Dw)
    print("instability criterion: ", (W * beta * kc > f * Dw).all())
    print("O: ", O)
    O_eq = Oam - kc / f * W
    print("O_eq: ", O_eq)
    #show difference with O
    print("O_eq - O: ", O_eq - O)
    #count the number of equal points out of all points
    print("number of equal points: ", np.sum(O_eq == O))


def get_std(W):
    return np.std(W)

def get_entropy(W):
    #select only positive values
    Wnotzero = W[W!=0]
    return -np.sum(Wnotzero * np.log2(Wnotzero))

target = "results/run69/"
W0_vector = [(120 *10** 4, 90* 10 ** 2)]
O0_vector = [0.21001]
for (W_low, W_high) in W0_vector:
    for O0 in O0_vector:
        print("(W_high(0), O(0)): ", W_high, O0)
        W_0 = np.load(target + "W0_W0_" + str(round(W_high, 1)) + "O0_" + str(O0)[2:5]+".npy")
        W_tmax = np.load(target + "Wtmax_W0_" + str(round(W_high, 1)) + "O0_" + str(O0)[2:5]+".npy")
        im = plt.imshow(W_tmax, cmap='hot_r', interpolation='nearest', animated=True)
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
        print(W_tmax)
        #transform worm density into worm count
        W=W_tmax*4/(128*128)
        im = plt.imshow(W, cmap='hot_r', interpolation='nearest', animated=True, vmin=0)
        cbar = plt.colorbar(im)
        cbar.set_label('Worm amount at time tmax')
        plt.show()
        check_instability_criterion(W_tmax, O_tmax, 1.90, -3.98 * 10 ** (-1), 2.25 * 10 ** (-2), 0.5, 0.65, 7.3 * 10 ** (-10), 0.21)

target = "simulation_results/"
W = np.loadtxt(target + "countAgent.csv",delimiter=",", dtype=str)
W = W.astype(float)
print(W)
print("average W: "+str(np.average(W)))
print("std(W): "+str(get_std(W)))
print("max worm density: ", np.max(W))
normalized_W = W/np.max(W)
print("entropy(W): "+str(get_entropy(normalized_W)))
#im = plt.imshow(W, cmap='hot', interpolation='nearest', animated=True)
#cbar = plt.colorbar(im)
#cbar.set_label('Worm density at time tmax')
#plt.show()