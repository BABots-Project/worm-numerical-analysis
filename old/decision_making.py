from matplotlib import pyplot as plt

import numerical_solver as ns
import numpy as np

parameters = {
    'c_min': 0.3,
    'epsilon': 0.01,
    'k': 1e3,
    'Da': 1e-9,
    'Db': 1e-9,
    'gamma_a': 1e-4,
    'gamma_b': 1e-4,
    'delta' : 0.01
}

N = ns.N
l = ns.l

def initial_conditions(wa0, wb0, size_of_spawn, size_of_a, size_of_b, ca0, cb0):
    wa = np.zeros((N, N))
    wb = np.zeros((N, N))
    ca = np.zeros((N, N))
    cb = np.zeros((N, N))

    #place wa0, wb0 on the grid at the center of the quadrants top-right and bottom-left based on the spawn size
    #start by defining the center point of the top-right quadrant:
    top_right_center = int(N/4), int(3*N/4)
    #define the area centered at the top-right quadrant center + the size of the spawn
    index_spawn_x1 = int(top_right_center[0] - size_of_spawn/2) #N/4 - N/4 = 0 | N/4-N/8 = N/8
    index_spawn_x2 = int(top_right_center[0] + size_of_spawn/2) #3N/4 + N/4 = N |
    index_spawn_y1 = int(top_right_center[1] - size_of_spawn/2) #3N/4 - N/4 = N/2
    index_spawn_y2 = int(top_right_center[1] + size_of_spawn/2) #3N/4 + N/4 = 5N/4
    wa[index_spawn_x1:index_spawn_x2, index_spawn_y1:index_spawn_y2] = wa0
    wb[index_spawn_x1:index_spawn_x2, index_spawn_y1:index_spawn_y2] = wb0

    #same for the bottom-left quadrant
    bottom_left_center = int(3*N/4), int(N/4)
    index_spawn_x1 = int(bottom_left_center[0] - size_of_spawn/2)
    index_spawn_x2 = int(bottom_left_center[0] + size_of_spawn/2)
    index_spawn_y1 = int(bottom_left_center[1] - size_of_spawn/2)
    index_spawn_y2 = int(bottom_left_center[1] + size_of_spawn/2)
    wa[index_spawn_x1:index_spawn_x2, index_spawn_y1:index_spawn_y2] = wa0
    wb[index_spawn_x1:index_spawn_x2, index_spawn_y1:index_spawn_y2] = wb0

    #u will be 1 - wa - wb
    u = 1 - wa - wb

    #set to 1 the points of ca and cb at the center of the quadrants at the top-left and bottom-right based on the sizes
    #top-left
    top_left_center = int(N/4), int(N/4)
    index_a_x1 = int(top_left_center[0] - size_of_a/2)
    index_a_x2 = int(top_left_center[0] + size_of_a/2)
    index_a_y1 = int(top_left_center[1] - size_of_a/2)
    index_a_y2 = int(top_left_center[1] + size_of_a/2)
    ca[index_a_x1:index_a_x2, index_a_y1:index_a_y2] = ca0

    #bottom-right
    bottom_right_center = int(3*N/4), int(3*N/4)
    index_b_x1 = int(bottom_right_center[0] - size_of_b/2)
    index_b_x2 = int(bottom_right_center[0] + size_of_b/2)
    index_b_y1 = int(bottom_right_center[1] - size_of_b/2)
    index_b_y2 = int(bottom_right_center[1] + size_of_b/2)
    cb[index_b_x1:index_b_x2, index_b_y1:index_b_y2] = cb0

    return wa, wb, u, ca, cb

def f(c, k, c_min):
    #sigmoid function 1/(1+exp(-k(c-c_min)))
    #check for overflow
    return 1/(1+np.exp(-k*(c-c_min)))



def run():
    wa0 = 0.5
    wb0 = 0.5
    size_of_spawn = N/4
    size_of_a = N/4
    size_of_b = N/4
    wa, wb, u, ca, cb = initial_conditions(wa0, wb0, size_of_spawn, size_of_a, size_of_b)

    dt = 0.01
    t = 0
    step_max = 1000
    step=0

    c_min = parameters['c_min']
    epsilon = parameters['epsilon']
    k = parameters['k']
    Da = parameters['Da']
    Db = parameters['Db']
    gamma_a = parameters['gamma_a']
    gamma_b = parameters['gamma_b']
    delta = parameters['delta']

    while step < step_max:
        m = delta-np.abs(ca-cb)
        try:
            f1 = f(ca, k, c_min)
            f2 = f(cb, k, c_min)
            f3 = f(m, k, epsilon)
        except:
            print('ca', ca)
            print('cb', cb)
            print('m', m)
            print('k', k)
            print('c_min', c_min)
            print('epsilon', epsilon)
            #print('f1', f1)
            #print('f2', f2)
            #print('f3', f3)
            break
        #check no of in f1,2,3:
        if np.isnan(f1).any(): #or np.isnan(f2).any() or np.isnan(f3).any():
            print('f1', f1)
            print('f2', f2)
            print('f3', f3)
            break
        pa = f1*f2*f3*wa/(wa+wb+1e-6)
        pb = (1-f1)*(1-f2)*f3*wb/(wa+wb+1e-12)
        r = 1-pa-pb
        #sample probability from -r/2 to r/2
        ra = r/2 - np.random.rand(N, N)*r
        rb = r/2 - ra
        dwa = pa*u - pb *wa + ra*wa
        dwb = pb*u - pa *wb + rb*wb
        dca = Da*ns.laplacian(ca) - gamma_a*ca
        dcb = Db*ns.laplacian(cb) - gamma_b*cb
        wa += dwa*dt
        wb += dwb*dt
        ca += dca*dt
        cb += dcb*dt
        u = 1 - wa - wb
        t += dt
        step+=1

    #plot ca and f1 one next to each other
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(wa, cmap='rainbow', interpolation='nearest')
    axs[1].imshow(wb, cmap='rainbow', interpolation='nearest')
    plt.show()

run()