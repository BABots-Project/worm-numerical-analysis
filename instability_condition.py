import random

from sympy import symbols, solve, Interval, Union
from swarming_simulator import updateParameters

def criterion(sigma, beta_a, beta_r, alfa_a, alfa_r, D_a, D_r, gamma_a, gamma_r, s_a, s_r):

    #ks = [0.0001, 1, 10000]
    solutions = []
    rho = symbols('rho')

    #for k in ks:
        #inequality = -sigma>rho*((-beta_a/(alfa_a+s_a*rho/gamma_a)*s_a)/(gamma_a+D_a*k**2)-beta_r/(alfa_r+s_r*rho/gamma_r)*s_r/(gamma_r+D_r*k**2))
    inequality = -sigma*(gamma_a+gamma_r)>rho*((-beta_a/(alfa_a+s_a*rho/gamma_a)*s_a)-beta_r/(alfa_r+s_r*rho/gamma_r)*s_r)

    solution = solve(inequality, rho)
    set_solution = solution.as_set()
    #print(set_solution)
    #access intervals of set
    if type(set_solution)==Interval:
        start = set_solution.start
        end = set_solution.end
        if end > 0:
            solutions.append(set_solution)
    elif type(set_solution)==Union:
        start = []
        end = []
        for interval in set_solution.args:
            start.append(interval.start)
            end.append(interval.end)
            if interval.end > 0:
                solutions.append(interval)
        #print(start, end)
        #check if the interval/intervals is/are real
        #if it is, append the solution to the solutions list
        #note: it's ok if start is negative as long as end is positive

    return solutions
'''
sigma, scale, rho_max, cushion, dt, beta_a, beta_r, alfa_a, alfa_r, D_a, D_r, gamma_a, gamma_r, s_a, s_r = updateParameters("parameters_swarming_copy.txt")
solutions = criterion(sigma, beta_a, beta_r, alfa_a, alfa_r, D_a, D_r, gamma_a, gamma_r, s_a, s_r)
picked_solution = random.choice(solutions)
start = picked_solution.start
end = picked_solution.end
print(start, end)
if end.is_infinite:
    print(random.uniform(start, 10e9))
print(random.uniform(start, end))
'''
#print(solutions)
