import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

import swarming_simulator


# Mocking the swarming_simulator for demonstration purposes
def swarming_simulator_main(args):
    accuracy, time = swarming_simulator.main(args)
    return time/(500000*0.01), accuracy

# Define parameter ranges
parameter_ranges = {
    "sigma": (5.555e-10, 5.555e-6),
    "scale": (0, 20),
    "cushion": (2e5, 2e9),
    "beta_a": (1.111e-11, 1.111e-7),
    "beta_r": (-1.111e-7, -1.111e-11),
    "alpha_a": (15e4, 15e8),
    "alpha_r": (15e4, 15e8),
    "D_a": (1.111e-12, 1.111e-8),
    "D_r": (1.111e-11, 1.111e-7),
    "gamma_a": (1e-4, 1),
    "gamma_r": (1e-5, 1e-1),
    "s_a": (1, 1e4),
    "s_r": (1e-1, 1e3),
    "rho0": (10e6, 600e6),
}

# Custom Problem Definition
class SwarmingOptimizationProblem(ElementwiseProblem):

    def __init__(self, parameter_ranges):
        self.param_keys = list(parameter_ranges.keys())
        self.bounds = np.array([parameter_ranges[key] for key in self.param_keys])
        print(self.bounds)
        super().__init__(n_var=len(self.param_keys),
                         n_obj=2,
                         n_constr=0,
                         xl=np.array(self.bounds[:, 0]),
                         xu=np.array(self.bounds[:, 1]))

    def _evaluate(self, x, out, *args, **kwargs):
        gen, i = args  # Generation and individual index
        parameter_dir = "../parameters_swarming_optimised.json"

        # Prepare args for the simulator
        sim_args = [gen, i, parameter_dir]
        print("Generation %d, Individual %d" % (gen, i))
        # Call the simulator (mocked here)
        time, accuracy = swarming_simulator_main(sim_args)

        # Set the objectives
        out["F"] = [time, -accuracy]  # Minimize time, maximize accuracy

# Initialize the problem
problem = SwarmingOptimizationProblem(parameter_ranges)

# Set up the algorithm
algorithm = NSGA2(pop_size=100)

# Perform the optimization
res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=True)

# Visualize the results
plot = Scatter()
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()
