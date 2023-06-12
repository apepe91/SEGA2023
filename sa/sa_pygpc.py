import math
import pygpc
import pandas as pd
import numpy as np


samplesXLSX = pd.read_excel("sa/01_in_ishigami1000.xlsx")
samples = np.transpose(samplesXLSX.loc[:, ["x1", "x2", "x3"]].to_numpy())

# collect samples and model output
p = dict()
p["x1"] = samples[0]
p["x2"] = samples[1]
p["x3"] = samples[2]
modelEval = samplesXLSX.loc[:, "out1"].to_numpy()
problem = pygpc.Problem()

# gPC options
options = dict()
options["method"] = "reg"
options["solver"] = "Moore-Penrose"
options["settings"] = None
options["order"] = [9, 9]
options["order_max"] = 9
options["interaction_order"] = 2
options["matrix_ratio"] = 2
options["error_type"] = "nrmsd"
options["n_samples_validation"] = 1e3
options["n_cpu"] = 0
options["fn_results"] = None
options["gradient_enhanced"] = True
options["gradient_calculation"] = "FD_1st2nd"
options["gradient_calculation_options"] = {"dx": 0.05, "distance_weight": -2}
options["backend"] = "omp"
options["grid"] = pygpc.Random
options["grid_options"] = None

# determine number of basis functions
n_coeffs = pygpc.get_num_coeffs_sparse(
    order_dim_max=options["order"],
    order_glob_max=options["order_max"],
    order_inter_max=options["interaction_order"],
    dim=3,
)

# generate grid
grid = pygpc.Random(
    parameters_random=problem.parameters_random,
    n_grid=options["matrix_ratio"] * n_coeffs,
    options={"seed": 1},
)

# initialize algorithm
algorithm = pygpc.Static(problem=problem, options=options, grid=grid)

# initialize gPC Session
session = pygpc.Session(algorithm=algorithm)

# run gPC algorithm
session, coeffs, results = session.run()
