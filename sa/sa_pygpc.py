import math
import pygpc
import pandas as pd
import numpy as np
import matplotlib
from collections import OrderedDict

save_session_format = (
    ".pkl"  # file format of saved gpc session ".hdf5" (slow) or ".pkl" (fast)
)

# load samples
samplesXLSX = pd.read_excel("sa/01_in_ishigami1000.xlsx")
samples = samplesXLSX.loc[:, ["x1", "x2", "x3"]].to_numpy()
modelEval = samplesXLSX.loc[:, "out1"].to_numpy()

Ns = len(samples)
M_dim = len(samples.T)
# print(samples.T[0])

# define input pdfs
parameters = OrderedDict()
parameters["x1"] = pygpc.Beta(
    pdf_shape=[1, 1], pdf_limits=[min(samples.T[0]), max(samples.T[0])]
)  # uniform distribution
parameters["x2"] = pygpc.Beta(
    pdf_shape=[1, 1], pdf_limits=[min(samples.T[1]), max(samples.T[1])]
)  # uniform distribution
parameters["x3"] = pygpc.Beta(
    pdf_shape=[1, 1], pdf_limits=[min(samples.T[2]), max(samples.T[2])]
)  # uniform distribution

# CREATE A GRID USING YOUR SAMPLING POINTS
grid = pygpc.RandomGrid(parameters_random=parameters, options={"n_grid": Ns, "seed": 1})


# gPC options
options = dict()
options["method"] = "reg"
options["solver"] = "LarsLasso"
options["settings"] = None
options["order"] = [10, 10, 10]
options["order_max"] = 10
options["interaction_order"] = 3
options["error_type"] = "loocv"
options["n_samples_validation"] = None
options["fn_results"] = modelEval
options["save_session_format"] = save_session_format
options["backend"] = "omp"
options["verbose"] = True

# estimate the number of gPC coefficients
n_coeffs = pygpc.get_num_coeffs_sparse(
    order_dim_max=options["order"],
    order_glob_max=options["order_max"],
    order_inter_max=options["interaction_order"],
    dim=len(parameters),
    order_inter_current=None,
    order_glob_max_norm=1.0,
)
print(n_coeffs)  # same as UQtab

# define algorithm
algorithm = pygpc.Static_IO(
    parameters=parameters, options=options, grid=grid, results=modelEval
)

# initialize gPC Session
session = pygpc.Session(algorithm=algorithm)

# run gPC algorithm
session, coeffs, results = session.run()
print(session)
