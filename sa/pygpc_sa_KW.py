"""
Algorithm: Static_IO
==============================
"""


# Windows users have to encapsulate the code into a main function to avoid multiprocessing errors.
def main():
    import pygpc
    import numpy as np
    import matplotlib

    # matplotlib.use("Qt5Agg")

    from collections import OrderedDict

    fn_results = "tmp/static_IO"  # filename of output
    save_session_format = (
        ".pkl"  # file format of saved gpc session ".hdf5" (slow) or ".pkl" (fast)
    )
    np.random.seed(1)

    # %%
    # Setup input and output data
    # ----------------------------------------------------------------------------------------------------------------

    # We artificially generate some coordinates for the input data the user has to provide where the model was sampled
    # LOAD YOUR SAMPLING POINTS IN THE NORMAL MODEL SPACE HERE (NO NORMALIZATION)
    n_grid = 150
    x1 = np.random.rand(n_grid) * 0.8 + 1.2
    x2 = 1.25
    x3 = np.random.rand(n_grid) * 0.6

    # define the properties of the random variables
    # DEFINE YOUR 4 RANDOM VARIABLES HERE
    parameters = OrderedDict()
    parameters["x1"] = pygpc.Beta(
        pdf_shape=[1, 1], pdf_limits=[1.2, 2]
    )  # uniform distribution
    parameters["x3"] = pygpc.Norm(pdf_shape=[1, 0.1])  # normal distribution

    # generate a grid object from the input data
    # my pygpc.RandomGrid does not accept coords, but options={"n_grid": n_grid, "seed": 1}
    grid = pygpc.RandomGrid(parameters_random=parameters, coords=np.vstack((x1, x3)).T)
    print(grid)
    # get output data (here: Peaks function)
    # LOAD YOUR RESULTS HERE shape: [150 x 2]
    results = (
        3.0 * (1 - x1) ** 2.0 * np.exp(-(x1**2) - (x3 + 1) ** 2)
        - 10.0 * (x1 / 5.0 - x1**3 - x3**5) * np.exp(-(x1**2) - x3**2)
        - 1.0 / 3 * np.exp(-((x1 + 1) ** 2) - x3**2)
    ) + x2
    # results = np.hstack((results[:, np.newaxis], 2*results[:, np.newaxis]))
    results = results[:, np.newaxis]

    # %%
    # Setting up the algorithm
    # ------------------------

    # gPC options
    options = dict()
    options["method"] = "reg"
    options["solver"] = "LarsLasso"
    options["settings"] = None
    options["order"] = [9, 9]
    options["order_max"] = 9
    options["interaction_order"] = 2  # 2 or 3 are sufficient
    options["error_type"] = "loocv"
    options["n_samples_validation"] = None
    options["fn_results"] = fn_results
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

    # n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=[6,10,15,19],
    #                                        order_glob_max=5,
    #                                        order_inter_max=2,
    #                                        dim=4,
    #                                        order_inter_current=None,
    #                                        order_glob_max_norm=1.0)

    # define algorithm
    algorithm = pygpc.Static_IO(
        parameters=parameters, options=options, grid=grid, results=results
    )

    # %%
    # Running the gpc
    # ---------------

    # initialize gPC Session
    session = pygpc.Session(algorithm=algorithm)

    # run gPC algorithm
    session, coeffs, results = session.run()

    # %%
    # Postprocessing
    # --------------

    # read session
    # session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)

    # Post-process gPC
    pygpc.get_sensitivities_hdf5(
        fn_gpc=options["fn_results"],
        output_idx=None,
        calc_sobol=True,
        calc_global_sens=True,
        calc_pdf=True,
        algorithm="standard",
    )

    # plot gPC approximation and IO data
    pygpc.plot_gpc(
        session=session,
        coeffs=coeffs,
        random_vars=["x1", "x3"],
        output_idx=0,
        n_grid=[100, 100],
        coords=grid.coords,
        results=results,
        fn_out=None,
    )

    sobol, gsens = pygpc.get_sens_summary(
        fn_results, parameters, "/home/kporzig/Desktop/sens_test.txt"
    )


# pygpc.plot_sens_summary(sobol, gsens, qois=np.arange(2), multiple_qoi=True, results=results)

# pygpc.plot_sens_summary(sobol, gsens)
# On Windows subprocesses will import (i.e. execute) the main module at start.
# You need to insert an if __name__ == '__main__': guard in the main module to avoid
# creating subprocesses recursively.
#
if __name__ == "__main__":
    main()
