def main():
    import pygpc
    import pandas as pd
    import numpy as np
    from collections import OrderedDict

    fn_results = "tmp/static_IO_gm"  # filename of output
    save_session_format = (
        ".pkl"  # file format of saved gpc session ".hdf5" (slow) or ".pkl" (fast)
    )

    # load samples
    samplesXLSX = pd.read_excel("sa/01_in_ishigami1000.xlsx")
    samples = samplesXLSX.loc[:, ["x1", "x2", "x3"]].to_numpy()

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
    grid = pygpc.RandomGrid(
        parameters_random=parameters,
        coords=np.vstack((samples.T[0], samples.T[1], samples.T[2])).T,
    )
    # grid = pygpc.RandomGrid(
    #     parameters_random=parameters, options={"n_grid": Ns, "seed": 1}
    # )

    modelEval = samplesXLSX.loc[:, "out1"].to_numpy()
    results = modelEval[:, np.newaxis]

    # gPC options
    options = dict()
    options["method"] = "reg"
    options["solver"] = "LarsLasso"
    options["settings"] = None
    options["order"] = [10, 10, 10]
    options["order_max"] = 10
    options["interaction_order"] = M_dim
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
    print("n_coeffs:", n_coeffs)  # same as UQtab

    # define algorithm
    algorithm = pygpc.Static_IO(
        parameters=parameters, options=options, grid=grid, results=results
    )

    # initialize gPC Session
    session = pygpc.Session(algorithm=algorithm)

    # run gPC algorithm
    session, coeffs, results = session.run()

    pygpc.get_sensitivities_hdf5(
        fn_gpc=options["fn_results"],
        output_idx=None,
        calc_sobol=True,
        calc_global_sens=True,
        calc_pdf=True,
        algorithm="standard",
    )

    pygpc.plot_gpc(
        session=session,
        coeffs=coeffs,
        random_vars=["x1", "x2", "x3"],
        output_idx=0,
        n_grid=[Ns, Ns],
        coords=grid.coords,
        results=results,
        fn_out=None,
    )


if __name__ == "__main__":
    main()
