import numpy as np
import pyapprox as pya
from pyapprox.benchmarks.benchmarks import setup_benchmark
from pyapprox.approximate import approximate

benchmark = setup_benchmark("ishigami", a=7, b=0.1)

num_samples = 1000
train_samples = pya.generate_independent_random_samples(benchmark.variable, num_samples)
train_vals = benchmark.fun(train_samples)

approx_res = approximate(
    train_samples,
    train_vals,
    "polynomial_chaos",
    {
        "basis_type": "hyperbolic_cross",
        "variable": benchmark.variable,
        "options": {"max_degree": 8},
    },
)
pce = approx_res.approx

res = pya.analyze_sensitivity_polynomial_chaos(pce)
