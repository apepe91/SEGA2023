import math
import numpy as np
import chaospy as cp
from matplotlib import pyplot

# define function for testing


def ishigamiFunction(x):
    x1, x2, x3 = x
    return math.sin(x1) + 7*math.sin(x2)*math.sin(x2) + 0.1 * x3 ** 4*math.sin(x1)


# define ishigami function parameters as input
x1 = cp.Uniform(-math.pi, math.pi)
x2 = cp.Normal(0, math.pi)
x3 = cp.Uniform(-math.pi, math.pi)
joint = cp.J(x1, x2, x3)
samples = joint.sample(1000, 'latin_hypercube')
# print('samples: ', samples)

# generate orthogonal basis polynomials
expansion = cp.generate_expansion(3, joint)

evaluations = np.array([ishigamiFunction(sample)
                       for sample in np.transpose(samples)])
# pyplot.plot(evaluations)
# pyplot.show()

# create polynomial approximation
approx_solver = cp.fit_regression(expansion, samples, evaluations)
# pyplot.plot(approx_solver(*samples))
# pyplot.show()
poly = cp.polynomial(approx_solver)
coeff = poly.coefficients
print('coeff: ', coeff)
exponents = poly.exponents
print('exponents: ', exponents)
