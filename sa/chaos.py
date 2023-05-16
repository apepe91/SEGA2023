import math
import numpy as np
import chaospy as cp
from matplotlib import pyplot


# class Solution(object):
class SobolIndices:
    @staticmethod
    def get_firstSobol(multiAlpha, PCE_coeff):
        totVariance_system = SobolIndices.get_totalVariance_fromPCE(PCE_coeff)
        print('totVariance_system: ', totVariance_system)
        sensitivity_indices = []

        for m in range(len(np.transpose(multiAlpha))-1):
            cumSum_PCEcoeff = []

            for alphaRow in range(1, len(multiAlpha)):
                if multiAlpha[alphaRow][m] != 0:
                    otherRows = []
                    for index, row in enumerate(multiAlpha):
                        if index != m and multiAlpha[m][cA] != 0:
                            otherRows.append(row[cA])

                    sum_otherRows = sum(otherRows)

                    if sum_otherRows == 0:
                        cumSum_PCEcoeff.append(PCE_coeff[alphaRow] ** 2)

                    if len(cumSum_PCEcoeff) == len(multiAlpha):
                        break

            sensitivity_indices.append(sum(cumSum_PCEcoeff))

        return [element / totVariance_system for element in sensitivity_indices]

    @staticmethod
    def get_totalSobol(multiAlpha, PCE_coeff):
        totVariance_system = SobolIndices.get_totalVariance_fromPCE(PCE_coeff)
        sensitivity_indices = []

        for m in range(len(multiAlpha)):
            cumSum_PCEcoeff = []

            for cA in range(1, len(PCE_coeff)):
                if multiAlpha[m][cA] != 0:
                    cumSum_PCEcoeff.append(PCE_coeff[cA] ** 2)

            sensitivity_indices.append(sum(cumSum_PCEcoeff))

        return [element / totVariance_system for element in sensitivity_indices]

    @staticmethod
    def get_totalVariance_fromPCE(PCE_coeff):
        PCE_coeff_without_first = PCE_coeff[1:]
        return sum(a ** 2 for a in PCE_coeff_without_first)


# solver = Solution()

# define function for testing
def ishigamiFunction(x):
    x1, x2, x3 = x
    return math.sin(x1) + 7*math.sin(x2)*math.sin(x2) + 0.1 * x3 ** 4*math.sin(x1)


# define ishigami function parameters as input
x1 = cp.Uniform(-math.pi, math.pi)
x2 = cp.Uniform(-math.pi, math.pi)
x3 = cp.Uniform(-math.pi, math.pi)
jointInputDistr = cp.J(x1, x2, x3)
samples = jointInputDistr.sample(1000, 'latin_hypercube')
# print('samples: ', samples)

# generate orthogonal basis polynomials
pce_degree = 3
expansion = cp.generate_expansion(pce_degree, jointInputDistr)

modelEval = np.array([ishigamiFunction(sample)
                      for sample in np.transpose(samples)])
# pyplot.plot(evaluations)
# pyplot.show()

# create polynomial approximation
pceModel = cp.fit_regression(expansion, samples, modelEval)
# pyplot.plot(approx_solver(*samples))
# pyplot.show()
poly = cp.polynomial(pceModel)
pceCoeff = poly.coefficients
print('coeff: ', pceCoeff)
pce_multiAlpha = poly.exponents
print('exponents: ', pce_multiAlpha)

sobol = SobolIndices()
firstSobol = sobol.get_firstSobol(pce_multiAlpha, pceCoeff)
print('First Sobol indices: ', firstSobol)
