import math
import pandas as pd
import numpy as np
import chaospy as cp


class SobolIndices:
    @staticmethod
    def firstSobol(multiAlpha, pceCoeff):
        totVar = SobolIndices.totalVar_fromPCE(pceCoeff)
        indices = []

        for m in range(len(np.transpose(multiAlpha))):
            cumSum = []

            for alphaRow in range(1, len(multiAlpha)):
                if multiAlpha[alphaRow][m] != 0:
                    if (
                        sum([x for i, x in enumerate(multiAlpha[alphaRow]) if i != m])
                        == 0
                    ):
                        cumSum.append(pceCoeff[alphaRow] ** 2)

                    if len(cumSum) == len(multiAlpha):
                        break

            indices.append(sum(cumSum))

        return [element / totVar for element in indices]

    @staticmethod
    def totalSobol(multiAlpha, pceCoeff):
        totVar = SobolIndices.totalVar_fromPCE(pceCoeff)
        indices = []

        for m in range(len(np.transpose(multiAlpha))):
            cumSum = []

            for alphaRow in range(1, len(multiAlpha)):
                if multiAlpha[alphaRow][m] != 0:
                    cumSum.append(pceCoeff[alphaRow] ** 2)

            indices.append(sum(cumSum))

        return [element / totVar for element in indices]

    @staticmethod
    def totalVar_fromPCE(PCE_coeff):
        PCE_coeff_without_first = PCE_coeff[1:]
        return sum(a**2 for a in PCE_coeff_without_first)


samplesXLSX = pd.read_excel("sa/01_in_ishigami1000.xlsx")
modelEval = samplesXLSX.loc[:, "out1"].to_numpy()
samples = np.transpose(samplesXLSX.loc[:, ["x1", "x2", "x3"]].to_numpy())
# print(modelEval)
# print(samples)


# define ishigami function parameters as input
x1 = cp.Uniform(-math.pi, math.pi)
x2 = cp.Uniform(-math.pi, math.pi)
x3 = cp.Uniform(-math.pi, math.pi)
jointInputDistr = cp.J(x1, x2, x3)
# samplesCP = jointInputDistr.sample(1000, "latin_hypercube")
# print("samples: ", samplesCP)

# generate orthogonal basis polynomials
pce_degree = 10
expansion = cp.generate_expansion(pce_degree, jointInputDistr)

# modelEvalCP = np.array([ishigamiFunction(sample) for sample in np.transpose(samplesCP)])
# print(modelEvalCP)
# pyplot.show()

# create polynomial approximation
pceModel = cp.fit_regression(expansion, samples, modelEval)
# pyplot.plot(approx_solver(*samples))
# pyplot.show()
poly = cp.polynomial(pceModel)
pceCoeff = poly.coefficients
print("coeff: ", pceCoeff)
pce_multiAlpha = poly.exponents
# print("exponents: ", pce_multiAlpha)

sobol = SobolIndices()
firstSobol = sobol.firstSobol(pce_multiAlpha, pceCoeff)
totalSobol = sobol.totalSobol(pce_multiAlpha, pceCoeff)
print("First Sobol indices: ", firstSobol)
print("Total Sobol indices: ", totalSobol)
