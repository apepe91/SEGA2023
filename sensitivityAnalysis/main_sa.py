import pandas as pd
import numpy as np
from userData import OneInput, OneOutput, Hidden
from pce import Pce
from sensitivityAnalysis import SensitivityAnalysis
from statisticalOp import Stati


class SEGAsensitivity(object):
    def computeSobolIndices(self, input, output):
        inputs = [OneInput(row) for row in input]
        inputPDF = []
        inputPDF.extend([input.pdf for input in inputs])
        u_matrix = []
        u_matrix.extend([input.u_vector.u for input in inputs])
        # outputs = [OneOutput(output)]

        hidden = [Hidden(input)]

        pce = Pce(input, output, u_matrix, inputPDF, hidden[0].pceDegree)

        result = SensitivityAnalysis(input, output, pce)
        SEGAsensitivity.printResults(self, result)
        return result

    def printResults(self, sensitivityResults):
        print(
            "First Sobol indices:",
            sensitivityResults.sensAnalysis[0].indices[0].firstSobol,
        )
        print(
            "Total Sobol indices:",
            sensitivityResults.sensAnalysis[0].indices[0].totalSobol,
        )


class SEGAcriteria(object):
    def computeP1(sensitivityIndices, inputDim):
        p1Sum = 0
        for index in sensitivityIndices.sensAnalysis[0].indices[0].firstSobol:
            p1Sum += abs(index - 1 / inputDim)
        return 1 - p1Sum

    def computeP2(sensitivityIndices):
        firstSobol = sensitivityIndices.sensAnalysis[0].indices[0].firstSobol
        totalSobol = sensitivityIndices.sensAnalysis[0].indices[0].totalSobol
        p2Sum = sum((a - b) for a, b in zip(totalSobol, firstSobol))
        return 1 - p2Sum

    def computeP3(values):
        weights = [0.6, 0.25, 0.15]
        mode_gm = Stati.mode(values)
        mean = Stati.mean(values)
        var_gm = Stati.sampleVar(values)
        std = Stati.sampleStd(values)
        skewNum = 0
        for value in values:
            skewNum += (value - mean) ** 3
        skew_gm = skewNum / ((len(values) - 1) * std**3)

        return weights[0] / mode_gm + weights[1] / var_gm + weights[2] / abs(skew_gm)

    def computeP4(values):
        weights = [0.6, 0.25, 0.15]
        mode_gm = Stati.mode(values)
        mean = Stati.mean(values)
        var_gm = Stati.sampleVar(values)
        std = Stati.sampleStd(values)
        skewNum = 0
        for value in values:
            skewNum += (value - mean) ** 3
        skew_gm = skewNum / ((len(values) - 1) * std**3)

        return weights[0] * mode_gm + weights[1] / var_gm + weights[2] / abs(skew_gm)


sensitivity = SEGAsensitivity()

samplesXLSX = pd.read_excel("test_fun/01_in_ishigami1000.xlsx")
modelEval = samplesXLSX.loc[:, "out1"].to_numpy()
samples = np.transpose(samplesXLSX.loc[:, ["x1", "x2", "x3"]].to_numpy())

sa = sensitivity.computeSobolIndices(samples, modelEval)

p1 = SEGAcriteria.computeP1(sa, len(samples))
p2 = SEGAcriteria.computeP2(sa)
p3_test = pd.read_excel("test_fun/p3_testing.xlsx")
p3values_test = np.transpose(p3_test.loc[:, ["good", "bad"]].to_numpy())
p4_test = pd.read_excel("test_fun/p3_testing.xlsx")
p4values_test = np.transpose(p4_test.loc[:, ["good", "bad"]].to_numpy())
p3_good = SEGAcriteria.computeP3(p3values_test[0])
print("p3good:", p3_good)
p3_bad = SEGAcriteria.computeP3(p3values_test[1])
print("p3bad:", p3_bad)
p4_good = SEGAcriteria.computeP4(p4values_test[0])
print("p4good:", p4_good)
p4_bad = SEGAcriteria.computeP4(p4values_test[1])
print("p4bad:", p4_bad)
