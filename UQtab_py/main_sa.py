import pandas as pd
import numpy as np
from userData import OneInput, OneOutput, Hidden
from pce import Pce
from sensitivityAnalysis import SensitivityAnalysis


class SEGAsensitivity(object):
    def computeSobolIndices(self, input, output):
        inputs = [OneInput(row) for row in input]
        inputPDF = []
        inputPDF.extend([input.pdf for input in inputs])
        u_matrix = []
        u_matrix.extend([input.u_vector.u for input in inputs])
        outputs = [OneOutput(output)]

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


class Criteria(object):
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


sensitivity = SEGAsensitivity()

samplesXLSX = pd.read_excel("sa/01_in_ishigami1000.xlsx")
modelEval = samplesXLSX.loc[:, "out1"].to_numpy()
samples = np.transpose(samplesXLSX.loc[:, ["x1", "x2", "x3"]].to_numpy())

sa = sensitivity.computeSobolIndices(samples, modelEval)

p1 = Criteria.computeP1(sa, len(samples))
p2 = Criteria.computeP2(sa)
