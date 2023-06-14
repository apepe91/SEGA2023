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

        sensitivityResults = SensitivityAnalysis(input, output, pce)
        SEGAsensitivity.printResults(self, sensitivityResults)

    def printResults(self, sensitivityResults):
        print(
            "First Sobol indices:",
            sensitivityResults.sensAnalysis[0].indices[0].firstSobol,
        )
        print(
            "Total Sobol indices:",
            sensitivityResults.sensAnalysis[0].indices[0].totalSobol,
        )


sensitivity = SEGAsensitivity()

samplesXLSX = pd.read_excel("sa/01_in_ishigami1000.xlsx")
modelEval = samplesXLSX.loc[:, "out1"].to_numpy()
samples = np.transpose(samplesXLSX.loc[:, ["x1", "x2", "x3"]].to_numpy())

result = sensitivity.computeSobolIndices(samples, modelEval)
