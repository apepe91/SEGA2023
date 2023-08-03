import pandas as pd
import numpy as np
from userData import OneInput, Hidden
from pce import Pce
from sensitivityAnalysis import SensitivityAnalysis
from statisticalOp import Stati
import matplotlib.pyplot as plt


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
        weights = [0.8, 0.17, 0.03]
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
        weights = [0.95, 0.04, 0.01]
        mode_gm = Stati.mode(values)
        mean = Stati.mean(values)
        var_gm = Stati.sampleVar(values)
        std = Stati.sampleStd(values)
        skewNum = 0
        for value in values:
            skewNum += (value - mean) ** 3
        skew_gm = skewNum / ((len(values) - 1) * std**3)

        return weights[0] * mode_gm + weights[1] / var_gm + weights[2] / abs(skew_gm)

    def computePJ(values, invalid_elements):
        weights = [0.5, 0.25, 0.05, 0.2]
        mode_gm = Stati.mode(values)
        mean = Stati.mean(values)
        var_gm = Stati.sampleVar(values)
        std = Stati.sampleStd(values)
        skewNum = 0
        for value in values:
            skewNum += (value - mean) ** 3
        skew_gm = skewNum / ((len(values) - 1) * std**3)

        return (
            weights[0] * mode_gm
            + weights[1] / var_gm
            + weights[2] / abs(skew_gm)
            + weights[3] / Stati.mean(invalid_elements)
        )


# p3_phase2 = pd.read_excel("test_fun/pj_phase2_SEGA23.xlsx")
# p3values_test = np.transpose(
#     p3_phase2.loc[:, ["tpvagenas", "iwm", "tpvagenas_inv", "iwm_inv"]].to_numpy())
# p3_tpvagenas = SEGAcriteria.computePJ(p3values_test[0], p3values_test[2])
# p3_iwm = SEGAcriteria.computePJ(p3values_test[1], p3values_test[3])
# print("pJ_tpvagens:", p3_tpvagenas)
# print("pJ_iwm:", p3_iwm)
# print("========================")

# p3_phase2 = pd.read_excel("test_fun/p3_phase2_SEGA23.xlsx")
# p3values_test = np.transpose(
#     p3_phase2.loc[:, ["tpvagenas", "iwm", "sunshine", "test"]].to_numpy())
# p3_tpvagenas = SEGAcriteria.computeP3(p3values_test[0])
# p3_iwm = SEGAcriteria.computeP3(p3values_test[1])
# p3_sunshine = SEGAcriteria.computeP3(p3values_test[2])
# p3_test = SEGAcriteria.computeP3(p3values_test[3])
# print("p3_hd_tpvagens:", p3_tpvagenas)
# print("p3_hd_iwm:", p3_iwm)
# print("p3_hd_sunshine:", p3_sunshine)
# print("p3_hd_test:", p3_test)
# print("========================")

# p4_phase2 = pd.read_excel("test_fun/p4_phase2_SEGA23.xlsx")
# p4values_test = np.transpose(
#     p4_phase2.loc[:, ["tpvagenas", "iwm", "sunshine", "test"]].to_numpy())
# p4_tpvagenas = SEGAcriteria.computeP4(p4values_test[0])
# p4_iwm = SEGAcriteria.computeP4(p4values_test[1])
# p4_sunshine = SEGAcriteria.computeP4(p4values_test[2])
# p4_test = SEGAcriteria.computeP4(p3values_test[3])
# print("p4_dice_tpvagens:", p4_tpvagenas)
# print("p4_dice_iwm:", p4_iwm)
# print("p4_dice_sunshine:", p4_sunshine)
# print("p4_dice_test:", p4_test)

sensitivity = SEGAsensitivity()
