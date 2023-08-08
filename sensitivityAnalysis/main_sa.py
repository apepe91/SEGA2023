import pandas as pd
import numpy as np
from userData import OneInput, Hidden
from pce import Pce
from sensitivityAnalysis import SensitivityAnalysis
from statisticalOp import Stati
import matplotlib.pyplot as plt
import json


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
        weights = [0.8, 0.15, 0.05]
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


# sensitivity = SEGAsensitivity()

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


def statsDict(input_vector):
    stats_dict = dict()
    stats_dict = {
        "mean": Stati.mean(input_vector),
        "median": Stati.median(input_vector),
        "mode": Stati.mode(input_vector),
        "std": Stati.sampleStd(input_vector),
        "var": Stati.sampleVar(input_vector),
        "skew": Stati.skewness(input_vector),
    }

    return stats_dict


def statsDict_p4(input_vector):
    stats_dict = dict()
    stats_dict = {
        "p4": SEGAcriteria.computeP4(input_vector),
    }

    return stats_dict


def statsDict_p3(input_vector):
    stats_dict = dict()
    stats_dict = {
        "p3": SEGAcriteria.computeP3(input_vector),
    }

    return stats_dict


def computeStats(metric):
    statsList = []
    for user in metric:
        results_user = metric[user][0:5]
        if Stati.sampleStd(results_user.to_numpy()) == 0:
            continue
        else:
            phase2_resultsStats = {user: statsDict(results_user.to_numpy())}
        statsList.append(phase2_resultsStats)
    return statsList


def computeMetricDSC(metric):
    statsList = []
    for user in metric:
        results_user = metric[user][0:5]
        phase2_resultsStats = {user: statsDict_p4(results_user.to_numpy())}
        statsList.append(phase2_resultsStats)
    return statsList


def computeMetricHD(metric):
    statsList = []
    for user in metric:
        results_user = metric[user][0:5]
        phase2_resultsStats = {user: statsDict_p3(results_user.to_numpy())}
        statsList.append(phase2_resultsStats)
    return statsList


phase2_results = pd.read_excel("test_fun/phase2_results.xlsx", dtype=np.float64)
user_names = list(
    pd.read_excel("test_fun/phase2_results.xlsx", nrows=1, engine="openpyxl")
)

phase2_jacobians = phase2_results[0:5]
phase2_dice = phase2_results[5:10]
phase2_hausdorf = phase2_results[10:15]

jaco_list = computeStats(phase2_jacobians)
dsc_list = computeStats(phase2_dice)
hd_list = computeStats(phase2_hausdorf)

with open("jacobian_list.json", "w") as fp:
    json.dump(jaco_list, fp)
with open("dsc_list.json", "w") as fp:
    json.dump(dsc_list, fp)
with open("hd_list.json", "w") as fp:
    json.dump(hd_list, fp)


p3_list = computeMetricHD(phase2_hausdorf)
p4_list = computeMetricDSC(phase2_dice)

with open("p3_HD_list.json", "w") as fp:
    json.dump(p3_list, fp)
with open("p4_DSC_list.json", "w") as fp:
    json.dump(p4_list, fp)
