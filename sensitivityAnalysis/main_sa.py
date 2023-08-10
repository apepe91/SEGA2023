import pandas as pd
import numpy as np
from userData import OneInput, Hidden
from pce import Pce
from sensitivityAnalysis import SensitivityAnalysis
from statisticalOp import Stati
import matplotlib.pyplot as plt
import json
import scipy.stats as ss


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
        var = Stati.sampleVar(values)
        std = Stati.sampleStd(values)
        skewNum = 0
        for value in values:
            skewNum += (value - mean) ** 3
        skew = skewNum / ((len(values) - 1) * std**3)

        return weights[0] / mode_gm + weights[1] / var + weights[2] / abs(skew)

    def computeP4(values):
        weights = [0.6, 0.25, 0.15]
        mode_gm = Stati.mode(values)
        mean = Stati.mean(values)
        var = Stati.sampleVar(values)
        std = Stati.sampleStd(values)
        skewNum = 0
        for value in values:
            skewNum += (value - mean) ** 3
        skew = skewNum / ((len(values) - 1) * std**3)

        return weights[0] * mode_gm + weights[1] / var + weights[2] / abs(skew)

    def computePJ(values, invalid_elements):
        weights = [0.3, 0.25, 0.15, 0.3]
        mode = Stati.mode(values)
        mean = Stati.mean(values)
        var = Stati.sampleVar(values)
        std = Stati.sampleStd(values)
        skewNum = 0
        for value in values:
            skewNum += (value - mean) ** 3
        skew = skewNum / ((len(values) - 1) * std**3)

        return (
            weights[0] * mode
            + weights[1] / var
            + weights[2] / abs(skew)
            + weights[3] / Stati.mean(invalid_elements)
        )

    def printFinalrankJ(rank_fin, names, rank_mode, rank_var, rank_skew, rank_invElem):
        sortedRank = np.argsort(rank_fin)
        for elem in sortedRank:
            print(
                names[elem],
                " - Final rank:",
                rank_fin[elem].round(3) + 1,
                " - Rank mode:",
                rank_mode[elem] + 1,
                " - Rank variance:",
                rank_var[elem] + 1,
                " - Rank skewness:",
                rank_skew[elem] + 1,
                " - Rank inv. elem.:",
                rank_invElem[elem] + 1,
            )

    def printFinalRankDSC_HD(rank_fin, names, rank_mode, rank_var, rank_skew):
        sortedRank = np.argsort(rank_fin)
        print("-" * 79)
        print(
            "| {:^15s} | {:^12s} | {:^12s} | {:^12s} | {:^12s} |".format(
                "user", "fin. rank", "rank mode", "rank var", "rank skew"
            )
        )
        print("-" * 79)
        for userNr in sortedRank:
            print(
                "| {:<15s} | {:^12.3f} | {:^12.1f} | {:^12.1f} | {:^12.1f} |".format(
                    names[userNr],
                    rank_fin[userNr],
                    rank_mode[userNr],
                    rank_var[userNr],
                    rank_skew[userNr],
                )
            )

        print("-" * 79)

    def printFinalRank_HD(rank_fin, rankings):
        sortedRank = np.argsort(rank_fin) + 1
        print("-" * 79)
        print(
            "| {:^15s} | {:^12s} | {:^12s} | {:^12s} | {:^12s} |".format(
                "user", "fin. rank", "rank mode", "rank var", "rank skew"
            )
        )
        print("-" * 79)
        for userNr in sortedRank:
            print(
                "| {:<15s} | {:^12.3f} | {:^12.1f} | {:^12.1f} | {:^12.1f} |".format(
                    rankings[0][userNr - 1],
                    rank_fin[userNr - 1],
                    rankings[1][userNr - 1],
                    rankings[2][userNr - 1],
                    rankings[3][userNr - 1],
                    # rank_mode[userNr],
                    # rank_var[userNr],
                    # rank_skew[userNr],
                )
            )

        print("-" * 79)

    def rankPJ(values, invElem, names):
        weights = [0.3, 0.25, 0.15, 0.3]
        mode_vec = []
        var_vec = []
        skew_vec = []
        invElem_vec = []
        names_vec = []
        for num, name in enumerate(names):
            if Stati.sampleVar(values[:, num]) == 0:
                continue
            if Stati.mode(values[:, num]) < 0:
                mode_vec.append(Stati.mode(values[:, num]) * (-1))
            else:
                mode_vec.append(Stati.mode(values[:, num]))

            var_vec.append(Stati.sampleVar(values[:, num]))
            skew_vec.append(Stati.skewness(values[:, num]))
            invElem_vec.append(Stati.mean(invElem[:, num]))
            names_vec.append(name)
        skew_vec = [abs(elem) for elem in skew_vec]

        rank_mode = np.argsort(mode_vec)
        rank_var = np.argsort(var_vec)
        rank_skew = np.argsort(skew_vec)
        rank_invElem = np.argsort(invElem_vec)

        rank_fin = []
        for num, _ in enumerate(rank_mode):
            rank_fin.append(
                rank_mode[num] * weights[0]
                + rank_var[num] * weights[1]
                + rank_skew[num] * weights[2]
                + rank_invElem[num] * weights[3]
            )

        SEGAcriteria.printFinalrankJ(
            rank_fin, names_vec, rank_mode, rank_var, rank_skew, rank_invElem
        )

        return rank_fin

    def rankDSC(values, names):
        weights = [0.5, 0.3, 0.2]
        mode_vec = []
        var_vec = []
        skew_vec = []

        names_vec = []
        for num, name in enumerate(names):
            mode_vec.append(Stati.median(values[:, num]))
            var_vec.append(Stati.sampleVar(values[:, num]))
            skew_vec.append(Stati.skewness(values[:, num]))
            names_vec.append(name)
        skew_vec = [abs(elem) for elem in skew_vec]

        rank_mode = np.argsort(mode_vec)[::-1]
        rank_var = np.argsort(var_vec)
        rank_skew = np.argsort(skew_vec)

        rank_fin = []
        for num, _ in enumerate(rank_mode):
            rank_fin.append(
                rank_mode[num] * weights[0]
                + rank_var[num] * weights[1]
                + rank_skew[num] * weights[2]
            )

        SEGAcriteria.printFinalRankDSC_HD(
            rank_fin, names_vec, rank_mode, rank_var, rank_skew
        )

        return rank_fin

    def rankHD(values, names):
        weights = [0.5, 0.3, 0.2]
        # mode_vec = np.array([])
        mode_vec = []
        var_vec = []
        skew_vec = []

        names_vec = []
        for num, name in enumerate(names):
            # mode_vec = np.append(mode_vec, np.array([Stati.median(values[:, num])]))
            mode_vec.append(Stati.median(values[:, num]))
            var_vec.append(Stati.sampleVar(values[:, num]))
            skew_vec.append(Stati.skewness(values[:, num]))
            names_vec.append(name)
        skew_vec = [abs(elem) for elem in skew_vec]

        rank_mode = np.argsort(mode_vec) + 1
        rank_var = np.argsort(var_vec) + 1
        rank_skew = np.argsort(skew_vec) + 1

        rank_fin = []
        for num, _ in enumerate(rank_mode):
            rank_fin.append(
                rank_mode[num] * weights[0]
                + rank_var[num] * weights[1]
                + rank_skew[num] * weights[2]
            )

        SEGAcriteria.printFinalRank_HD(
            rank_fin, [names_vec, rank_mode, rank_var, rank_skew]
        )

        return rank_fin


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

phase2_invElem = phase2_results[0:5]
phase2_jacobians = phase2_results[5:10]
phase2_dice = phase2_results[10:15]
phase2_hausdorf = phase2_results[15:20]

pJ_phase2 = SEGAcriteria.rankPJ(
    phase2_jacobians.to_numpy(), phase2_invElem.to_numpy(), user_names
)
print("--> HD")
pHD_phase2 = SEGAcriteria.rankHD(phase2_hausdorf.to_numpy(), user_names)
print("--> DSC")
pDSC_phase2 = SEGAcriteria.rankDSC(phase2_dice.to_numpy(), user_names)

jaco_list = computeStats(phase2_jacobians)
dsc_list = computeStats(phase2_dice)
hd_list = computeStats(phase2_hausdorf)

with open("test_fun/jacobian_list.json", "w") as fp:
    json.dump(jaco_list, fp)
with open("test_fun/dsc_list.json", "w") as fp:
    json.dump(dsc_list, fp)
with open("test_fun/hd_list.json", "w") as fp:
    json.dump(hd_list, fp)


p3_list = computeMetricHD(phase2_hausdorf)
p4_list = computeMetricDSC(phase2_dice)

with open("test_fun/p3_HD_list.json", "w") as fp:
    json.dump(p3_list, fp)
with open("test_fun/p4_DSC_list.json", "w") as fp:
    json.dump(p4_list, fp)


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
