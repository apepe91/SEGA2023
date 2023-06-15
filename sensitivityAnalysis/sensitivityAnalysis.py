from typing import List, Optional
from pce import Pce, OutputPce


def unzip(iterable):
    return list(zip(*iterable))


class iSobolIndices:
    def __init__(self, firstSobol, totalSobol):
        self.firstSobol = firstSobol
        self.totalSobol = totalSobol


class SAresults:
    def __init__(self, var_name, firstSobol, totalSobol):
        self.var_name = var_name
        self.firstSobol = firstSobol
        self.totalSobol = totalSobol


class SensitivityAnalysis:
    def __init__(self, input, output, pces):
        self.sensAnalysis = [OneSobolIndices(input, output, pce) for pce in pces.pce]


class OneSobolIndices:
    def __init__(self, input, output, pce: OutputPce):
        self.input = input
        self.output = output
        self.indices = [self.getSobol_indices(unzip(pce.alphaIdx), pce.coeffs)]

    def getSobol_indices(self, multiAlpha, PCE_coeff) -> iSobolIndices:
        firstSobol = self.get_firstSobol(multiAlpha, PCE_coeff)
        totalSobol = self.get_totalSobol(multiAlpha, PCE_coeff)

        SA_results = iSobolIndices(firstSobol=firstSobol, totalSobol=totalSobol)
        return SA_results

    def get_firstSobol(self, multiAlpha, PCE_coeff) -> List[float]:
        totVariance_system = self.get_totalVariance_fromPCE(PCE_coeff)
        sensitivity_indices = []

        for m in range(len(multiAlpha)):
            cumSum_PCEcoeff = []

            for cA in range(1, len(PCE_coeff)):
                if multiAlpha[m][cA] != 0:
                    otherRows = [
                        row[cA] for index, row in enumerate(multiAlpha) if index != m
                    ]
                    sum_otherRows = sum(otherRows)

                    if sum_otherRows == 0:
                        cumSum_PCEcoeff.append(PCE_coeff[cA] ** 2)

                    # if len(cumSum_PCEcoeff) == len(multiAlpha):
                    #     break

            sensitivity_indices.append(sum(cumSum_PCEcoeff))

        return [element / totVariance_system for element in sensitivity_indices]

    def get_totalSobol(self, multiAlpha, PCE_coeff) -> List[float]:
        totVariance_system = self.get_totalVariance_fromPCE(PCE_coeff)
        sensitivity_indices = []

        for m in range(len(multiAlpha)):
            cumSum_PCEcoeff = []

            for cA in range(1, len(PCE_coeff)):
                if multiAlpha[m][cA] != 0:
                    cumSum_PCEcoeff.append(PCE_coeff[cA] ** 2)

            sensitivity_indices.append(sum(cumSum_PCEcoeff))

        return [element / totVariance_system for element in sensitivity_indices]

    def get_totalVariance_fromPCE(self, PCE_coeff) -> float:
        PCE_coeff_without_first = PCE_coeff[1:]

        return sum(a**2 for a in PCE_coeff_without_first)
