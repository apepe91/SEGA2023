import math
from typing import List
from scipy.special import erfcinv
from scipy.stats import norm


class Stati:
    @staticmethod
    def skewness(values):
        skewNum = 0
        mean = Stati.mean(values)
        for value in values:
            skewNum += (value - mean) ** 3
        return skewNum / ((len(values) - 1) * Stati.sampleStd(values) ** 3)

    @staticmethod
    def mean(vector: List[float]) -> float:
        if len(vector) == 0:
            return None
        return sum(vector) / len(vector)

    @staticmethod
    def median(vector: List[float]) -> float:
        return Stati.quantile(vector, 0.5)

    @staticmethod
    def mode(vector):
        mode = math.inf
        maxCount = 0
        for i in range(len(vector)):
            count = 0
            for j in range(len(vector)):
                if vector[j] == vector[i]:
                    count += 1
            if count > maxCount:
                maxCount = count
                mode = vector[i]
        return mode

    @staticmethod
    def sampleVar(x: List[float]) -> float:
        if len(x) < 2:
            return None
        return Stati.sumNthPow(x, 2) / (len(x) - 1)

    @staticmethod
    def sumNthPow(x: List[float], n: int) -> float:
        meanValue = Stati.mean(x)
        totalSum = 0
        for i in range(len(x)):
            if n == 2:
                tempValue = x[i] - meanValue
                totalSum += tempValue * tempValue
            else:
                totalSum += math.pow(x[i] - meanValue, n)
        return totalSum

    @staticmethod
    def sampleStd(vector: List[float]) -> float:
        return math.sqrt(Stati.sampleVar(vector))

    @staticmethod
    def quantile(array: List[float], p: float) -> float:
        sortedArray = sorted(array)
        pos = (len(sortedArray) - 1) * p
        base = math.floor(pos)
        rest = pos - base
        if len(sortedArray) > base + 1:
            return sortedArray[base] + rest * (
                sortedArray[base + 1] - sortedArray[base]
            )
        else:
            return sortedArray[base]

    @staticmethod
    def eCDF(data: List[float], sort: bool = False) -> List[float]:
        sortedData = sorted(data)
        cdfValues = [0] * len(data)
        step = 1 / len(data)
        if sort:
            for i in range(len(sortedData)):
                cdfValues[i] = (i + 1) * step
        else:
            for i in range(len(data)):
                rank = sortedData.index(data[i])
                cdfValues[i] = (rank + 1) * step
        return cdfValues

    @staticmethod
    def prob_stdNormal(value: float, mean: float, sigma: float) -> float:
        if sigma < 0:
            raise ValueError("negative sigma.")
        elif sigma == 0:
            if value < mean:
                return 0
            else:
                return 1
        else:
            return Stati.cdf_stdNormal_pt((value - mean) / sigma)

    @staticmethod
    def invCdf_normal_pt(cdf_x: float, std_mu: float, std_sigma: float) -> float:
        if std_sigma <= 0:
            raise ValueError("Inverse normal CDF: variance is <= 0.")
        if cdf_x < 0 and cdf_x > 1:
            raise ValueError(
                "Inverse normal CDF: unfeasible F(x): smaller than 0 or larger than 1."
            )
        x0 = -math.sqrt(2) * erfcinv(2 * cdf_x)
        x = std_sigma * x0 + std_mu
        return x

    @staticmethod
    def pdf_normal(vector: List[float], mean: float, std: float) -> List[float]:
        if std <= 0:
            raise ValueError(
                "Normal probability density: variance must be larger than 0."
            )
        pdf_value = []
        for i in range(len(vector)):
            pdf_value.append(Stati.pdf_normal_pt(vector[i], mean, std))
        return pdf_value

    @staticmethod
    def pdf_normal_pt(x: float, mean: float, std: float) -> float:
        s2 = math.pow(std, 2.0)
        A = 1.0 / math.sqrt(s2 * 2.0 * math.pi)
        B = -1.0 / (2.0 * s2)
        return A * math.exp(B * math.pow(x - mean, 2.0))

    @staticmethod
    def pdf_gumbel(vector: List[float], location: float, scale: float) -> List[float]:
        if scale <= 0:
            return []
        pdf_value = []
        for i in range(len(vector)):
            pdf_value.append(Stati.pdf_gumbel_pt(vector[i], location, scale))
        return pdf_value

    @staticmethod
    def pdf_gumbel_pt(x: float, location: float, scale: float) -> float:
        z = (x - location) / scale
        return math.exp(z - math.exp(z)) / scale

    @staticmethod
    def cdf_uniform(
        vector: List[float], lowBound: float, uppBound: float
    ) -> List[float]:
        cdf_value = []
        for i in range(len(vector)):
            cdf_value.append(Stati.cdf_uniform_pt(vector[i], lowBound, uppBound))
        return cdf_value

    @staticmethod
    def cdf_uniform_pt(x: float, a: float, b: float) -> float:
        if math.isnan(x) or math.isnan(a) or math.isnan(b) or a >= b:
            return math.nan
        if x < a:
            return 0.0
        if x >= b:
            return 1.0
        return (x - a) / (b - a)

    @staticmethod
    def cdf_stdNormal_pt(X: float) -> float:
        T = 1 / (1 + 0.2316419 * abs(X))
        D = 0.3989423 * math.exp((-X * X) / 2)
        Prob = (
            D
            * T
            * (
                0.3193815
                + T * (-0.3565638 + T * (1.781478 + T * (-1.821256 + T * 1.330274)))
            )
        )
        if X > 0:
            Prob = 1 - Prob
        return Prob

    @staticmethod
    def cdf_normal(vector: List[float], mean: float, variance: float) -> List[float]:
        return [norm.cdf(x, mean, variance) for x in vector]

    @staticmethod
    def cdf_gumbel(vector: List[float], pdf_params: List[float]) -> List[float]:
        return [
            1 - Stati.cdf_gumbel_pt(x, pdf_params[0], pdf_params[1]) for x in vector
        ]

    @staticmethod
    def cdf_gumbel_pt(x: float, mu: float, sigma: float) -> float:
        z = (x - mu) / sigma
        return 1 - math.exp(-math.exp(z))
