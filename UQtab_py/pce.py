import math
from pce_predict import PcePredict
from regression import regression


def getPCEdegree(input):
    M = len(input)
    maxDegree = 10
    p = 1
    for p in range(1, maxDegree):
        if len(list(zip(*input))) < 3 * math.comb(M + p, p):
            if p - 1 == 0:
                return 1
            else:
                return p - 1
    return maxDegree


class iOrthogonalPolynomial:
    def __init__(self, name, bounds, degree, Aterms, Bterms):
        self.name = name
        self.bounds = bounds
        self.degree = degree
        self.Aterms = Aterms
        self.Bterms = Bterms


class Pce:
    def __init__(self, input_data, output, u_matrix, pdf, pce_degree):
        self.pce = [
            OutputPce(input_data, out.data, u_matrix, pdf, pce_degree) for out in output
        ]
        self.predictPce = [PcePredict(self.pce) for _ in output]


class OutputPce:
    def __init__(self, input_data, output, u_matrix, pdf, pce_degree):
        self.input = input_data
        self.output = output
        self.u_matrix = u_matrix
        self.input_PDF = pdf
        self.degree = getPCEdegree(input_data)
        self.uniPoly = self.get_multivariate_orthogonalPoly(pdf, pce_degree)
        self.alphaIdx = self.generateMultiIndex(self.degree, len(self.input_PDF))
        self.uniP_val = self.evaluated_multivariate_orthoPoly(
            self.uniPoly, self.u_matrix
        )
        self.Psi_alpha = self.compute_Psi_alpha_matrix(self.alphaIdx, self.uniP_val)
        self.coeffs = regression(self.output, self.Psi_alpha)
        self.cardAlpha = len(self.alphaIdx)

    def evaluated_multivariate_orthoPoly(self, multiPoly, u_matrix):
        multiVariate_matrix = []
        for i in range(len(u_matrix)):
            multiVariate_matrix.append(
                self.evaluate_univariate_orthoPoly(multiPoly[i], u_matrix[i])
            )

        return multiVariate_matrix

    def evaluate_univariate_orthoPoly(self, multiPoly, u_vector):
        univariate_matrix = []  # Ns x P
        for ns in range(len(u_vector)):
            univariate_matrix.append(
                self.evaluate_orthoPoly_dataPoint(
                    multiPoly["Aterms"],
                    multiPoly["Bterms"],
                    multiPoly["degree"],
                    u_vector[ns],
                )
            )
        return univariate_matrix

    def evaluate_orthoPoly_dataPoint(self, Aterms, Bterms, degree, u_dataPt):
        orthoPoly_pointEvaluation = []
        orthoPoly_pointEvaluation.append(0)
        orthoPoly_pointEvaluation.append(Bterms[0])

        for d in range(degree):
            orthoPoly_pointEvaluation.append(
                (u_dataPt - Aterms[d])
                * (orthoPoly_pointEvaluation[d + 1] / Bterms[d + 1])
                - (orthoPoly_pointEvaluation[d] * Bterms[d]) / Bterms[d + 1]
            )

        return orthoPoly_pointEvaluation.pop(0)

    def compute_Psi_alpha_matrix(self, alpha, Mo):
        Ns = len(Mo[0])
        M = len(Mo)
        cardA = len(alpha)
        Psi_alpha = []

        for ns in range(Ns):
            Psi_alpha.append([1] * cardA)
            for cA in range(cardA):
                for m in range(M):
                    deg = alpha[cA][m]
                    if deg != 0:
                        Psi_alpha[ns][cA] *= Mo[m][ns][deg]

        return Psi_alpha

    def generateMultiIndex(self, degree, M):
        result = []

        def generateHelper(curr, totalSum, remaining):
            if remaining == 0:
                result.append(curr)
                return

            for i in range(totalSum, -1, -1):
                next_ = curr + [i]
                generateHelper(next_, totalSum - i, remaining - 1)

        generateHelper([], degree, M)
        return result[::-1]

    def get_multivariate_orthogonalPoly(self, inputPDF, degree):
        univPoly = []

        for i in range(len(inputPDF)):
            if inputPDF[i].name == "uniform":
                univPoly.append(self.univariate_legendrePoly(degree))
            elif inputPDF[i].name == "normal":
                univPoly.append(self.univariate_hermitePoly(degree))
            else:
                raise ValueError(
                    "Orthogonal polynomials construction: solution not yet implemented."
                )

        return univPoly

    def univariate_legendrePoly(self, degree):
        a_terms = []
        b_terms = []

        for i in range(degree + 1):
            a_terms.append(0)
            if i == 0:
                b_terms.append(1)
            else:
                b_terms.append((1 / (4 - i ** (-2))) ** 0.5)

        # b_terms[0] = 1

        legendre_poly = {
            "name": "legendre",
            "bounds": [-1, 1],
            "degree": degree,
            "Aterms": a_terms,
            "Bterms": b_terms,
        }

        return legendre_poly

    def univariate_hermitePoly(self, degree):
        a_terms = []
        b_terms = []

        for i in range(degree + 1):
            a_terms.append(0)
            b_terms.append(i**0.5)

        b_terms[0] = 1

        hermite_poly = {
            "name": "hermite",
            "bounds": [-float("inf"), float("inf")],
            "degree": degree,
            "Aterms": a_terms,
            "Bterms": b_terms,
        }

        return hermite_poly
