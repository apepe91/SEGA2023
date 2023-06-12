import math


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
