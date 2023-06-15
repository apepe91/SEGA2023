import math


class OutputPce:
    def __init__(self, input, output, u_matrix, pdf, pce_degree):
        self.input = input
        self.output = output
        self.u_matrix = u_matrix
        self.input_PDF = pdf
        self.degree = self.getPCEdegree(input)
        self.uniPoly = self.get_multivariate_orthogonal_poly(pdf, pce_degree)
        self.alphaIdx = self.generate_multi_index(self.degree, len(self.input_PDF))
        self.uniP_val = self.evaluated_multivariate_ortho_poly(
            self.uniPoly, self.u_matrix
        )
        self.Psi_alpha = self.compute_Psi_alpha_matrix(self.alphaIdx, self.uniP_val)
        self.coeffs = self.regression(self.output, self.Psi_alpha)
        self.cardAlpha = len(self.alphaIdx)

    def evaluated_multivariate_ortho_poly(self, multiPoly, u_matrix):
        multiVariate_matrix = []
        for i in range(len(u_matrix)):
            multiVariate_matrix.append(
                self.evaluate_univariate_ortho_poly(multiPoly[i], u_matrix[i])
            )

        return multiVariate_matrix

    def evaluate_univariate_ortho_poly(self, multiPoly, u_vector):
        univariate_matrix = []
        for ns in range(len(u_vector)):
            univariate_matrix.append(
                self.evaluate_ortho_poly_data_point(
                    multiPoly["Aterms"],
                    multiPoly["Bterms"],
                    multiPoly["degree"],
                    u_vector[ns],
                )
            )

        return univariate_matrix

    def evaluate_ortho_poly_data_point(self, Aterms, Bterms, degree, u_dataPt):
        orthoPoly_point_evaluation = [0] * (degree + 1)
        orthoPoly_point_evaluation[1] = Bterms[0]
        for d in range(degree):
            orthoPoly_point_evaluation[d + 2] = (
                (u_dataPt - Aterms[d])
                * (orthoPoly_point_evaluation[d + 1] / Bterms[d + 1])
            ) - ((orthoPoly_point_evaluation[d] * Bterms[d]) / Bterms[d + 1])

        return orthoPoly_point_evaluation[1:]

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

    def generate_multi_index(self, degree, M):
        result = []

        def generate_helper(curr, _sum, remaining):
            if remaining == 0:
                result.append(curr)
                return

            for i in range(_sum, -1, -1):
                next_index = curr + [i]
                generate_helper(next_index, _sum - i, remaining - 1)

        generate_helper([], degree, M)
        return result[::-1]

    def get_multivariate_orthogonal_poly(self, inputPDF, degree):
        univPoly = []

        for i in range(len(inputPDF)):
            if inputPDF[i]["name"] == "uniform":
                univPoly.append(self.univariate_legendre_poly(degree))
            elif inputPDF[i]["name"] == "normal":
                univPoly.append(self.univariate_hermite_poly(degree))
            else:
                raise ValueError(
                    "Orthogonal polynomials construction: solution not yet implemented."
                )
        return univPoly

    def univariate_legendre_poly(self, degree):
        a_terms = [0] * (degree + 1)
        b_terms = [0] * (degree + 1)

        for i in range(degree + 1):
            a_terms[i] = 0
            b_terms[i] = (1 / (4 - (i**-2))) ** 0.5

        b_terms[0] = 1

        legendre_poly = {
            "name": "legendre",
            "bounds": [-1, 1],
            "degree": degree,
            "Aterms": a_terms,
            "Bterms": b_terms,
        }

        return legendre_poly

    def univariate_hermite_poly(self, degree):
        a_terms = [0] * (degree + 1)
        b_terms = [0] * (degree + 1)

        for i in range(degree + 1):
            a_terms[i] = 0
            b_terms[i] = i**0.5

        b_terms[0] = 1

        hermite_poly = {
            "name": "hermite",
            "bounds": [-float("inf"), float("inf")],
            "degree": degree,
            "Aterms": a_terms,
            "Bterms": b_terms,
        }

        return hermite_poly

    def getPCEdegree(input):
        M = len(input)
        maxDegree = 10
        p = 1
        while p < maxDegree:
            if len(list(zip(*input))) < 3 * math.comb(M + p, p):
                if p - 1 == 0:
                    return 1
                else:
                    return p - 1
            p += 1

        return maxDegree
