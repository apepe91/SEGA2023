import math
from statisticalOp import Stati
from dataOp import Dati
from typing import List, Optional


class Inferred_pdf:
    def __init__(self):
        self.name = ""
        self.nickname = ""
        self.parameters = []
        self.bounds = []
        self.likelihood = 0
        self.AIC = 0
        self.BIC = 0
        self.KS = 0
        self.selectionResult = 0


class iStdPdf:
    def __init__(self):
        self.name = ""
        self.parameters = []
        self.bounds = Optional[List[float]]  # Optional bounds


class IsoProb_transform:
    def __init__(self, data, pdf):
        self.x = data
        self.pdf = pdf
        self.cdf = self.getCDF_vector(data, pdf)
        self.u = self.isoprobTransf_stdPDF_vector(self.cdf, pdf)

    def getCDF_vector(self, data, pdf):
        if pdf.name == "uniform":
            cdf_x = Stati.cdf_uniform(data, pdf.parameters[0], pdf.parameters[1])
            return cdf_x
        elif pdf.name == "normal":
            cdf_x = Stati.cdf_normal(data, pdf.parameters[0], pdf.parameters[1])
            return cdf_x
        else:
            raise ValueError("UQTAB Error - Iso Prob not yet implemented")

    def isoprobTransf_stdPDF_vector(self, cdf_x, pdf):
        if pdf.name == "uniform":
            cdf_u = []
            pdf_stdUniform = {
                "name": "uniform",
                "parameters": [-1, 1],
                "bounds": [-1, 1],
            }

            for i in range(len(cdf_x)):
                cdf_u.append(
                    pdf_stdUniform["parameters"][0]
                    + (
                        pdf_stdUniform["parameters"][1]
                        - pdf_stdUniform["parameters"][0]
                    )
                    * cdf_x[i]
                )

            return cdf_u
        elif pdf["name"] == "normal":
            cdf_u = []
            pdf_stdNormal = {"name": "normal", "parameters": [0, 1]}

            for i in range(len(cdf_x)):
                cdf_u.append(
                    Stati.invCdf_normal_pt(
                        cdf_x[i],
                        pdf_stdNormal["parameters"][0],
                        pdf_stdNormal["parameters"][1],
                    )
                )
            return cdf_u
        else:
            raise ValueError("UQTAB Error: Iso Prob not yet implemented")


class OnePdf:
    def __init__(self, data=None):
        inferredPDF = self.statInference(data)

        self.name = inferredPDF["name"]
        self.nickname = inferredPDF["nickname"]
        self.parameters = inferredPDF["parameters"]
        self.bounds = inferredPDF["bounds"]
        self.likelihood = inferredPDF["likelihood"]
        self.AIC = inferredPDF["AIC"]
        self.BIC = inferredPDF["BIC"]
        self.KS = inferredPDF["KS"]
        self.selectionResult = inferredPDF["selectionResult"]

    def statInference(self, data):
        selectionWeights = [0.3, 0.2, 0.5]
        tested_pdf = []

        tested_pdf.append(self.inferenceUniform(data, selectionWeights))
        tested_pdf.append(self.inferenceNormal(data, selectionWeights))
        tested_pdf.append(self.inferenceGumbel(data, selectionWeights))

        # Sort the tested PDF based on selection method (PATENT)
        tested_pdf.sort(key=lambda x: x["selectionResult"])

        return tested_pdf[0]

    def uniformPdf_likelihood(self, randomVector, pdfParams):
        return len(randomVector) * math.log(1 / (pdfParams[1] - pdfParams[0]))

    def uniformPdf_parameters(self, randomVector):
        return [
            min(randomVector)
            - (max(randomVector) - min(randomVector)) / len(randomVector),
            max(randomVector)
            + (max(randomVector) - min(randomVector)) / len(randomVector),
        ]

    def inferenceUniform(self, randomVector, weights) -> Inferred_pdf:
        uniformPdf_params = self.uniformPdf_parameters(randomVector)
        uniformLikelihood = self.uniformPdf_likelihood(randomVector, uniformPdf_params)

        aic_pdf = self.computeAIC(uniformPdf_params, uniformLikelihood)
        bic_pdf = self.computeBIC(randomVector, uniformPdf_params, uniformLikelihood)
        ks_stat = self.computeKStest(
            randomVector,
            sorted(
                Stati.cdf_uniform(
                    randomVector, uniformPdf_params[0], uniformPdf_params[1]
                )
            ),
        )

        return {
            "name": "uniform",
            "nickname": "U",
            "parameters": uniformPdf_params,
            "bounds": [min(randomVector), max(randomVector)],
            "likelihood": uniformLikelihood,
            "AIC": aic_pdf,
            "BIC": bic_pdf,
            "KS": ks_stat,
            "selectionResult": weights[0] * aic_pdf
            + weights[1] * bic_pdf
            + weights[2] * ks_stat,
        }

    def normalPdf_likelihood(self, randomVector):
        return sum(math.log(value) for value in self.normalPdf_values(randomVector))

    def normalPdf_values(self, randomVector):
        estimatedParams = self.normalPdf_estimatedParam(randomVector)
        return Stati.pdf_normal(randomVector, estimatedParams[0], estimatedParams[1])

    def normalPdf_estimatedParam(self, randomVector):
        return [Stati.mean(randomVector), Stati.sampleStd(randomVector)]

    def inferenceNormal(self, randomVector, weights) -> Inferred_pdf:
        normalPdf_EstimatedParameters = self.normalPdf_estimatedParam(randomVector)
        normalLikelihood = self.normalPdf_likelihood(randomVector)
        aic_pdf = self.computeAIC(normalPdf_EstimatedParameters, normalLikelihood)
        bic_pdf = self.computeBIC(
            randomVector, normalPdf_EstimatedParameters, normalLikelihood
        )
        ks_stat = self.computeKStest(
            randomVector,
            Dati.sortElements(
                Stati.cdf_normal(
                    randomVector,
                    normalPdf_EstimatedParameters[0],
                    normalPdf_EstimatedParameters[1],
                )
            ),
        )

        return {
            "name": "normal",
            "nickname": "N",
            "parameters": normalPdf_EstimatedParameters,
            "bounds": [min(randomVector), max(randomVector)],
            "likelihood": normalLikelihood,
            "AIC": aic_pdf,
            "BIC": bic_pdf,
            "KS": ks_stat,
            "selectionResult": weights[0] * aic_pdf
            + weights[1] * bic_pdf
            + weights[2] * ks_stat,
        }

    def inferenceGumbel(self, randVec, w):
        sampleMean = Stati.mean(randVec)
        sampleVar = Stati.sampleVar(randVec)

        initialBeta = math.sqrt(6 * sampleVar) / math.pi

        newBeta0 = initialBeta
        diffParam = 1
        newBeta = 0
        newMu = 0

        # MLE of MU and BETA
        while diffParam > 1e-6:
            numerator = sum(x * math.exp(-x / newBeta0) for x in randVec)
            denominator = sum(math.exp(-x / newBeta0) for x in randVec)
            newBeta = sampleMean - numerator / denominator

            newMu = newBeta * math.log(
                (1 / len(randVec)) * sum(math.exp(-x / newBeta) for x in randVec)
            )

            diffParam = abs(newBeta - newBeta0)
            newBeta0 = (newBeta + newBeta0) / 2

        pdf_params = [newMu, newBeta]
        gumbel_pdf_value = Stati.pdf_gumbel(randVec, pdf_params[0], pdf_params[1])

        if len(gumbel_pdf_value) == 0:
            gumbel_pdf = {
                "name": "gumbel - failed",
                "nickname": "ND",
                "parameters": pdf_params,
                "bounds": [min(randVec), max(randVec)],
                "likelihood": float("-inf"),
                "AIC": float("inf"),  # Akaike Informative Criterion
                "BIC": float("inf"),  # Bayesian Informative Criterion
                "KS": 0,  # Kolmogorov-Smirnov test
                "selectionResult": 0,
            }
            return gumbel_pdf

        likelihood = sum(math.log(x) for x in gumbel_pdf_value)

        aic_pdf = self.computeAIC(pdf_params, likelihood)

        bic_pdf = self.computeBIC(randVec, pdf_params, likelihood)

        ksstat = self.computeKStest(
            randVec, Dati.sortElements(Stati.cdf_gumbel(randVec, pdf_params))
        )

        selectionCrit = w[0] * aic_pdf + w[1] * bic_pdf + w[2] * ksstat

        gumbel_pdf = {
            "name": "gumbel",
            "nickname": "Gumb.",
            "parameters": pdf_params,
            "bounds": [min(randVec), max(randVec)],
            "likelihood": likelihood,
            "AIC": aic_pdf,  # Akaike Informative Criterion
            "BIC": bic_pdf,  # Bayesian Informative Criterion
            "KS": ksstat,  # Kolmogorov-Smirnov test
            "selectionResult": selectionCrit,
        }
        return gumbel_pdf

    def computeBIC(self, randomVector, pdfParameters, likelihood):
        return math.log(len(randomVector)) * len(pdfParameters) - 2 * likelihood

    def computeAIC(self, pdfParameters, likelihood):
        return 2 * len(pdfParameters) - 2 * likelihood

    def computeKStest(self, vector, cdfx_test):
        ecdf_x = [0] + Stati.eCDF(vector, True)

        # Compute the vertical differences for jumps approaching from the left
        delta1 = [ecdf - cdfx_test[i] for i, ecdf in enumerate(ecdf_x[:-1])]

        # Compute the vertical differences for jumps approaching from the right
        delta2 = [ecdf - cdfx_test[i] for i, ecdf in enumerate(ecdf_x[1:])]

        deltaCDF = list(map(abs, delta1 + delta2))
        KSstatistic = max(deltaCDF)

        return KSstatistic
