import pandas as pd
import numpy as np
from userData import Input, OneInput, Output, OneOutput, UserData, Hidden
from typing import List
from pce import Pce

samplesXLSX = pd.read_excel("sa/01_in_ishigami1000.xlsx")
modelEval = samplesXLSX.loc[:, "out1"].to_numpy()
samples = np.transpose(samplesXLSX.loc[:, ["x1", "x2", "x3"]].to_numpy())

inputs = [OneInput(row) for row in samples]
inputPDF = []
inputPDF.extend([input.pdf for input in inputs])
u_matrix = []
u_matrix.extend([input.u_vector.u for input in inputs])
outputs = [OneOutput(modelEval)]

hidden = [Hidden(samples)]


pce = Pce(samples, modelEval, u_matrix, inputPDF, hidden[0].pceDegree)
