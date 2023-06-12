import math
import pandas as pd
import numpy as np
import chaospy as cp

samplesXLSX = pd.read_excel("sa/01_in_ishigami1000.xlsx")
modelEval = samplesXLSX.loc[:, "out1"].to_numpy()
samples = np.transpose(samplesXLSX.loc[:, ["x1", "x2", "x3"]].to_numpy())


class PyCl:
    def __init__(self):
        a = 2
