import pandas as pd
import numpy as np
from userData import Input, OneInput, Output, OneOutput, UserData, Hidden
import copy
from typing import List


class UserData:
    def __init__(self):
        self.input = Input([])
        self.output = Output([])
        self.inputs = []
        self.outputs = []
        self.hid = Hidden([])


class DataService:
    emptyData = UserData()

    data = copy.deepcopy(emptyData)


samplesXLSX = pd.read_excel("sa/01_in_ishigami1000.xlsx")
modelEval = samplesXLSX.loc[:, "out1"].to_numpy()
samples = np.transpose(samplesXLSX.loc[:, ["x1", "x2", "x3"]].to_numpy())

inputs = [OneInput(row) for row in samples]

outputs = [OneOutput(modelEval)]
