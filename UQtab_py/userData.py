from pdf_model import IsoProb_transform, OnePdf
from pce import getPCEdegree


def convertStringToNumber(value):
    try:
        if isinstance(value, int) or isinstance(value, float):
            return value
        else:
            return float(value.replace(",", ""))
    except:
        return None


def unzip(iterable):
    return list(zip(*iterable))


class Hidden:
    def __init__(self, inputData):
        self.pceDegree = getPCEdegree(inputData)


class UserData:
    def __init__(self):
        self.input = Input()
        self.inputs = []
        self.output = Output()
        self.outputs = []
        self.hid = Hidden()


class Input:
    def __init__(self, raw):
        self.inputPDF = []
        self.u_matrix = []
        # raw_noHead = [col[1:] for col in raw]
        self.data = raw
        # self.names = [col[0] for col in raw]
        self.M = len(raw)
        self.Ns = len(list(zip(*self.data)))


class OneInput:
    def __init__(self, raw):
        # self.raw = raw
        self.data = raw
        # self.name = raw[0]
        self.Ns = len(self.data)
        self.pdf = OnePdf(self.data)
        # Isoprobabilistic transform
        self.u_vector = IsoProb_transform(self.data, self.pdf)


class Output:
    def __init__(self, raw):
        dataNoHeaders = [col[1:] for col in raw]
        self.data = [[convertStringToNumber(x) for x in col] for col in dataNoHeaders]
        self.names = [col[0] for col in raw]
        self.Nout = len(raw)
        self.Ns = len(unzip(self.data))


class OneOutput:
    def __init__(self, raw):
        raw_noHead = raw[1:]
        self.data = [convertStringToNumber(x) for x in raw_noHead]
        self.name = raw[0]
        self.Ns = len(self.data)
        self.pdf = OnePdf(self.data)
