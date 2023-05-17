class Output:
    def __init__(self, raw):
        dataNoHeaders = [col[1:] for col in raw]
        self.data = [[convertStringToNumber(x) for x in col] for col in dataNoHeaders]
        self.names = [col[0] for col in raw]
        self.Nout = len(raw)
        self.Ns = len(list(zip(*self.data)))
        self.outputPDF = []


class OneOutput:
    def __init__(self, raw):
        raw_noHead = raw[1:]
        self.data = [convertStringToNumber(x) for x in raw_noHead]
        self.name = raw[0]
        self.Ns = len(self.data)
        self.pdf = OnePdf(self.data)


class Input:
    def __init__(self, raw):
        raw_noHead = [col[1:] for col in raw]
        self.data = [[convertStringToNumber(x) for x in col] for col in raw_noHead]
        self.names = [col[0] for col in raw]
        self.M = len(raw)
        self.Ns = len(list(zip(*self.data)))
        self.inputPDF = []
        self.u_matrix = []


class OneInput:
    def __init__(self, raw):
        raw_noHead = raw[1:]
        self.data = [convertStringToNumber(x) for x in raw_noHead]
        self.name = raw[0]
        self.Ns = len(self.data)
        self.pdf = OnePdf(self.data)
        self.u_vector = IsoProb_transform(self.data, self.pdf)
