from pdf_model import IsoProb_transform, OnePdf


def convertStringToNumber(value):
    try:
        if isinstance(value, int) or isinstance(value, float):
            return value
        else:
            return float(value.replace(",", ""))
    except:
        return None


class Input:
    def __init__(self, raw):
        self.inputPDF = []
        self.u_matrix = []
        raw_noHead = [col[1:] for col in raw]
        self.data = [[convertStringToNumber(x) for x in col] for col in raw_noHead]
        self.names = [col[0] for col in raw]
        self.M = len(raw)
        self.Ns = len(list(zip(*self.data)))


class OneInput:
    def __init__(self, raw):
        self.raw = raw
        raw_noHead = raw[1:]
        self.data = list(map(lambda x: convertStringToNumber(x), raw_noHead))
        self.name = raw[0]
        self.Ns = len(self.data)
        self.pdf = OnePdf(self.data)
        # Isoprobabilistic transform
        self.u_vector = IsoProb_transform(self.data, self.pdf)
