from matrixOp import Matri
from copy import deepcopy
from userData import OnePdf


class PcePredict:
    def __init__(self, pce):
        self.out_pred = Matri.zeros1D(len(pce[0].input[0]))
        Psi0 = Matri.ones1D(len(pce[0].input[0]))

        for ncoeff in range(len(pce[0].coeffs)):
            Psi = deepcopy(Psi0)

            for m in range(len(pce[0].input)):
                if pce[0].alphaIdx[ncoeff][m] != 0:
                    ind_m = pce[0].alphaIdx[ncoeff][m]
                    Psi = [
                        x * pce[0].uniP_val[m][idx][ind_m] for idx, x in enumerate(Psi)
                    ]

            self.out_pred = [
                x + Psi[idx] * pce[0].coeffs[ncoeff]
                for idx, x in enumerate(self.out_pred)
            ]

        self.out_pdf = OnePdf(self.out_pred)
