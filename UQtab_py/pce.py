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
