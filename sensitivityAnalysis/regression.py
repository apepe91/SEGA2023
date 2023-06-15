from scipy.linalg import svd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def regression(y, A):
    results = sm.OLS(y, A).fit()

    return results.params
