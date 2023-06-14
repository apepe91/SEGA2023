from scipy.linalg import svd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def regression(y, A):
    results = sm.OLS(y, A).fit()

    # u, s, vh = svd(A, full_matrices=True)
    # q = np.dot(u.T, y)
    # b_new = np.dot(np.dot(vh.T, np.diag(1 / s)), q)
    return results.params
