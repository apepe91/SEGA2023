from scipy.linalg import svd
import numpy as np


def regression(y, A):
    u, s, vh = svd(A, full_matrices=False)
    q = np.dot(u.T, y)
    b_new = np.dot(np.dot(vh.T, np.diag(1 / s)), q)
    return b_new.flatten()
