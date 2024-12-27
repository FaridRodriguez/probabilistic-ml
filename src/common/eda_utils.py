import numpy as np


def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = [i / n for i in range(1, n + 1)]
    return x, y


def pearson_corr_coef(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    corr_mat = np.corrcoef(x, y)
    return corr_mat[0, 1]