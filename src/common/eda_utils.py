import numpy as np


def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = [i / n for i in range(1, n + 1)]
    return x, y