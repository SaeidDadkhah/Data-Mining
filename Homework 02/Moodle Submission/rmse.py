from numpy import mean


def rmse(a, b):
    return mean((a - b) ** 2) ** 0.5
