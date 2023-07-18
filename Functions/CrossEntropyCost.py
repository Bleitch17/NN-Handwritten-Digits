import numpy as np


class CrossEntropyCost:

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y)*np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        # z parameter is kept to keep the interface consistent with other cost functions
        return a - y
