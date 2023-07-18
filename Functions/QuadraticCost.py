import numpy as np
from Functions import Sigmoid


class QuadraticCost:
    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        return (a - y) * Sigmoid.sigmoid_prime(z)
