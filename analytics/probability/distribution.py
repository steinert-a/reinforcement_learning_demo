import numpy as np
import math


def poisson_distribution(alpha):
    """Poisson distribution
    Alpha is defining the expected value and variance of the distribution.
    """
    exp_lambda = np.exp(-alpha)
    factorial = np.vectorize(lambda n: math.gamma(n+1))
    return lambda n: exp_lambda * alpha**n / factorial(n)

