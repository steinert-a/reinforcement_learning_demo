import numpy as np
import math

def softmax(x):
    x = np.array(x, dtype=float)
    if x.size == 0:
        return []
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def poisson_distribution(alpha):
    """Poisson distribution
    Alpha is defining the expected value and variance of the distribution.
    """
    exp_lambda = np.exp(-alpha)
    factorial = np.vectorize(lambda n: math.gamma(n+1))
    return lambda n: exp_lambda * alpha**n / factorial(n)

