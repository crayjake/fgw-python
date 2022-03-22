# -- library/helpers.py --
# Author: Jake Cray
# GitHub: crayjake/fgw-python
''' contains helper functions such as the Heaviside function '''

import numpy as np

def H(time: int):
    return 1 if time > 0 else 0

def F(x, L):
    return (1 / (np.cosh(x / (L * 1000)))) ** 2

def G(x, L):
    return 1000 * L * np.tanh(x / (L * 1000))