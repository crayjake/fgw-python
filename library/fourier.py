# -- library/fourier.py --
# Author: Jake Cray
# Github: crayjake/fgw-python
''' Fourier transform and inverse '''

from .integration import trapezium
from cmath import exp, pi

# global steps for transforms
steps = 2500
L = 10000 # forcing width

# cache of fourier transforms
cachedFourier = {}
def Fourier(func, k, m=50,s=2500):
    val = func(10)
    if func in cachedFourier:
        if k in cachedFourier[val]:
            return cachedFourier[val][k]
        
    
    xMax = m * L
    dx = xMax / s
    # e^ix   = cos(x) + i*sin(x)
    # e^-ikx = e^i(-kx) = cos(-kx) + i*sin(-kx) = cos(kx) - i*sin(kx)
    function = lambda x: exp(k * x * -1j) * func(x) 
    value = trapezium(function, (-xMax, xMax), dx)

    if func not in cachedFourier:
        cachedFourier[val] = {}
    cachedFourier[val][k] = value

    return value


def InverseFourier(func, x, m=10,s=2500):
    kMax = m / L
    dk = kMax / s

    function = lambda k: exp(k * x * 1j) * func(k)

    return trapezium(function, (-kMax, kMax), dk, True) / (2 * pi) 
