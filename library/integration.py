# -- library/integration.py --
# Author: Jake Cray
# Github: crayjake/fgw-python
''' integration approximation '''


from math import floor
from tqdm import tqdm


def trapezium(func, bounds, delta, display = False):
    result = delta * (func(bounds[0]) + func(bounds[1])) / 2

    steps = (bounds[1] - bounds[0]) / delta

    if display:
        print(f'trapezium: {bounds}, {delta}')

        for i in tqdm(range(1, floor(steps))):
            result += delta * func(bounds[0] + (delta * i))

    else:
        for i in range(1, floor(steps)):
            result += delta * func(bounds[0] + (delta * i))

    return result
    

def simpson(func, bounds, delta):
    result = delta * (func(bounds[0]) + func(bounds[1])) / 3

    steps = (bounds[1] - bounds[0]) / delta
    for i in range(1, floor(steps)):
        add = 2 * delta * func(bounds[0] + (delta * i))

        if i % 2 != 0:
            add *= 2

        result += add / 3