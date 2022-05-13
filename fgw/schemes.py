# -- fgw/schemes.py --
# Author: Jake Cray
# GitHub: crayjake/fgw-python
''' Crank-Nicolson schemes '''


from .structures import DataStruct


def CrankNicolson( dt: float, data: DataStruct ):
    data.t += dt

    return data