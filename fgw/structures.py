# -- fgw/structures.py --
# Author: Jake Cray
# GitHub: crayjake/fgw-python
''' Structures of the variables to simulate '''

import numpy as np

class DataStruct:
    pass


class DefaultData(DataStruct):
    def GenerateInitialData(self, shape):
        print(f'generating initial data')
        empty = np.zeros(shape)
        self.u = np.copy(empty)
        self.v = np.copy(empty)
        self.w = np.copy(empty)
        self.p = np.copy(empty)
        self.b = np.copy(empty)
        self.rho = np.copy(empty)
        self.t = 0
