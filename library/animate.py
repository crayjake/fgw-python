# -- library/animate.py --
# Author: Jake Cray
# GitHub: crayjake/fgw-python
''' class for animating/displaying the data '''

import time

import numpy as np

from typing import Callable
from matplotlib import colors
from matplotlib import pyplot as plt
plt.rcParams['pcolor.shading'] ='nearest'
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

from helpers import middleX, middleZ
from structures import State, Meta

'''
TODO:
     - Display a single timestep
       -> 1D vs 2D (requires a converter)
     - Generate and save a whole time series
     - Add option to overlay multiple graphs, i.e a streamplot and a colourplot
       -> then you could get the stream + heat plot or just a colourplot of a different variable
     - Show or Hide the sponge layer
'''

def display(data: State, meta: Meta, converter: Callable[[np.ndarray, Meta], np.ndarray], prefix='', sponge=True):
        skip = 1#math.ceil(data.meta.x.shape[1]/200)

        inp = converter(data, meta)
        fig, ax = plt.subplots()

        maxValue = max(np.max(inp.b), -np.min(inp.b))
        #maxValue = 0.3
        divnorm = colors.TwoSlopeNorm(vmin=-maxValue, vcenter=0, vmax=maxValue)


        cmap = plt.get_cmap('bwr', 31)
        cmap = 'bwr'
        c = ax.pcolor(middleX(meta.x, sponge), middleX(meta.z, sponge), middleX(inp.b, sponge) * (273 / 10), cmap=cmap, zorder=0, norm=divnorm)
 
        sp = ax.streamplot(middleZ(meta.X, sponge), meta.Z, middleX(inp.u, sponge), middleX(inp.w, sponge), color='k', arrowsize=1, density=0.5, linewidth=0.5, zorder=1)#, linewidth=lw)#,    density=0.8) # color=lw, cmap='Greys')

        print(f'MetaAlpha from display: {np.max(meta.spongeAlphaVectorized(meta.x[0,:]))}')
        ax.plot(meta.x[0,:], 50000 * meta.spongeAlphaVectorized(meta.x[0,:]) / meta.spongeStrength, 'g:')

        fig.colorbar(c, ax=ax)

        ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x / 1000), ',')))
        ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x / 1000), ',')))
        timeString = time.strftime('%H:%M:%S', time.gmtime(inp.t))
        plt.title(f't = {timeString}')
        plt.show()