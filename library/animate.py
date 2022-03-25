# -- library/animate.py --
# Author: Jake Cray
# GitHub: crayjake/fgw-python
''' class for animating/displaying the data '''

import time

import numpy as np

from typing import Callable
from matplotlib import colors
from matplotlib import pyplot as plt

plt.rcParams['pcolor.shading'] = 'nearest'
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 200

from helpers import middleX, middleZ
from structures import State, Meta

def display(data: State,
            meta: Meta,
            converter: Callable[[np.ndarray, Meta], np.ndarray],
            prefix='',
            maxValue = 0.3,
            sponge=False):

    # convert data using provided converter
    # this turns our 1D data into 2D by adding z-dep
    inp = converter(data, meta)
    fig, ax = plt.subplots()

    # find max value and set up colour bar
    #maxValue = max(np.max(inp.b), -np.min(inp.b))
    #maxValue = 8e-8
    divnorm = colors.TwoSlopeNorm(vmin=-maxValue, vcenter=0, vmax=maxValue)

    # set up colour map to have 31 discrete values
    cmap = plt.get_cmap('bwr', 31)
    cmap = 'bwr'
    # colour plot
    c = ax.pcolor(middleX(meta.x, sponge),
                  middleX(meta.z, sponge),
                  middleX(inp.b, sponge) * (273 / 10),
                  cmap=cmap,
                  zorder=0,
                  norm=divnorm)
    
    # streamplot
    '''sp = ax.streamplot(
        middleZ(meta.X, sponge),
        meta.Z,
        middleX(inp.u, sponge),
        middleX(inp.w, sponge),
        color='k',
        arrowsize=1,
        density=0.5,
        linewidth=0.5,
        zorder=1)
        '''

    # plot alpha as a green dotted line (plotted as a percentage of the total depth)
    ax.plot(
        meta.x[0, :],
        meta.D * meta.spongeAlphaVectorized(meta.x[0, :]) / meta.spongeStrength,
        'g:')


    # add colour bar and format plot
    fig.colorbar(c, ax=ax)
    ax.get_xaxis().set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, p: format(int(x / 1000), ',')))
    ax.get_yaxis().set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, p: format(int(x / 1000), ',')))
    timeString = time.strftime('%H:%M:%S', time.gmtime(inp.t))
    plt.title(f't = {timeString}')

    # show plot
    plt.show()
