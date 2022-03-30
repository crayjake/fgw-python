# -- library/animate.py --
# Author: Jake Cray
# GitHub: crayjake/fgw-python
''' class for animating/displaying the data '''

import os
import string
import time

import numpy as np

from typing import Callable
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# plt.rcParams['pcolor.shading'] = 'nearest'
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 250

plt.style.use('seaborn-pastel')

import tqdm


from helpers import middleX, middleZ
from structures import State, Meta

def display(data:            State,
            meta:            Meta,
            converter:       Callable[[np.ndarray, Meta], np.ndarray],
            prefix:          string  = '',
            maxValue:        float   = 0.3,
            showSpongeLayer: bool    = False,
            showStreamPlot:  bool    = False,
            cmapDivisions:   int     = 31):

    # convert data using provided converter
    # this turns our 1D data into 2D by adding z-dep
    inp = converter(data, meta)
    fig, ax = plt.subplots()

    # find max value and set up colour bar
    #maxValue = max(np.max(inp.b), -np.min(inp.b))
    #maxValue = 8e-8
    divnorm = colors.TwoSlopeNorm(vmin=-maxValue, vcenter=0, vmax=maxValue)

    # set up colour map to have 31 discrete values
    cmap = plt.get_cmap('bwr', cmapDivisions)
    #cmap = 'bwr'

    # colour plot
    c = ax.pcolor(middleX(meta.x, showSpongeLayer),
                  middleX(meta.z, showSpongeLayer),
                  middleX(inp.b,  showSpongeLayer) * (273 / 10),
                  cmap=cmap,
                  zorder=0,
                  norm=divnorm)
    
    # streamplot
    if showStreamPlot:
        sp = ax.streamplot(
            middleZ(meta.X, showSpongeLayer),
            meta.Z,
            middleX(inp.u,  showSpongeLayer),
            middleX(inp.w,  showSpongeLayer),
            color='k',
            arrowsize=1,
            density=0.5,
            linewidth=0.5,
            zorder=1)


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


def animation(dataArray:       np.array,
              meta:            Meta,
              converter:       Callable[[np.ndarray, Meta], np.ndarray],
              prefix:          string  = '',
              maxValue:        float   = 0.3,
              showSpongeLayer: bool    = False,
              showStreamPlot:  bool    = False,
              cmapDivisions:   int     = 20,
              skip:            int     = 2,
              directory:       string  = 'test'):



    # find max value and set up colour bar
    #maxValue = max(np.max(inp.b), -np.min(inp.b))
    #maxValue = 8e-8
    divnorm = colors.TwoSlopeNorm(vmin=-maxValue, vcenter=0, vmax=maxValue)

    # set up colour map to have 31 discrete values
    cmap = plt.get_cmap('bwr', cmapDivisions)
    #cmap = 'bwr'

    x = meta.x[::skip, ::skip]
    z = meta.z[::skip, ::skip]

    if not os.path.exists(f'data/{directory}'):
        os.makedirs(f'data/{directory}')


    for i in tqdm.tqdm(range(len(dataArray))):
        data = dataArray[i]

        fig, ax = plt.subplots()
        # add colour bar and format plot
        ax.get_xaxis().set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, p: format(int(x / 1000), ',')))
        ax.get_yaxis().set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, p: format(int(x / 1000), ',')))


        from matplotlib.cm import ScalarMappable
        level_boundaries = np.linspace(-maxValue, maxValue, cmapDivisions + 1)


        # convert data using provided converter
        # this turns our 1D data into 2D by adding z-dep
        inp = converter(data, meta)

        #levels = [i for i in list(range(24))]
        #levels = [((i - (levels[-1]/2)) / 20) for i in levels]
        levels = [i for i in list(range(24))]
        levels = [((i - (levels[-1]/2)) / 30) for i in levels]

        # colour plot
        '''c = ax.pcolormesh(middleX(x, showSpongeLayer),
                      middleX(z, showSpongeLayer),
                      middleX(inp.b[::skip,::skip],  showSpongeLayer) * (273 / 10),
                      cmap=cmap,
                      zorder=0,
                      norm=divnorm,
                      shading='auto')'''
        c = ax.contourf(
                      middleX(x, showSpongeLayer),
                      middleX(z, showSpongeLayer),
                      middleX(inp.b[::skip,::skip],  showSpongeLayer) * (273 / 10),
                      cmap=cmap,
                      zorder=0,
                      levels=levels,
                      extend='both')
        cbar = fig.colorbar(c, ax=ax, ticks=[-0.3,-0.2,-0.1,0,0.1,0.2,0.3])


        # [-0.3,-0.2,-0.1,0,0.1,0.2,0.3]
        # [-0.25,-0.15,-0.05,0.05,0.15,0.25]

        # streamplot
        if showStreamPlot:
            sp = ax.streamplot(
                middleZ(meta.X[::skip], showSpongeLayer),
                meta.Z[::skip],
                middleX(inp.u[::skip,::skip],  showSpongeLayer),
                middleX(inp.w[::skip,::skip],  showSpongeLayer),
                color='k',
                arrowsize=1,
                density=0.5,
                linewidth=0.5,
                zorder=1)


        # plot alpha as a green dotted line (plotted as a percentage of the total depth)
        ax.plot(
            meta.x[0, :],
            meta.D * meta.spongeAlphaVectorized(meta.x[0, :]) / meta.spongeStrength,
            'k:', linewidth=0.75)


        m, s = divmod(inp.t, 60)
        h, m = divmod(m, 60)
        plt.title(f't = {h}:{"0" * (2 - len(str(m)))}{m}')

        #  save figure

        plt.xlim([0, np.max(meta.X)])
        plt.ylim([0, np.max(meta.Z)])

        fig.tight_layout()
        plt.savefig(f'data/{directory}/{i}.jpg')

        #if i == 0:
          #plt.show()


        plt.cla()
        plt.close(fig)


