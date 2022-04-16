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
import matplotlib.pyplot as plt

# plt.rcParams['pcolor.shading'] = 'nearest'
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 250

import tqdm

from math import sqrt
from helpers import middleX, middleZ
from structures import State, Meta
from converters import phi, phiPrime, rhoZero

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
                  middleX(inp.v,  showSpongeLayer) * (273 / 10),
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
              directory:       string  = 'test',
              oneSided:        bool    = True
              ):

    print(f'Starting animation: {directory}')

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


        # convert data using provided converter
        # this turns our 1D data into 2D by adding z-dep
        inp = converter(data, meta)

        levels = [i for i in list(range(cmapDivisions))]
        levels = [((i - (levels[-1]/2))) for i in levels]
        levels = [maxValue * i / levels[-1] for i in levels]

        ticks = [-0.3,-0.2,-0.1,0,0.1,0.2,0.3]
        ticks = [float('%.2g' % (maxValue * i / 0.3)) for i in ticks]

        # colour plot
        c = ax.contourf(
                      x,
                      z,
                      inp.b[::skip,::skip] * (273 / 10),
                      cmap='bwr',
                      zorder=0,
                      levels=levels,
                      extend='both')
        cbar = fig.colorbar(c, ax=ax, ticks=ticks)


        # streamplot
        if showStreamPlot:
            U = inp.u[::skip,::skip]
            W = inp.w[::skip,::skip]

            magnitude = U ** 2 + W ** 2

            U = np.where(magnitude > 0.001 * np.max(magnitude), U, np.zeros(U.shape))
            W = np.where(magnitude > 0.001 * np.max(magnitude), W, np.zeros(W.shape))


            stream = ax.streamplot(
                meta.X[::skip],
                meta.Z[::skip],
                U,
                W,
                color='k',
                #norm = divnorm,
                #cmap = plt.get_cmap('bwr', 21),
                linewidth = 1,
                arrowsize = 1,
                density = 1
            )


        # plot alpha as a green dotted line (plotted as a percentage of the total depth)
        if showSpongeLayer:
            ax.plot(
                meta.x[0, :],
                meta.D * meta.spongeAlphaVectorized(meta.x[0, :]) / meta.spongeStrength,
                'k:', linewidth=1)


        m, s = divmod(inp.t, 60)
        h, m = divmod(m, 60)
        plt.title(f't = {h}:{"0" * (2 - len(str(m)))}{m}')

        #  save figure

        spongeWidth = 0 if showSpongeLayer else meta.spongeWidth
        if oneSided:
            plt.xlim([0, np.max(meta.X) - spongeWidth])
        else:
            plt.xlim([np.min(meta.X) + spongeWidth, np.max(meta.X) - spongeWidth])
        plt.ylim([0, np.max(meta.Z)])

        fig.tight_layout()
        plt.savefig(f'data/{directory}/{i}.jpg')

        #if i == 0:
          #plt.show()


        plt.cla()
        plt.close(fig)



def plotAxes(ax, x, z, data, meta, converter, levels, showStreamPlot, showSpongeLayer, skip):
     # add colour bar and format plot
    ax.get_xaxis().set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, p: format(int(x / 1000), ',')))
    ax.get_yaxis().set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, p: format(int(x / 1000), ',')))


    # convert data using provided converter
    # this turns our 1D data into 2D by adding z-dep
    inp = converter(data, meta)

    
    # colour plot
    c = ax.contourf(
                    x,
                    z,
                    inp.b[::skip,::skip] * (273 / 10),
                    cmap='bwr',
                    zorder=0,
                    levels=levels,
                    extend='both')





    # streamplot
    if showStreamPlot:
        U = inp.u[::skip,::skip]
        W = inp.w[::skip,::skip]

        magnitude = U ** 2 + W ** 2

        U = np.where(magnitude > 0.001 * np.max(magnitude), U, np.zeros(U.shape))
        W = np.where(magnitude > 0.001 * np.max(magnitude), W, np.zeros(W.shape))


        stream = ax.streamplot(
            meta.X[::skip],
            meta.Z[::skip],
            U,
            W,
            color='k',
            #norm = divnorm,
            #cmap = plt.get_cmap('bwr', 21),
            linewidth = 1,
            arrowsize = 1,
            density = 1
        )


    # plot alpha as a green dotted line (plotted as a percentage of the total depth)
    if showSpongeLayer:
        ax.plot(
            meta.x[0, :],
            meta.D * meta.spongeAlphaVectorized(meta.x[0, :]) / meta.spongeStrength,
            'k:', linewidth=1)


    m, s = divmod(inp.t, 60)
    h, m = divmod(m, 60)
    ax.legend(title=f't = {h}:{"0" * (2 - len(str(m)))}{m}', loc='upper left', labelspacing=0)

    return c


# plots a set of images, left and right
def plotGroup(leftData:        np.array,
              rightData:       np.array,
              meta:            Meta,
              converter:       Callable[[np.ndarray, Meta], np.ndarray],
              maxValue:        float   = 0.3,
              showSpongeLayer: bool    = False,
              showStreamPlot:  bool    = False,
              cmapDivisions:   int     = 20,
              skip:            int     = 2,
              directory:       string  = 'test',
              oneSided:        bool    = True,
              figsize = (15,15)
              ):

    print(f'Starting animation: {directory}')

    x = meta.x[::skip, ::skip]
    z = meta.z[::skip, ::skip]

    if not os.path.exists(f'data/groups'):
        os.makedirs(f'data/groups')

    height = max(len(leftData), len(rightData))
    fig, axes = plt.subplots(height, 2, sharex=True, sharey=True, figsize=figsize)

    if len(axes.shape) == 2:
        leftAxes = axes[:,0]
        rightAxes = axes[:,1]
    else:
        leftAxes = [axes[0]]
        rightAxes = [axes[1]]

    levels = [i for i in list(range(cmapDivisions))]
    levels = [((i - (levels[-1]/2))) for i in levels]
    levels = [maxValue * i / levels[-1] for i in levels]

    ticks = [-0.3,-0.2,-0.1,0,0.1,0.2,0.3]
    ticks = [float('%.2g' % (maxValue * i / 0.3)) for i in ticks]


    for i in range(height):
        lax = leftAxes[i]
        rax = rightAxes[i]

        if i >= len(leftData):
            lax.axis('off')
        else:
            left  = leftData[i]

            c = plotAxes(lax, x, z, left, meta, converter, levels, showStreamPlot, showSpongeLayer, skip)
            
            spongeWidth = 0 if showSpongeLayer else meta.spongeWidth
            if oneSided:
                lax.set_xlim([0, np.max(meta.X) - spongeWidth])
            else:
                lax.set_xlim([np.min(meta.X) + spongeWidth, np.max(meta.X) - spongeWidth])
            lax.set_ylim([0, np.max(meta.Z)])

        if i >= len(rightData):
            rax.axis('off')
        else:
            right = rightData[i]
           
            c = plotAxes(rax, x, z, right, meta, converter, levels, showStreamPlot, showSpongeLayer, skip)

            spongeWidth = 0 if showSpongeLayer else meta.spongeWidth
            if oneSided:
                rax.set_xlim([0, np.max(meta.X) - spongeWidth])
            else:
                rax.set_xlim([np.min(meta.X) + spongeWidth, np.max(meta.X) - spongeWidth])
            rax.set_ylim([0, np.max(meta.Z)])


    fig.tight_layout()
    # axes.ravel().tolist()
    cbar = fig.colorbar(c, ax=axes.ravel().tolist(), ticks=ticks, location='bottom', shrink=0.5, pad=0.05)

    plt.savefig(f'data/groups/{directory}.jpg', bbox_inches = 'tight', pad_inches = 0)
