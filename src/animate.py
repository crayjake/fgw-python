## Generation of plots/videos/gifs based off of .npy files
import numpy as np
from tqdm import tqdm
import os
import math
import time

import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

from matplotlib import colors

from data import Data

import subprocess

from enum import Enum

class Plot(Enum):
    STREAM=0,
    LINE=1

class LineType(Enum):
    U=1,
    W=2,
    B=3,
    P=4,
    V=5,
    ALL=0

class animator():

    def images(self, data: Data, path, title:bool = False, plotType:Plot = Plot.STREAM, lineType:LineType = None, sponge=True):
        skip = int(data.meta.x.shape[1]/200)
        prefix = ''
        if plotType == Plot.STREAM:
            prefix = 'stream'
        elif plotType == Plot.LINE:
            prefix = 'line'

        if not os.path.exists(f'{path}/images'):
            os.makedirs(f'{path}/images')
        
        if not os.path.exists(f'{path}/images/{prefix}'):
            os.makedirs(f'{path}/images/{prefix}')
        
        x = middleZ(data.meta.X, sponge)
        z = data.meta.Z
        index = int(data.data[0].t / data.meta.dt)
        for inp in tqdm(data.data):
            fig, ax = plt.subplots()
            
            if plotType == Plot.STREAM:
                divnorm = colors.TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3)
                c = ax.pcolor(middleX(data.meta.x, sponge), middleX(data.meta.z, sponge), middleX(inp.b, sponge) * (273 / 10), cmap=plt.get_cmap('bwr', 30), zorder=0, norm=divnorm)
                sp = ax.streamplot(x, z, middleX(inp.u, sponge), middleX(inp.w, sponge), color='k',   arrowsize=1, density=0.5, linewidth=0.5, zorder=1)#, linewidth=lw)#,    density=0.8) # color=lw, cmap='Greys')

                fig.colorbar(c, ax=ax)

            elif plotType == Plot.LINE:
                if lineType == LineType.U or lineType == LineType.ALL:
                    ax.plot(middleX(data.meta.x, sponge)[0,::skip], middleX(inp.u, sponge)[0][::skip], f'k{"--" if lineType == LineType.ALL else ""}')
                if lineType == LineType.W or lineType == LineType.ALL:
                    ax.plot(middleX(data.meta.x, sponge)[0,::skip], middleX(inp.w, sponge)[0][::skip]*10, f'k{"-" if lineType == LineType.ALL else ""}')
                if lineType == LineType.B or lineType == LineType.ALL:
                    ax.plot(middleX(data.meta.x, sponge)[0,::skip], middleX(inp.b, sponge)[0][::skip] * (273 / 10), f'k{":" if lineType == LineType.ALL else ""}')
                if lineType == LineType.P:
                    ax.plot(middleX(data.meta.x, sponge)[0,::skip], middleX(inp.p, sponge)[0][::skip]*10, f'k')

            if title:        
                timeString = time.strftime('%H:%M:%S', time.gmtime(inp.t))
                plt.title(f't = {timeString}')

            ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x / 1000), ',')))
            ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x / 1000), ',')))
            fig.savefig(f'{path}/images/{prefix}/{index}.jpg', bbox_inches='tight', transparent=False, facecolor='white')
            # Clear the current axes.
            plt.cla() 
            # Clear the current figure.
            plt.clf() 
            # Closes all the figure windows.
            plt.close('all')
            
            index += 1

    def gif(self, input: str, output: str, framerate: int = 10):
        file = output
        if not output.__contains__('.gif'):
            file = f'{output}/run.gif'
        
        subprocess.run([f'echo Y | ffmpeg -framerate {framerate} -start_number 0 -i {input}/%d.jpg -vf "scale=-2:512" -pix_fmt yuv420p {file}'], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    def display(self, data, t, prefix='', sponge=True):
        skip = 1#math.ceil(data.meta.x.shape[1]/200)
        #print(f'SKIP: {math.ceil(data.meta.x.shape[1]/200)}')
        inp = data.data[t]
        fig, ax = plt.subplots()

        #divnorm = colors.BoundaryNorm(np.linspace(-0.3, 0.3, 30), plt.get_cmap('bwr').N)
        divnorm = colors.TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3)
        #divnorm=colors.TwoSlopeNorm(vcenter=0)

        #c = ax.pcolor(data.meta.x[::,::skip]/1000, data.meta.z[::,::skip]/1000, inp.b[::,::skip] * (273 / 10), cmap=plt.get_cmap('bwr', 30), shading='auto', zorder=0, norm=divnorm)
        c = ax.pcolor(middleX(data.meta.x, sponge), middleX(data.meta.z, sponge), middleX(inp.b, sponge) * (273 / 10), cmap=plt.get_cmap('bwr', 30), zorder=0, norm=divnorm)
 
        #skip = int(middleX(data.meta.x, sponge).shape[1]/200)
        sp = ax.streamplot(middleZ(data.meta.X, sponge), data.meta.Z, middleX(inp.u, sponge), middleX(inp.w, sponge), color='k', arrowsize=1, density=0.5, linewidth=0.5, zorder=1)#, linewidth=lw)#,    density=0.8) # color=lw, cmap='Greys')
    
        #countour = ax.contou

        fig.colorbar(c, ax=ax)

        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x / 1000), ',')))
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x / 1000), ',')))
        timeString = time.strftime('%H:%M:%S', time.gmtime(inp.t))
        plt.title(f't = {timeString}')
        plt.show()

    def display_line(self, data, time, lineType = LineType.ALL, prefix='', sponge=True):
        skip = math.ceil(middleX(data.meta.x).shape[1]/200)
        inp = data.data[time]
        fig, ax = plt.subplots()

        if lineType == LineType.U or lineType == LineType.ALL:
            ax.plot(middleX(data.meta.x, sponge)[0], middleX(inp.u, sponge)[0], f'b{"--" if lineType == LineType.ALL else ""}')
        if lineType == LineType.V or lineType == LineType.ALL:
            ax.plot(middleX(data.meta.x, sponge)[0], middleX(inp.v, sponge)[0], f'p{"--" if lineType == LineType.ALL else ""}')
        if lineType == LineType.W or lineType == LineType.ALL:
            ax.plot(middleX(data.meta.x, sponge)[0], middleX(inp.w, sponge)[0]*10, f'r{"-" if lineType == LineType.ALL else ""}')
        if lineType == LineType.B or lineType == LineType.ALL:
            ax.plot(middleX(data.meta.x, sponge)[0], middleX(inp.b, sponge)[0]*10, f'k{":" if lineType == LineType.ALL else ""}')
        if lineType == LineType.P or lineType == LineType.ALL:
            ax.plot(middleX(data.meta.x, sponge)[0,::skip], middleX(inp.p, sponge)[0][::skip]*10, f'g:')

        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x / 1000), ',')))
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x / 1000), ',')))
        plt.title(f'{prefix} at t = {inp.t}')
        plt.show()

# gets middle 3/4 of list (2nd dim)
def middleX(arr, sponge):
    if not sponge:
        return arr        
    return arr[::,math.ceil(len(arr)/8):][::,:-math.ceil(len(arr)/8)]

# gets middle 3/4 of list (1st dim)
def middleZ(arr, sponge):
    if not sponge:
        return arr
    return arr[math.ceil(len(arr)/8):][:-math.ceil(len(arr)/8)]
