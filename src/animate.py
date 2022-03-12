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

    def images(self, data: Data, path, title:bool = False, plotType:Plot = Plot.STREAM, lineType:LineType = None):
        skip = int(data.meta.x.shape[1]/200)
        prefix = ''
        if plotType == Plot.STREAM:
            prefix = 'stream'
        elif plotType == Plot.LINE:
            prefix = 'line'

        if not os.path.exists(f'{path}/images'):
            os.mkdir(f'{path}/images')
        
        if not os.path.exists(f'{path}/images/{prefix}'):
            os.mkdir(f'{path}/images/{prefix}')
        
        x = data.meta.x[::,::skip]/1000
        z = data.meta.z[::,::skip]/1000
        index = int(data.data[0].t / data.meta.dt)
        for inp in tqdm(data.data):
            fig, ax = plt.subplots()
            
            if plotType == Plot.STREAM:
                c = ax.pcolor(x, z, inp.b[::,::skip] * (273 / 10), cmap='plasma', shading='auto', zorder=0)#, norm=divnorm)
                sp = ax.streamplot(x, z, inp.u[::,::skip], inp.w[::,::skip], color='k',   arrowsize=1, density=1, linewidth=0.5, zorder=1)#, linewidth=lw)#,    density=0.8) # color=lw, cmap='Greys')

                fig.colorbar(c, ax=ax)

            elif plotType == Plot.LINE:
                if lineType == LineType.U or lineType == LineType.ALL:
                    ax.plot(data.meta.x[0,::skip]/1000, inp.u[0][::skip], f'k{"--" if lineType == LineType.ALL else ""}')
                if lineType == LineType.W or lineType == LineType.ALL:
                    ax.plot(data.meta.x[0,::skip]/1000, inp.w[0][::skip]*10, f'k{"-" if lineType == LineType.ALL else ""}')
                if lineType == LineType.B or lineType == LineType.ALL:
                    ax.plot(data.meta.x[0,::skip]/1000, inp.b[0][::skip] * (273 / 10), f'k{":" if lineType == LineType.ALL else ""}')
                if lineType == LineType.P:
                    ax.plot(data.meta.x[0,::skip]/1000, inp.p[0][::skip]*10, f'k')

            if title:        
                timeString = time.strftime('%H:%M:%S', time.gmtime(inp.t))
                plt.title(f't = {timeString}')

            fig.savefig(f'{path}/images/{prefix}/{index}.jpg', bbox_inches='tight', transparent=False, facecolor='white')
            plt.close(fig)
            
            index += 1

    def gif(self, input: str, output: str, framerate: int = 10):
        file = output
        if not output.__contains__('.gif'):
            file = f'{output}/run.gif'
        
        subprocess.run([f'echo Y | ffmpeg -framerate {framerate} -start_number 0 -i {input}/%d.jpg -vf "scale=-2:512" -pix_fmt yuv420p {file}'], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    def display(self, data, time, prefix=''):
        skip = 1#math.ceil(data.meta.x.shape[1]/200)
        #print(f'SKIP: {math.ceil(data.meta.x.shape[1]/200)}')
        inp = data.data[time]
        fig, ax = plt.subplots()

        #divnorm = colors.BoundaryNorm(np.linspace(-0.3, 0.3, 30), plt.get_cmap('bwr').N)
        divnorm = colors.TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3)
        #divnorm=colors.TwoSlopeNorm(vcenter=0)

        #c = ax.pcolor(data.meta.x[::,::skip]/1000, data.meta.z[::,::skip]/1000, inp.b[::,::skip] * (273 / 10), cmap=plt.get_cmap('bwr', 30), shading='auto', zorder=0, norm=divnorm)
        c = ax.pcolor(data.meta.x/1000, data.meta.z/1000, inp.b * (273 / 10), cmap=plt.get_cmap('bwr', 30), zorder=0, norm=divnorm)
 
        fig.colorbar(c, ax=ax)
        
        timeString = time.strftime('%H:%M:%S', time.gmtime(inp.t))
        plt.title(f't = {timeString}')
        plt.show()

    def display_line(self, data, time, lineType = LineType.ALL, prefix=''):
        skip = math.ceil(data.meta.x.shape[1]/200)
        inp = data.data[time]
        fig, ax = plt.subplots()

        if lineType == LineType.U or lineType == LineType.ALL:
            ax.plot(data.meta.x[0,::skip]/1000, inp.u[0][::skip], f'b{"--" if lineType == LineType.ALL else ""}')
        if lineType == LineType.V or lineType == LineType.ALL:
            ax.plot(data.meta.x[0,::skip]/1000, inp.v[0][::skip], f'p{"--" if lineType == LineType.ALL else ""}')
        if lineType == LineType.W or lineType == LineType.ALL:
            ax.plot(data.meta.x[0,::skip]/1000, inp.w[0][::skip]*10, f'r{"-" if lineType == LineType.ALL else ""}')
        if lineType == LineType.B or lineType == LineType.ALL:
            ax.plot(data.meta.x[0,::skip]/1000, inp.b[0][::skip]*10, f'k{":" if lineType == LineType.ALL else ""}')
        if lineType == LineType.P or lineType == LineType.ALL:
            ax.plot(data.meta.x[0,::skip]/1000, inp.p[0][::skip]*10, f'g:')
        plt.title(f'{prefix} at t = {inp.t}')
        plt.show()