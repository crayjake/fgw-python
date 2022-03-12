from generator import evaluate, simulate, convert, CrankNicolson, CrankNicolsonDeep
from animate import LineType, animator, Plot
from data import Meta, Data

import numpy as np

import gc
import os

anim = animator()

def stream(meta: Meta, name: str, path='../data'):
    print(f'\nStarting stream run: {name}\n')
    print(f'Simulating...')
    #simulationData = simulate(meta, CrankNicolson)
    simulationData = simulate(meta, CrankNicolsonDeep)

    print('Converting...')
    simulation, contToken = convert(simulationData)

    if not os.path.exists(f'{path}/{name}'):
        print(f'Creating directory: {path}/{name}')
        os.mkdir(f'{path}/{name}')
            
    print(f'Generating images...')
    anim.images(simulation, f'{path}/{name}', )
    #print(f'Creating gif...')
    #anim.gif(f'{path}/{name}/images/stream', f'{path}/{name}/stream.gif')
    #
    #print(f'Saving metadata...')
    #meta.json(f'{path}/{name}')
    #
    #print(f'Saving data...')
    #simulationData.save(f'{path}/{name}')

    simulationData = None
    simulation = None
    contToken = None
    gc.collect()
    print(f'\nFinished stream run: {name}\n')

def line(meta: Meta, name: str, lineType:LineType = LineType.ALL, path='../data'):
    #print(f'\nStarting line run: {name}\n')
    #print(f'Simulating...')
    #data = simulate(meta, CrankNicolson)
    data = simulate(meta, CrankNicolsonDeep)
    
    if not os.path.exists(f'{path}/{name}'):
        print(f'Creating directory: {path}/{name}')
        os.mkdir(f'{path}/{name}')
            
    print(f'Generating images...')
    anim.images(data, f'{path}/{name}', plotType=Plot.LINE, lineType=lineType)
    #print(f'Creating gif...')
    #anim.gif(f'{path}/{name}/images/line', f'{path}/{name}/line.gif')
    #
    #print(f'Saving metadata...')
    #meta.json(f'{path}/{name}')
    #
    #print(f'Saving data...')
    #data.save(f'{path}/{name}')

    data = None
    gc.collect()
    print(f'\nFinished line run: {name}\n')