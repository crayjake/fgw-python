# -- library/generate.py --
# Author: Jake Cray
# GitHub: crayjake/fgw-python
''' method for generating the data '''

import tqdm

from structures import *
from typing import Callable

def generate(meta: Meta, step: Callable[[State, int, np.ndarray, np.ndarray], State]):
    print(f'Generating data...')
    output = np.array([])

    empty = np.array([np.zeros(meta.x.shape[1], dtype='float64')] * len(meta.js))

    u   = np.copy(empty)
    v   = np.copy(empty)
    w   = np.copy(empty)
    b   = np.copy(empty)
    p   = np.copy(empty)
    rho = np.copy(empty)

    inp = State(u=u, v=v, w=w, b=b, p=p, rho=rho, t=0)
    inp._2D()

    output = np.append(output, [inp], 0)

    saveEvery = meta.timesteps if meta.saveEvery == 0 else meta.saveEvery
    previousSample = inp

    for i in range(len(meta.js)):
        print(f'Creating coefficient matrices for mode: {meta.js[i]}')
        B    = meta.B(   meta.js[i])
        Ainv = meta.Ainv(meta.js[i])

        print(f'Generating data for mode: {meta.js[i]}')
        for t in tqdm.tqdm(range(1, int(meta.timesteps))):
            inp = State(u=np.copy(empty), v=np.copy(empty), w=np.copy(empty), b=np.copy(empty), p=np.copy(empty), rho=np.copy(empty), t=t*meta.dt)
            if t % saveEvery == 0 and not (i == 0):
                inp = output[t % saveEvery]


            
            sample     = step(meta=meta, inp=previousSample, i=i, Ainv=Ainv, B=B)
            inp.u[i]   = sample.u
            inp.v[i]   = sample.v
            inp.w[i]   = sample.w
            inp.b[i]   = sample.b
            inp.p[i]   = sample.p
            inp.rho[i] = sample.rho

            inp._2D()
            previousSample = inp

            if t % saveEvery == 0:
                if i == 0:
                    # if this is a timestep to save then do so
                    output = np.append(output, [inp], 0)
                else:
                    # if this is a timestep to save then do so
                    output[t % saveEvery] = [inp]                

    return output