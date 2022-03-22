# -*- coding:utf-8 -*-
# Author: Jake Cray
# GitHub: crayjake/fgw-python
# File:   library/generate.py
''' method for generating the data '''

import tqdm

from structures import *
from typing import Callable

def generate(meta: Meta, step: Callable[[State, int], State]):
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

    for t in tqdm.tqdm(range(1, int(meta.timesteps))):
        inp = State(u=np.copy(empty), v=np.copy(empty), w=np.copy(empty), b=np.copy(empty), p=np.copy(empty), rho=np.copy(empty), t=t*meta.dt)
        for i in range(len(meta.js)):
            # step the data using provided scheme
            sample     = step(meta=meta, inp=previousSample, i=i)
            inp.u[i]   = sample.u
            inp.v[i]   = sample.v
            inp.w[i]   = sample.w
            inp.b[i]   = sample.b
            inp.p[i]   = sample.p
            inp.rho[i] = sample.rho
        # ensure data is correct shape
        inp._2D()
        previousSample = inp
        
        # if this is a timestep to save then do so
        if t % saveEvery == 0:
            output = np.append(output, [inp], 0)

    return output