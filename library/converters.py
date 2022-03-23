# -- library/converters.py --
# Author: Jake Cray
# GitHub: crayjake/fgw-python
''' contains all the converters (for adding z-dependence) '''

'''
TODO:
     - edit converter to allow the selection of a single mode 
     - add converter for shallow atmosphere (also single mode option)
'''

import tqdm
import numpy as np

from math import sqrt

from structures import Meta, State

def converter(inp: State, meta: Meta, mode: int = -1) -> np.ndarray:
    start = 0

    #output = np.array([])

    # defining variables for simplicity
    h     = meta.h * 1000
    rho_s = 1
    D     = meta.D
    N     = meta.N
    A_j   = sqrt(2 / (rho_s * (N ** 2) * D))

    #for i in tqdm.tqdm(range(start, data.data.shape[0])):
    #inp = data.data[i]
    
    u   = np.zeros(meta.x.shape)
    v   = np.zeros(meta.x.shape)
    w   = np.zeros(meta.x.shape)
    b   = np.zeros(meta.x.shape)
    p   = np.zeros(meta.x.shape)
    rho = np.zeros(meta.x.shape)
    t   = inp.t

    for i in range(len(meta.js)):
        if mode > -1:
            if mode != meta.js[i]:
                continue

        c_jSquared = (((N * meta.D) ** 2) / (((meta.js[i] * np.pi) ** 2) + ((meta.D ** 2)/(4 * (h ** 2))))) # wavespeed
        for z in range(meta.x.shape[0]):
            # phi like
            w[z, :]   += (inp.w[i]
                         * phi(z, meta.js[i], h, meta.D, meta.x.shape[0]-1, A_j))
            rho[z, :] += (inp.rho[i]
                         * ((rhoZero(z, rho_s, h, meta.D, meta.x.shape[0]-1) * (N ** 2)) / (c_jSquared))
                         * phi(z, meta.js[i], h, meta.D, meta.x.shape[0]-1, A_j))
            b[z, :]   += (inp.b[i] * ((N ** 2) / c_jSquared)
                         * phi(z, meta.js[i], h, meta.D, meta.x.shape[0]-1, A_j))

            # phi' like
            u[z, :]   += (inp.u[i]
                         * phiPrime(z, meta.js[i], h, meta.D, meta.x.shape[0]-1, A_j))
            v[z, :]   += (inp.v[i]
                         * phiPrime(z, meta.js[i], h, meta.D, meta.x.shape[0]-1, A_j))
            p[z, :]   += (inp.p[i] * rhoZero(z, rho_s, h, meta.D, meta.x.shape[0]-1)
                         * phiPrime(z, meta.js[i], h, meta.D, meta.x.shape[0]-1, A_j))
            sample = State(u=u, v=v, w=w, b=b, p=p, rho=rho, t=t)
        
        
        #output = np.append(output, [sample], 0)

    return sample

def phi(z, j, h, D, Dz, A):
    return A * np.exp(z / (2 * Dz * (h/D))) * np.sin((j * np.pi * z) / Dz)

def phiPrime(z, j, h, D, Dz, A):
    return A * np.exp(z / (2 * Dz * (h/D))) * (((1 / (2 * h)) * (np.sin((j * np.pi * z) / Dz))) + (((j * np.pi) / D) * (np.cos((j * np.pi * z) / Dz))))

def rhoZero(z, rho_s, h, D, Dz):
    return rho_s * np.exp(-z / (Dz * (h/D)))