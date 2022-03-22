# -- library/schemes.py --
# Author: Jake Cray
# GitHub: crayjake/fgw-python
''' contains the implementations of the numerical schemes '''

import numpy as np

from math import sqrt

from structures import *
from helpers import H

def CrankNicolsonDeep(meta: Meta, inp: State, i: int) -> State:
    # matrices
    Ainv = meta.Ainv[i]
    B = meta.B[i]
    D1 = meta.D1
    D2 = meta.D2

    # parameters (for simpler code)
    T = meta.T
    L = meta.L
    N = meta.N
    dt = meta.dt
    f = meta.f

    # sponge damping
    def alpha(xs):
        x = np.copy(xs)
        return np.where(((meta.W / 2) - meta.spongeWidth) < abs(x), 4 * meta.spongeStrength * (abs(x) - ((meta.W / 2) - meta.spongeWidth)) / (meta.W / 2), 0)

    # deep param
    j = meta.js[i]
    h = meta.h * 1000
    rho_s = 1
    g = 10 # gravity
    N = meta.N
    D = meta.D
    D_t = meta.D_t * 1000

    # modal variables
    A_j = sqrt(2 / (rho_s * (N ** 2) * meta.D))

    if h == 0:
        c_jSquared = ((N * D) / (meta.js[i] * pi)) ** 2 # wavespeed
    else:
        c_jSquared = ((N * D) ** 2) / (((meta.js[i] * pi) ** 2) + ((D ** 2)/(4 * (h ** 2)))) # wavespeed

    

    # calculating S_j
    if ((j * D_t / D) - 1) == 0:
        S_j = A_j * (rho_s / 2) * D_t
    else:
        S_A = np.sin((np.pi) * ((j * D_t / D) - 1)) / ((j * D_t / D) - 1)
        S_B = np.sin((np.pi) * ((j * D_t / D) + 1)) / ((j * D_t / D) + 1)
        S_j = A_j * (rho_s / 2) * (D_t / np.pi) * (S_A - S_B)

    S_j = S_j * meta.heatingForm(meta.x[0, :], L) * meta.S0 * (2 - H((inp.t + meta.dt) - T) - H(inp.t - T)) / 2

    # step
    U = (B @ inp.u[i]) + (dt * f * inp.v[i] * (2 + (dt * alpha(meta.x[0, :]))) / 2) - (dt * (D1 @ (inp.p[i] * (2 + (dt * alpha(meta.x[0, :]))) / 2))) + ((dt ** 2) * (c_jSquared) * (D1 @ S_j) / 2)
    u = Ainv @ U

    v = (inp.v[i] - (dt * f * (u + inp.u[i]) / 2)) / (1 + (dt * alpha(meta.x[0, :])))

    w = (-1) * ((D1 @ (u + inp.u[i])) + inp.w[i])

    p = (inp.p[i] + ((dt * c_jSquared) * (((w + inp.w[i]) / 2) - S_j))) / (1 + (dt * alpha(meta.x[0, :])))
   
    rho = ((1 / g) * (inp.p[i] + p)) - inp.rho[i] # -(1/g)(dp/dz)

    b = -p - inp.p[i] - inp.b[i]


    return State(u=u, v=v, w=w, p=p, b=b, rho=rho, t=inp.t + meta.dt)
