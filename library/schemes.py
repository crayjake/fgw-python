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
    alpha = lambda x: meta.spongeAlphaVectorized(x) 


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

    al = alpha(meta.x[0, :])

    # step
    U = (B @ inp.u[i]) + (dt * f * inp.v[i] * (2 + (dt * al)) / 2) - (dt * (D1 @ (inp.p[i] * (2 + (dt * al)) / 2))) + ((dt ** 2) * (c_jSquared) * (D1 @ S_j) / 2)
    u = Ainv @ U

    v = (inp.v[i] - (dt * f * (u + inp.u[i]) / 2)) / (1 + (dt * al))

    w = (-1) * ((D1 @ (u + inp.u[i])) + inp.w[i])

    p = (inp.p[i] + ((dt * c_jSquared) * (((w + inp.w[i]) / 2) - S_j))) / (1 + (dt * al))
   
    rho = ((1 / g) * (inp.p[i] + p)) - inp.rho[i] # -(1/g)(dp/dz)

    b = -p - inp.p[i] - inp.b[i]


    return State(u=u, v=v, w=w, p=p, b=b, rho=rho, t=inp.t + meta.dt)

def Simple(meta: Meta, inp: State, i: int) -> State:
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
    alpha = lambda x: meta.spongeAlphaVectorized(x) 


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

    
    al = alpha(meta.x[0, :])

    S_j = meta.heatingForm(meta.x[0, :], L) * meta.S0 * (2 - H((inp.t + meta.dt) - T) - H(inp.t - T)) / 2


    # step
    if inp.t == 0:
        inp.u[:] = meta.heatingForm(meta.x[0, :], L)
        
    U = (B @ inp.u[i]) + (dt * inp.v[i] * (2 + (al * dt)) / 2)
    u = Ainv @ U

    v = (inp.v[i] + (dt * (D2 @ (u + inp.u[i]) / 2))) / (1 + (al * dt))



    return State(u=u, v=v, w=inp.w, p=inp.p, b=u, rho=inp.rho, t=inp.t + meta.dt)








def NoRotationExact(meta: Meta, time: int, mode: int) -> State:
    x = meta.x[0, :]

    # deep param
    j = mode
    h = meta.h * 1000
    rho_s = 1
    g = 10 # gravity
    N = meta.N
    D = meta.D
    D_t = meta.D_t * 1000

    # modal variables
    A_j = sqrt(2 / (rho_s * (N ** 2) * meta.D))
    
    if h == 0:
        c_jSquared = ((N * D) / (mode * pi)) ** 2 # wavespeed
    else:
        c_jSquared = ((N * D) ** 2) / (((mode * pi) ** 2) + ((D ** 2)/(4 * (h ** 2)))) # wavespeed

    # calculating S_j
    if ((j * D_t / D) - 1) == 0:
        S_j = A_j * (rho_s / 2) * D_t
    else:
        S_A = np.sin((np.pi) * ((j * D_t / D) - 1)) / ((j * D_t / D) - 1)
        S_B = np.sin((np.pi) * ((j * D_t / D) + 1)) / ((j * D_t / D) + 1)
        S_j = A_j * (rho_s / 2) * (D_t / np.pi) * (S_A - S_B)

    S_j = S_j * meta.S0


    u   = evalU(meta, x, time, mode, S_j, sqrt(c_jSquared))
    v   = np.zeros(x.shape)
    w   = evalW(meta, x, time, mode, S_j, sqrt(c_jSquared))
    b   = evalB(meta, x, time, mode, S_j, sqrt(c_jSquared))
    p   = np.zeros(x.shape)
    rho = np.zeros(x.shape)

    return State(u=u, v=v, w=w, p=p, b=b, rho=rho, t=time)



def evalW(meta: Meta, x, t, j, S0sigmaN, c_j):
    L = meta.L

    part1 = (1 - H(t - meta.T)) * F(x, L)
    part2 = - (F(x + (c_j * t), L) + F(x - (c_j * t), L)) / 2
    part3 = H(t - meta.T) * (F(x + (c_j * (t - meta.T)), L) + F(x - (c_j * (t - meta.T)), L)) / 2

    return S0sigmaN * (part1 + part2 + part3)


def evalU(meta: Meta, x, t, j, S0sigmaN, c_j):
    L = meta.L

    part1 = - (1 - H(t - meta.T)) * G(x, L)
    part2 = (G(x + (c_j * t), L) + G(x - (c_j * t), L)) / 2
    part3 = - H(t - meta.T) * (G(x + (c_j * (t - meta.T)), L) + G(x - (c_j * (t - meta.T)), L)) / 2

    return S0sigmaN * (part1 + part2 + part3)


def evalB(meta: Meta, x, t, j, S0sigmaN, c_j):
    L = meta.L

    part1 = - (G(x + (c_j * t), L) - G(x - (c_j * t), L)) / 2
    part2 = H(t - meta.T) * (G(x + (c_j * (t - meta.T)), L) - G(x - (c_j * (t - meta.T)), L)) / 2

    return - c_j * S0sigmaN * (part1 + part2)