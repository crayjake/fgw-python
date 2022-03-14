## Simulation of forced gravity wave equations
from cmath import sqrt
from curses import A_STANDOUT
from tkinter import E
import numpy as np
from numpy.lib import save
import tqdm
from typing import Callable

from data import DataSample, Data, Meta

import gc

import os
import math

#region Simulation methods

#region Simulation methods
def CrankNicolsonDeepDamped(meta: Meta, inp: DataSample, i: int) -> DataSample:
    # matrices
    Ainv = meta.Ainv_damped[i]
    B = meta.B_damped[i]
    D1 = meta.D1
    D2 = meta.D2

    # parameters (for simpler code)
    T = meta.Ts[i]
    L = meta.L
    N = meta.N
    dt = meta.dt
    f = meta.f

    def alpha(xs):
        al = meta.alpha
        x = np.copy(xs)
        return al * (np.where(x > (3 * meta.space / 8), al * (abs(x) - (3 * (meta.space) / 8)) / (meta.space/2), 0) + np.where(x < -(3 * meta.space / 8), al * (abs(x) - (3 * (meta.space) / 8)) / (meta.space/2), 0))

    # deep param
    j = meta.js[i]
    h = 100000 # scale height
    rho_s = 1
    g = 10 # gravity
    N = 0.01
    D = meta.D
    D_t = 10000

    # modal variables
    A_j = math.sqrt(2 / (rho_s * (N ** 2) * meta.D))
    c_jSquared = (((N * meta.D) ** 2) / (((meta.js[i] * np.pi) ** 2) + ((meta.D ** 2)/(4 * (h ** 2))))) # wavespeed

    # calculating S_j
    if ((j * D_t / D) - 1) == 0:
        S_j = A_j * (rho_s / 2) * D_t
    else:
        S_A = np.sin((np.pi) * ((j * D_t / D) - 1)) / ((j * D_t / D) - 1)
        S_B = np.sin((np.pi) * ((j * D_t / D) + 1)) / ((j * D_t / D) + 1)
        S_j = A_j * (rho_s / 2) * (D_t / np.pi) * (S_A - S_B)

    S_j = S_j * F(meta.x[0, :], L) * meta.S0 * (H(T - (inp.t + meta.dt)) + H(T - inp.t)) / 2

    # step
    U = (B @ inp.u[i]) + (dt * f * inp.v[i] * (2 + (dt * alpha(meta.x[0, :])))) - (dt * (D1 @ (inp.p[i] * (2 + (dt * alpha(meta.x[0, :])))))) + ((dt ** 2) * (c_jSquared) * (D1 @ S_j) / 2)
    u = Ainv @ U

    v = (inp.v[i] - (dt * f * (u + inp.u[i]) / 2)) / (1 + (dt * alpha(meta.x[0, :])))

    w = (-1) * ((D1 @ (u + inp.u[i])) + inp.w[i])

    p = (inp.p[i] + ((dt * c_jSquared) * (((w + inp.w[i]) / 2) - S_j))) / (1 + (dt * alpha(meta.x[0, :])))

    rho = ((1 / g) * (inp.p[i] + p)) - inp.rho[i] # -(1/g)(dp/dz)

    b = -p - inp.p[i] - inp.b[i]

    return DataSample(b=b, u=u, v=v, w=w, p=p, rho=rho, t=inp.t + meta.dt)

def CrankNicolsonDeep(meta: Meta, inp: DataSample, i: int) -> DataSample:
    # matrices
    Ainv = meta.Ainv_new[i]
    B = meta.B_new[i]
    D1 = meta.D1
    D2 = meta.D2

    # parameters (for simpler code)
    T = meta.Ts[i]
    L = meta.L
    N = meta.N
    dt = meta.dt
    f = meta.f
    al = meta.alpha

    # deep param
    j = meta.js[i]
    h = 100000 # scale height
    rho_s = 1
    g = 10 # gravity
    N = 0.01
    D = meta.D
    D_t = 10000

    # modal variables
    A_j = math.sqrt(2 / (rho_s * (N ** 2) * meta.D))
    c_jSquared = (((N * meta.D) ** 2) / (((meta.js[i] * np.pi) ** 2) + ((meta.D ** 2)/(4 * (h ** 2))))) # wavespeed

    # calculating S_j
    if ((j * D_t / D) - 1) == 0:
        S_j = A_j * (rho_s / 2) * D_t
    else:
        S_A = np.sin((np.pi) * ((j * D_t / D) - 1)) / ((j * D_t / D) - 1)
        S_B = np.sin((np.pi) * ((j * D_t / D) + 1)) / ((j * D_t / D) + 1)
        S_j = A_j * (rho_s / 2) * (D_t / np.pi) * (S_A - S_B)

    S_j = S_j * F(meta.x[0, :], L) * meta.S0 * (H(T - (inp.t + meta.dt)) + H(T - inp.t)) / 2

    # step
    U = (B @ inp.u[i]) + (dt * f * inp.v[i]) - (dt * (D1 @ inp.p[i])) + ((dt ** 2) * (c_jSquared) * (D1 @ S_j) / 2)
    u = Ainv @ U

    v = inp.v[i] - (dt * f * (u + inp.u[i]) / 2)

    w = (-1) * ((D1 @ (u + inp.u[i])) + inp.w[i])

    p = inp.p[i] + ((dt * c_jSquared) * (-1) * ((D1 @ (u + inp.u[i]) / 2) + S_j))

    rho = ((1 / g) * (inp.p[i] + p)) - inp.rho[i] # -(1/g)(dp/dz)

    b = -p - inp.p[i] - inp.b[i]

    return DataSample(b=b, u=u, v=v, w=w, p=p, rho=rho, t=inp.t + meta.dt)

def CrankNicolson(meta: Meta, inp: DataSample, i: int) -> DataSample:
    # matrices
    Ainv = meta.Ainv_uNEW[i]
    A = meta.A_uNEW[i]
    B = meta.B_uNEW[i]
    D1 = meta.D1
    D2 = meta.D2

    # parameters (for simpler code)
    m = meta.js[i] * np.pi / meta.D
    Q = meta.Qs[i]
    T = meta.Ts[i]
    L = meta.L
    S = meta.S0 * Q * F(meta.x[0, :], L) * (H(T - (inp.t + meta.dt)) + H(T - inp.t)) / 2
    N = meta.N
    dt = meta.dt

    n = meta.n
    f = meta.f
    al = meta.alpha

    alphPlus = 1 + (al * dt / 2)
    alphMinus = 1 - (al * dt / 2)

    # step
    U = (B @ inp.u[i]) + ((2 * n / m) * (D1 @ inp.w[i])) + (dt * ((f * inp.v[i]) + (((1 + (alphMinus/alphPlus))) * (D1 @ inp.b[i]) / (2 * m)))) + ((dt ** 2) * (D1 @ S) / (2 * m * alphPlus))
    u = Ainv @ U

    v = inp.v[i] - (dt * f * (u + inp.u[i]) / 2)

    w = (-(D1 @ (u + inp.u[i])) / m) - inp.w[i]

    b = ((inp.b[i] * alphMinus) + (dt * (S - ((N ** 2) * (w + inp.w[i]) / 2)))) / alphPlus

    p = (((2 * n) / (dt)) * (w - inp.w[i])) - ((b + inp.b[i]) / m) - inp.p[i]
    
    return DataSample(b=b, u=u, v=v, w=w, p=p, t=inp.t + meta.dt)

def EulerStep(meta: Meta, inp: DataSample, m: int, heat: int, heatTime: int) -> DataSample:
    
    # WARNING: not sure this is up to date
    
    mode = m * np.pi / meta.D
   
    u = inp.u + meta.dt * (-(np.matmul(meta.D1, inp.p)))
    v = inp.v
    # w = inp.w - (0.5/mode) * meta.dt * (np.matmul(meta.D1, inp.u + u))
    w = - (np.matmul(meta.D1, inp.u) / mode)
    b = inp.b + meta.dt * ((heat * F(meta.x[0, :], meta.L) * meta.H(heatTime - (inp.t + meta.dt))) - (meta.N * meta.N * inp.w))
    p = - b / mode
    
    return DataSample(b=b, u=u, v=v, w=w, p=p, t=inp.t + meta.dt)
#endregion

#region Functions
def H(time: int):
    return 1 if time > 0 else 0

def F(x, L):
    return (1 / (np.cosh(x / L))) ** 2

# F = dG/dx
def G(x, L):
    return L * np.tanh(x / L)

def w(meta: Meta, x, z, t, Q, j):
    D = meta.D
    L = meta.L
    N = meta.N
    part1 = 2*F(x, L)
    part2 = -F(x + (N*D*t)/(j*np.pi), L)
    part3 = -F(x - (N*D*t)/(j*np.pi), L)
    return (meta.S0 * Q/(2*(N**2))) * (part1 + part2 + part3)# * np.sin((j*np.pi*z)/D)

def b(meta: Meta, x, z, t, Q, j):
    D = meta.D
    L = meta.L
    N = meta.N
    part1 = G(x + (N*D*t)/(j*np.pi), L)
    part2 = - G(x - (N*D*t)/(j*np.pi), L)
    return ((meta.S0 * Q * j*np.pi) /(2*N*D)) * (part1 + part2)# * np.sin((j*np.pi*z)/D)

def u(meta: Meta, x, z, t, Q, j):
    D = meta.D
    L = meta.L
    N = meta.N
    part1 = G(x + (((N*D*t)/(j*np.pi))), L)
    part2 = G(x - (((N*D*t)/(j*np.pi))), L)
    part3 = -2 * G(x, L)
    return ((meta.S0 * Q * j * np.pi)/(2 * D * N * N)) * (part1 + part2 + part3)# * np.cos((j*np.pi*z)/D)
#endregion

def simulate(meta: Meta, step: Callable[[DataSample, int], DataSample]):
    output = Data(data=np.array([]), meta=meta)

    empty = np.array([np.zeros(meta.x.shape[1], dtype='float32')]*len(meta.js))

    u = np.copy(empty)
    v = np.copy(empty)
    w = np.copy(empty)
    b = np.copy(empty)
    p = np.copy(empty)
    rho = np.copy(empty)

    inp = DataSample(u=u, v=v, w=w, b=b, p=p, rho=rho, t=0)
    inp._2D()
    output.addSample(inp)

    saveEvery = meta.timesteps if meta.saveEvery == 0 else meta.saveEvery
    previousSample = inp

    for t in tqdm.tqdm(range(1, int(meta.timesteps))):
        inp = DataSample(u=np.copy(empty), v=np.copy(empty), w=np.copy(empty), b=np.copy(empty), p=np.copy(empty), rho=np.copy(empty), t=t*meta.dt)
        for i in range(len(meta.js)):
            sample = step(meta=meta, inp=previousSample, i=i)
            inp.u[i] = sample.u
            inp.v[i] = sample.v
            inp.w[i] = sample.w
            inp.b[i] = sample.b
            inp.p[i] = sample.p
            inp.rho[i] = sample.rho
        inp._2D()
        previousSample = inp
        if t % saveEvery == 0:
            output.addSample(inp)

    return output

def evaluate(meta: Meta):
    output = Data(data=np.array([]), meta=meta)

    saveEvery = meta.timesteps if meta.saveEvery == 0 else meta.saveEvery
    empty = np.array([np.zeros(meta.x.shape[1])]*len(meta.js))
    for t in tqdm.tqdm(range(int(meta.timesteps))):
        if t % saveEvery != 0: continue
        
        time = t * meta.dt
        
        inp = DataSample(u=np.copy(empty), v=np.copy(empty), w=np.copy(empty), b=np.copy(empty), p=np.copy(empty), t=time)
        for i in range(len(meta.js)):
            inp.u[i] = u(meta, meta.x[0, :], 0, time, meta.Qs[i], meta.js[i])
            inp.w[i] = w(meta, meta.x[0, :], 0, time, meta.Qs[i], meta.js[i])
            inp.b[i] = b(meta, meta.x[0, :], 0, time, meta.Qs[i], meta.js[i])
            inp.p[i] = -(meta.D/(meta.js[i]*np.pi)) * inp.b[i]

        inp.v = inp.u # so not empty

        inp._2D()
        output.addSample(inp)
   
    return output

def convert(data: Data, contToken: DataSample = None):
    meta = data.meta
    start = 0
    if contToken != None:
        print(f'Continuation token provided')
        start = int(contToken.t / meta.dt) + 1
        contToken = None

    output = Data(np.array([]), meta)

    h = 100000 # scale height
    rho_s = 1
    D = meta.D
    N = 0.01
    A_j = math.sqrt(2 / (rho_s * (N ** 2) * D))

    for i in tqdm.tqdm(range(start, data.data.shape[0])):
        inp = data.data[i]
        
        u = np.zeros(meta.x.shape)
        v = np.zeros(meta.x.shape)
        w = np.zeros(meta.x.shape)
        b = np.zeros(meta.x.shape)
        p = np.zeros(meta.x.shape)
        rho = np.zeros(meta.x.shape)
        t = inp.t

        for i in range(len(meta.js)):
            c_jSquared = (((N * meta.D) ** 2) / (((meta.js[i] * np.pi) ** 2) + ((meta.D ** 2)/(4 * (h ** 2))))) # wavespeed
            for z in range(meta.x.shape[0]):
                # phi like
                w[z, :] += inp.w[i] * phi(z, meta.js[i], h, meta.D, meta.x.shape[0]-1, A_j)
                rho[z, :] += inp.rho[i] * ((rhoZero(z, rho_s, h, meta.D, meta.x.shape[0]-1) * (N ** 2)) / (c_jSquared)) * phi(z, meta.js[i], h, meta.D, meta.x.shape[0]-1, A_j)
                b[z, :] +=  inp.b[i] * ((N ** 2) / c_jSquared) * phi(z, meta.js[i], h, meta.D, meta.x.shape[0]-1, A_j)

                # phi' like
                u[z, :] += inp.u[i] * phiPrime(z, meta.js[i], h, meta.D, meta.x.shape[0]-1, A_j)
                v[z, :] += inp.v[i] * phiPrime(z, meta.js[i], h, meta.D, meta.x.shape[0]-1, A_j)
                p[z, :] += inp.p[i] * rhoZero(z, rho_s, h, meta.D, meta.x.shape[0]-1) *  phiPrime(z, meta.js[i], h, meta.D, meta.x.shape[0]-1, A_j)

                sample = DataSample(u=u, v=v, w=w, b=b, p=p, rho=rho, t=t)
        

        output.addSample(sample)

        # Max RAM handling code
        #memory = round(python_process.memory_info()[0]/2.**30, 2)  # memory use in GB
        
        #if memory != previousMemory:
        #    previousMemory = memory
        #    log.info(f'Memory: {memory}/{meta.maxGB}')

        #if memory >= meta.maxGB:
        #    contToken = inp
        #    print(f'Max memory reached, returning continuation token')
        #    return output, contToken

    return output, None


def phi(z, j, h, D, Dz, A):
    return A * np.exp(z / (2 * Dz * (h/D))) * np.sin((j * np.pi * z) / Dz)

def phiPrime(z, j, h, D, Dz, A):
    return A * np.exp(z / (2 * Dz * (h/D))) * (((1 / (2 * h)) * (np.sin((j * np.pi * z) / Dz))) + (((j * np.pi) / D) * (np.cos((j * np.pi * z) / Dz))))

def rhoZero(z, rho_s, h, D, Dz):
    return rho_s * np.exp(-z / (Dz * (h/D)))