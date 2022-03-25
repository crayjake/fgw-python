# -- library/structures.py --
# Author: Jake Cray
# GitHub: crayjake/fgw-python
''' contains all the data structures '''

import numpy as np

from dataclasses import dataclass
from math import ceil, sqrt, pi
from typing import Callable

from helpers import F, G

# structure for the state of a model at a specific time
@dataclass
class State:
    u:   np.ndarray # horizontal velocity
    v:   np.ndarray # 'y' component of velocity
    w:   np.ndarray # vertical velocity
    p:   np.ndarray # pressure
    b:   np.ndarray # buoyancy
    rho: np.ndarray # density
    t:   int        # time

    def _2D(self):
        self.u   = np.atleast_2d(self.u)
        self.v   = np.atleast_2d(self.v)
        self.w   = np.atleast_2d(self.w)
        self.p   = np.atleast_2d(self.p)
        self.b   = np.atleast_2d(self.b)
        self.rho = np.atleast_2d(self.rho)


# structure for the metadata of the system
@dataclass
class Meta:
    js:             np.ndarray       # modes
    
    width:          int              # width of system                - km
    depth:          int              # depth of system                - km
    
    time:           int              # simulation time                - s
    T:              int = 1e10       # time of pulsed heating         - s
    dt:             float = 10       # time step for the simulation   - s
        
    spacesteps:     int = 1000       # spatial resolution
    
    D_t:            int = 10         # height of troposphere if deep  - km
    L:              int = 10         # horizontal scale of heating    - km
    N:              float = 0.01     # buoyancy frequency             - /s
    f:              int = 0          # coriolis parameter
    h:              int = 0        # scale height                   - km
    S0:             float = 3.6e-5   # maximum forcing
       
    sponge:         int = 0          # fraction of width to sponge
    damping:        int = 0          # strength of damping
    
    saveEvery:      int = 0          # save every 'n' samples

    heatingForm:     Callable[[np.ndarray, int], float] = F

    # run after dataclass init
    def __post_init__(self):
        print(f'Starting metadata generation')
        self.dx = int(1000 * self.width / self.spacesteps)
        self.dz = int(1000 * self.depth / self.spacesteps)

        self.timesteps = ceil(self.time / self.dt)

        print(f'Setting up the space')
        self.X = 1000 * np.linspace(- self.width / 2, self.width / 2, self.spacesteps, endpoint = False)
        self.Z = 1000 * np.linspace(0, self.depth, self.spacesteps, endpoint = True)
        self.x, self.z = np.meshgrid(self.X, self.Z)
        
        self.GenerateMatrices()

    # NOTE: if changing to/from sponge/deep then must regenerate matrices
    def GenerateMatrices(self):
        print(f'Generating finite difference matrices')
        spacesteps = self.spacesteps
        D1 = np.zeros((spacesteps, spacesteps), dtype='float64')
        D2 = np.zeros((spacesteps, spacesteps), dtype='float64')

        cx=0
        while cx < spacesteps:
            D1[cx, (cx+1) % spacesteps] =  1
            D1[cx, (cx-1) % spacesteps] = -1

            D2[cx, (cx-1) % spacesteps] =  1
            D2[cx, (cx)   % spacesteps] = -2
            D2[cx, (cx+1) % spacesteps] =  1

            cx = cx + 1

        self.D1 = D1 / (2 * self.dx)
        self.D2 = D2 / (self.dx ** 2)

        print(f'Generating Crank-Nicolson matrices')
        # get width and depth in m
        W = self.width * 1000
        D = self.depth * 1000
        self.W = W
        self.D = D

        # check if shallow or deep atmosphere
        if self.h == 0:
            print(f'Shallow atmosphere!')
            self.c_max = (self.N * D) / pi
            
            c_squared = lambda j : (((self.N * D) ** 2) / (((j * pi) ** 2)))

        else:
            print(f'Deep atmosphere!')
            self.c_max = sqrt(((self.N * D) ** 2) / (((pi) ** 2) + ((D ** 2)/(4 * ((self.h * 1000) ** 2)))))

            c_squared = lambda j : (((self.N * D) ** 2) / (((j * pi) ** 2) + ((D ** 2)/(4 * ((self.h * 1000) ** 2)))))

        A_bulk = np.array([(((self.f**2) * (self.dt ** 2) / 4) - (c_squared(j) * (self.dt ** 2) * self.D2 / 4)) for j in self.js])


        # check if using a sponge layer
        if (self.sponge == 0) and (self.damping == 0):
            print(f'Not using a sponge layer')
            spongeWidth = 0
            spongeStrength = 0
            

        else: 
            print(f'Using a sponge layer')
            spongeWidth = (W / 2) * self.sponge
            spongeStrength = self.damping * (self.c_max / spongeWidth)
            print(f'DEBUG: spongeStrength: {spongeStrength}')
            print(f'DEBUG: spongeWidth:    {spongeWidth}')
            print(f'DEBUG: c_max:          {self.c_max}')

        self.spongeWidth    = spongeWidth
        self.spongeStrength = spongeStrength

        self.A = np.array([np.eye(self.spacesteps)] * len(A_bulk), dtype='float64')
        self.B = np.array([np.eye(self.spacesteps)] * len(A_bulk), dtype='float64')

        for a in range(len(A_bulk)):
            alphas = np.array([])
            for i in range(self.spacesteps):
                x = (i - (self.spacesteps / 2)) * (W / self.spacesteps)

                alpha = self.spongeAlpha(x)
                alphas = np.append(alphas, alpha)


                #self.A[a][i][i] *= (1 + self.dt * alpha) ** 2
                #self.B[a][i][i] *= (1 +self.dt * alpha)
                self.A[a][i][i] = (1 + self.dt * alpha) ** 2
                self.B[a][i][i] = (1 + self.dt * alpha) ** 1

            self.A[a] += A_bulk[a]
            self.B[a] -= A_bulk[a]

            print(f'Alpha: {np.min(alphas)}, {np.max(alphas)/spongeStrength}')
        self.Ainv = np.array([np.linalg.inv(A) for A in self.A], dtype='float64')


    # NOTE: if changing to/from sponge/deep then must regenerate matrices
    def GenerateMatricesTest(self):
        print(f'Generating finite difference matrices')
        spacesteps = self.spacesteps
        D1 = np.zeros((spacesteps, spacesteps), dtype='float64')
        D2 = np.zeros((spacesteps, spacesteps), dtype='float64')

        cx=0
        while cx < spacesteps:
            D1[cx, (cx+1) % spacesteps] =  1
            D1[cx, (cx-1) % spacesteps] = -1

            D2[cx, (cx-1) % spacesteps] =  1
            D2[cx, (cx)   % spacesteps] = -2
            D2[cx, (cx+1) % spacesteps] =  1

            cx = cx + 1

        self.D1 = D1 / (2 * self.dx)
        self.D2 = D2 / (self.dx ** 2)

        print(f'Generating Crank-Nicolson matrices')
        # get width and depth in m
        W = self.width * 1000
        D = self.depth * 1000
        self.W = W
        self.D = D

        self.c_max = 1
        A_bulk = np.array([((self.dt ** 2) * self.D2 / 4) for j in self.js])


        # check if using a sponge layer
        if (self.sponge == 0) and (self.damping == 0):
            print(f'Not using a sponge layer')
            spongeWidth = 0
            spongeStrength = 0
            

        else: 
            print(f'Using a sponge layer')
            spongeWidth = (W / 2) * self.sponge
            spongeStrength = self.damping  * (self.c_max / spongeWidth)
            print(f'DEBUG: spongeStrength: {spongeStrength}')
            print(f'DEBUG: spongeWidth:    {spongeWidth}')
            print(f'DEBUG: c_max:          {self.c_max}')

        self.spongeWidth    = spongeWidth
        self.spongeStrength = spongeStrength

        self.A = np.array([np.eye(self.spacesteps)] * len(A_bulk), dtype='float64')
        self.B = np.array([np.eye(self.spacesteps)] * len(A_bulk), dtype='float64')

        for a in range(len(A_bulk)):
            alphas = np.array([])
            for i in range(self.spacesteps):
                x = (i - (self.spacesteps / 2)) * (W / self.spacesteps)

                alpha = self.spongeAlpha(x)
                alphas = np.append(alphas, alpha)

                self.A[a][i][i] = (1 + (self.dt * alpha)) ** 2
                self.B[a][i][i] = (1 + (self.dt * alpha))

            self.A[a] += A_bulk[a]
            self.B[a] -= A_bulk[a]

            print(f'Alpha: {np.min(alphas)}, {np.max(alphas)/spongeStrength}')
        self.Ainv = np.array([np.linalg.inv(A) for A in self.A], dtype='float64')


    def spongeAlphaVectorized(self, xs):
        x = np.copy(xs)
        for i in range(len(x)):
            x[i] = self.spongeAlpha(x[i])

        return x

    def spongeAlpha(self, x):
        al = 0
        if ((self.W / 2) * (1 - self.sponge)) < abs(x):
            # then we are within the sponge layer
            # al = (2 / (self.W * self.sponge)) * (abs(x) - ((self.W / 2) * (1 - self.sponge)))
            val = (abs(x) - ((self.W / 2) * (1 - self.sponge)))
            al = np.sin(0.5 * np.pi * val / ((self.W / 2) * self.sponge)) ** 2
                
        return self.spongeStrength * (al)