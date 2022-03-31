# -- library/structures.py --
# Author: Jake Cray
# GitHub: crayjake/fgw-python
''' contains all the data structures '''

from xmlrpc.client import Boolean
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
        
    spacesteps:     int = 750       # spatial resolution
    
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

    # generate all the matrices or just use metadata for visualising already simulated data
    generateData: Boolean = True

    # run after dataclass init
    def __post_init__(self):
        print(f'Starting metadata generation')
        #self.dx = 300
        #self.dz = 200
        self.dx = int(1000 * self.width / self.spacesteps)
        self.dz = int(1000 * self.depth / self.spacesteps)

        self.h_spacesteps = int(1000 * self.width / self.dx)
        self.v_spacesteps = int(1000 * self.depth / self.dz)

        self.timesteps = ceil(self.time / self.dt)

        print(f'Setting up the space')
        self.X = 1000 * np.linspace(- self.width / 2, self.width / 2, self.h_spacesteps, endpoint = False)
        self.Z = 1000 * np.linspace(0, self.depth, self.v_spacesteps, endpoint = True)
        self.x, self.z = np.meshgrid(self.X, self.Z)
        
        self.W = self.width * 1000
        self.D = self.depth * 1000

        print(f'Generating finite difference matrices')
        #spacesteps = self.spacesteps
        h_spacesteps = self.h_spacesteps
        D1 = np.zeros((h_spacesteps, h_spacesteps), dtype='float64')
        D2 = np.zeros((h_spacesteps, h_spacesteps), dtype='float64')

        cx=0
        while cx < h_spacesteps:
            D1[cx, (cx+1) % h_spacesteps] =  1
            D1[cx, (cx-1) % h_spacesteps] = -1

            D2[cx, (cx-1) % h_spacesteps] =  1
            D2[cx, (cx)   % h_spacesteps] = -2
            D2[cx, (cx+1) % h_spacesteps] =  1

            cx = cx + 1

        self.D1 = D1 / (2 * self.dx)
        self.D2 = D2 / (self.dx ** 2)

        # get width and depth in m
        W = self.W
        D = self.D

        # check if shallow or deep atmosphere
        if self.h == 0:
            print(f'Shallow atmosphere!')
            self.c_max = (self.N * D) / pi
            
            self.c_squared = lambda j : (((self.N * D) ** 2) / (((j * pi) ** 2)))

        else:
            print(f'Deep atmosphere!')
            self.c_max = sqrt(((self.N * D) ** 2) / (((pi) ** 2) + ((D ** 2)/(4 * ((self.h * 1000) ** 2)))))

            self.c_squared = lambda j : (((self.N * D) ** 2) / (((j * pi) ** 2) + ((D ** 2)/(4 * ((self.h * 1000) ** 2)))))

 
        # check if using a sponge layer
        if (self.sponge == 0) and (self.damping == 0):
            print(f'Not using a sponge layer')
            spongeWidth = 0
            spongeStrength = 0
            

        else: 
            print(f'Setting up sponge layer')
            spongeWidth = (W / 2) * self.sponge
            timeInSponge = spongeWidth / self.c_max
            theoryTimeInSponge = timeInSponge / self.dt
            spongeStrength = self.damping / timeInSponge

        self.spongeWidth    = spongeWidth
        self.spongeStrength = spongeStrength


        if self.generateData:
            self.GenerateData()

    # NOTE: if changing to/from sponge/deep then must regenerate matrices
    def GenerateData(self):
        print(f'Generating coefficient matrices')
        A_rotation = np.eye(self.h_spacesteps) * ((self.f**2) * (self.dt ** 2) / 4)
        self.A_bulk = lambda j: (A_rotation - (self.c_squared(j) * (self.dt ** 2) * self.D2 / 4))

        self.A_val = np.zeros((self.h_spacesteps, self.h_spacesteps))
        self.B_val = np.zeros((self.h_spacesteps, self.h_spacesteps))
        
        for i in range(self.h_spacesteps):
            x = (i - (self.h_spacesteps / 2)) * (self.W / self.h_spacesteps)
            alpha = self.spongeAlpha(x)
            
            self.A_val[i][i] = (1 + self.dt * alpha) ** 2
            self.B_val[i][i] = (1 + self.dt * alpha) ** 1

        self.A = lambda j: self.A_val + self.A_bulk(j)
        self.B = lambda j: self.B_val - self.A_bulk(j)


        self.Ainv = lambda j: np.linalg.inv(self.A(j))


    def A(self, j):
        A = np.zeros(self.h_spacesteps, self.h_spacesteps)
        for i in range(self.h_spacesteps):
            x = (i - (self.h_spacesteps / 2)) * (self.W / self.h_spacesteps)
            alpha = self.spongeAlpha(x)
            self.A[i][i] = (1 + self.dt * alpha) ** 2



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