from cmath import pi
import numpy as np
import pickle
from dataclasses import dataclass
import lzma
import json
from math import ceil

@dataclass
class DataSample:
    b:   np.ndarray
    u:   np.ndarray
    w:   np.ndarray
    p:   np.ndarray
    v:   np.ndarray
    rho: np.ndarray
    t:   float = 0

    def __add__(self, other):
        if self.t != other.t:
            return None
        else:
            Tb = self.b + other.b
            Tu = self.u + other.u
            Tw = self.w + other.w
            Tv = self.v + other.v
            Tp = self.p + other.p
            Trho = self.rho + other.rho
            return DataSample(u=Tu, v=Tv, w=Tw, b=Tb, p=Tp, rho=Trho, t=self.t)
    
    def _2D(self):
        self.u = np.atleast_2d(self.u)
        self.v = np.atleast_2d(self.v)
        self.w = np.atleast_2d(self.w)
        self.b = np.atleast_2d(self.b)
        self.p = np.atleast_2d(self.p)
        self.rho = np.atleast_2d(self.rho)
    
    def save(self, path):
        with lzma.open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(path):
        with lzma.open(path, 'rb') as f:
            d = pickle.load(f)
            return d

def serialize(o):
    dictionary = dict(o.__dict__)
    dictionary.pop('x', None)
    dictionary.pop('z', None)
    dictionary.pop('X', None)
    dictionary.pop('Z', None)
    dictionary.pop('D1', None)
    dictionary.pop('D2', None)
    dictionary.pop('A_u', None)
    dictionary.pop('B_u', None)
    dictionary.pop('Ainv_u', None)
    dictionary.pop('A_uNEW', None)
    dictionary.pop('B_uNEW', None)
    dictionary.pop('Ainv_uNEW', None)
    
    return dictionary

@dataclass
class Meta:
    Qs:         np.ndarray
    js:         np.ndarray
    Ts:         np.ndarray = None
 
    time:       int = 15500
    dt:         float = 10
 
    width:      int = 100
    D:          int = 10000
    L:          int = 50000
 
    N:          float = 0.01
     
    f:          int = 0
    n:          int = 0
    alpha:      float = 0
 
    x:          np.ndarray = None
    z:          np.ndarray = None

    maxGB:      int = 16 # 16GB
    saveEvery:  int = 0

    S0:         float = 3.6e-5

    def __post_init__(self):
        print(f'Generating metadata:')
        print(f'Setting up space...')
        self.space = self.width * self.L
        self.dx = 100
        self.dz = 50
        #self.spacesteps = int(self.space / self.dx)
        self.spacesteps = 1500
        self.dx = int(self.space / self.spacesteps)
        self.dz = int(self.D / self.spacesteps)

        self.timesteps = ceil(self.time / self.dt)

        self.X = np.linspace(-self.space / 2, self.space / 2, self.spacesteps, endpoint=True, dtype='float32')
        #self.X = np.linspace(0, self.space, self.spacesteps, endpoint=False, dtype='float64')
        self.Z = np.linspace(0, self.D, self.spacesteps, endpoint=True, dtype='float64')
        self.x, self.z = np.meshgrid(self.X, self.Z)

        if len(self.Qs) != len(self.js):
            self.Qs = np.repeat(self.Qs, len(self.js))
            self.Ts = np.repeat(self.Ts, len(self.js))

        self.generateMatrices()

    def generateMatrices(self):
        # make differentiation matrices D1 and D2
        print(f'Creating matrices...')
        spacesteps = self.spacesteps
        D1 = np.zeros((spacesteps, spacesteps), dtype='float64')
        D2 = np.zeros((spacesteps, spacesteps), dtype='float64')
        eye = np.zeros((spacesteps, spacesteps), dtype='float64')
        cx=0
        while cx < spacesteps:
            D1[cx, (cx+1) % spacesteps] = 1
            D1[cx, (cx-1) % spacesteps] = -1

            D2[cx, (cx-1) % spacesteps] =  1
            D2[cx, (cx) % spacesteps]   = -2
            D2[cx, (cx+1) % spacesteps] =  1

            cx = cx + 1

        eye = np.eye(spacesteps)


        self.D1 = D1 / (2*self.dx)
        #self.D1 = D1 / (self.dx)
        #self.D2 = D2 / (self.dx**2)
        self.D2 = D2 / (self.dx**2)

        self.D1 = self.D1.astype('float64')
        self.D2 = self.D2.astype('float64')

        h = 100000 # m - scale height
        A_bulk = np.array([(((self.f**2) * (self.dt ** 2) / 4) - ((((self.N * self.D) ** 2) / (((j * pi) ** 2) + ((self.D ** 2)/(4 * (h ** 2))))) * (self.dt ** 2) * self.D2 / 4)) for j in self.js])
    
        #self.A_new = np.array([eye + A for A in A_bulk])
        #self.B_new = np.array([eye - A for A in A_bulk])
        #self.Ainv_new = np.array([np.linalg.inv(A) for A in self.A_new], dtype='float64')


        import math
        self.spongeWidth = (self.space / 2) / 5 # 1 / 5 of the width (where width is x > 0)
        self.c_max = math.sqrt(((self.N * self.D) ** 2) / (((pi) ** 2) + ((self.D ** 2)/(4 * (h ** 2)))))
        self.alpha = 4 * (self.c_max / self.spongeWidth) # change 4 or 5

        #self.A_damped = np.array([(eye * ((1 + (self.dt * self.alpha)) ** 2)) + A for A in A_bulk])
        #self.B_damped = np.array([(eye * (1 + (self.dt * self.alpha))) - A for A in A_bulk])
        #self.Ainv_damped = np.array([np.linalg.inv(A) for A in self.A_damped], dtype='float64')
        
        self.A_damped = np.array([np.eye(self.spacesteps)] * len(A_bulk))
        self.B_damped = np.array([np.eye(self.spacesteps)] * len(A_bulk))

        for a_index in range(len(A_bulk)):
            for i in range(self.spacesteps):
                x = i - (self.spacesteps / 2)
                self.A_damped[a_index][i][i] *= (1 + (self.dt * self.alphaX(x, self.spacesteps / 2))) ** 2
                self.B_damped[a_index][i][i] *= (1 + (self.dt * self.alphaX(x, self.spacesteps / 2)))
            
        self.Ainv_damped = np.array([np.linalg.inv(A) for A in self.A_damped], dtype='float64')


    def alphaX(self, x, width):
        if 3 * (width) / 4 < abs(x) < width:
            # then a ramps up
            return 4 * self.alpha * (abs(x) - (3 * (width) / 4)) / (width)
        else:
            return 0



    def __str__(self):
        extra = ""
        if self.f != 0 or self.n != 0 or self.alpha != 0:
            extra = f'_f{self.f}_n{self.n}_alpha{self.alpha}'
        return f'j{"-".join(map(str, self.js))}_D{self.D}_L{self.L}_Q{("_Q_".join(map(str,self.Qs))).replace(".","-")}_N{str(self.N).replace(".","-")}_T{("_T_".join(map(str, self.Ts))).replace(".","-")}_T{self.time}_dt{str(self.dt).replace(".","-")}{extra}'

    def title(self, t):
        return f'Plot for t={int(t/60)}mins, \n N={self.N}, depth={self.D/1000}km, width={self.width}*{self.L/1000}km, \n modes=[{", ".join  (map(str, self.js))}], Qs=[{(", ".join(map(str,self.Qs)))}]'
    
    def json(self, path):        
        with open(f'{path}/meta.json', 'w', encoding='utf-8') as f:
            json.dump(serialize(self), f, default=lambda o: o.tolist(), indent=4)

    def load(path):
        meta = Meta()
        with open(f'{path}', 'w', encoding='utf-8') as f:
            meta.__dict__ = json.loads(f)
        return meta

@dataclass
class Data:
    data: np.ndarray
    meta: Meta

    def addSample(self, sample: DataSample):
        self.data = np.append(self.data, [sample], 0)

    def save(self, path, saveMeta:bool = False):
        dataToSave = Data(meta=self.meta if saveMeta else None, data=self.data)
        with lzma.open(f'{path}/{"META" if saveMeta else ""}data.pickle.xz', 'wb') as f:
            pickle.dump(dataToSave, f)

    def load(path):
        with lzma.open(path, 'rb') as f:
            d = pickle.load(f)
            return d