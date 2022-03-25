# -- library/testing.py --
# Author: Jake Cray
# GitHub: crayjake/fgw-python
''' example usage '''

# imports
from structures import *
from generate import generate
from schemes import CrankNicolsonDeep
from converters import converter
from animate import display

import numpy as np

# define the metadata -> sets up our environment
'''
this creates A, B, D1, D2 matrices - see structures.py/Meta.__post_init__()
'''
meta = Meta(
    js        = np.array([2]), # list of modes to generate, only using mode 2 currently
    width     = 540,           # 540 km  ->  (one side) ~ 270km/150ms-1 = 1800s = 30min
    depth     = 50,            # 50 km
    h         = 100,           # scale height/depth is 100 km
    time      = 60*60*2,       # 2 hrs
    T         = 60*10,         # 10 mins
    sponge    = 1 / 2,         # fraction of width to use as sponge layer
    damping   = 4,             # sponge damping strength
    dt        = 10,            # timestep is 10 secs
    saveEvery = 2,             # only save every 2 States
    spacesteps= 1000           # spatial resolution
)

# generate the data using the CrankNicolsonDeep step
'''
this uses the CrankNicolsonDeep - see schemes.py for the implementation of the CN scheme
'''
data = generate(meta=meta, step=CrankNicolsonDeep)

# this just displays a given timestep
time = 100
display(data[time], meta, converter, sponge=False)