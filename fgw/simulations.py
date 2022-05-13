# -- fgw/simulations.py --
# Author: Jake Cray
# GitHub: crayjake/fgw-python
'''  '''

# imports
from fgw.structures import DataStruct
import numpy as np

from typing      import TypeVar
from math        import sqrt, sin, pi

from .interfaces import SimulationInterface
from .schemes    import CrankNicolson

# system without Boussinesq approximation -> deep atmosphere
class Deep(SimulationInterface):
    def __init__(
        self,
        width:                int,        #                              ( km )
        depth:                int,        #                              ( km )
        horizontalResolution: int,        # number of grid points
        verticalResolution:   int,        # number of grid points

        time:                 float,      # time to simulate             ( s )
        dt:                   float,      # timesteps
        
        latitude:             float,      # angle from equator, used in calculating coriolis

        spongePercentage:     float,      # percentage of width to damp
        damping:              float,      # damping strength

        S0:                   float,      # maximum forcing (heating)
        N:                    float,      # buoyancy frequency           ( /s )
        h:                    float,      # scale height                 ( km )

        modes:                np.ndarray, # list of modes
            
        # specific variables
        heatingScaleWidth:    float,      # scale width of heating
    
        initialData:          DataStruct = None,          # initial data
    ):
        super().__init__(
            width,
            depth,
            horizontalResolution,
            verticalResolution,
            time,
            dt,
            latitude,
            spongePercentage,
            damping,
            S0,
            N,
            h,
            modes,
            initialData,
        )
        self.heatingScaleWidth = heatingScaleWidth

    # --- abstract methods ---
    # horizontal form of the heating
    def HeatingHorizontal( self, x ):
        return ( 1 / ( np.cosh( x / self.heatingScaleWidth * 1000 ) ) ) ** 2
    
    # S_j mode decomposition of heating
    def HeatingDecomposition( self, mode ):
        D_t   = 10000
        rho_s = 1
        A   = sqrt( 2 / ( rho_s * ( self.N ** 2 ) * self.depth ) )

        if ( ( mode * D_t / self.depth ) - 1 ) == 0:
            S = A * ( rho_s / 2 ) * D_t
        else:
            S_A = sin( pi * ( ( mode * D_t / self.depth ) - 1) ) / ( ( mode * D_t / self.depth ) - 1 )
            S_B = sin( pi * ( ( mode * D_t / self.depth ) + 1) ) / ( ( mode * D_t / self.depth ) + 1)
            S = A * ( rho_s / 2 ) * ( D_t / np.pi ) * ( S_A - S_B )

        return S

    # rho_0 (z)
    def InitialDensity( self, z ):
        rho_s = 1
        return rho_s * np.exp( -z / self.h )

    # phi (z)
    def VerticalDependence( self, z ):
        pass

    # set the simulation step as the default CN scheme
    def SimulationStep( self, data: DataStruct ) -> DataStruct:
        return CrankNicolson( self.dt, data )
    
    # converts data to 2D
    def Convert( self, data: list ) -> list:
        pass
