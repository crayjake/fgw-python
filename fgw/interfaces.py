# -- fgw/interfaces.py --
# Author: Jake Cray
# GitHub: crayjake/fgw-python
''' Interface for defining a simulation type '''

# imports
from math import sin, sqrt, pi, ceil
from typing import TypeVar, List

import numpy as np

G = TypeVar('G')

# class defining all the required info for a simulation
class SimulationInterface:

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
        
        initialData:          G,          # initial data
    ):
        # convert width and depth to m
        self.width = 1000 * width
        self.depth = 1000 * depth

        # calculate the space steps
        self.dx = width / horizontalResolution
        self.dz = depth / verticalResolution

        # attach parameters to object
        self.modes = modes
        self.horizontalResolution = horizontalResolution
        self.verticalResolution   = verticalResolution
        self.dt = dt
        self.intialData = initialData

        # calculate the coriolis parameter
        self.f  = (2 * 7.2921 * 1e-5) * sin( latitude )

        # calculate the speed of the fastest moving mode
        self.speedFactor = 0
        if not h == 0:
            self.speedFactor = ( self.depth ** 2 ) / ( ( h * 2000 ) ** 2 )
        self.cMax = sqrt( ( ( N * self.depth ) ** 2 )
                            / ( ( pi ** 2 ) + self.speedFactor ) )

        # calculate the sponge information
        self.spongeWidth    = self.width * spongePercentage
        timeInSponge        = self.spongeWidth / self.cMax
        self.spongeStrength = damping / timeInSponge

        # generate simulation grid
        self.GenerateSimulationGrid()

        # generate matrices
        self.GenerateDifferentiationMatrices()
        self.GenerateCoefficientMatrices()

        pass


    # --- abstract methods ---
    # horizontal form of the heating
    def HeatingHorizontal( self, x ):
        pass
    
    # S_j mode decomposition of heating
    def HeatingDecomposition( self, mode ):
        pass

    # rho_0 (z)
    def InitialDensity( self, z ):
        pass

    # phi (z)
    def VerticalDependence( self, z ):
        pass

    # method to step the simulation forward
    def SimulationStep( self, data: G ) -> G:
        pass

    # converts data to 2D
    def Convert( self, data: List[ G ] ) -> List[ G ]:
        pass

    # --- overridable methods ---
    def GenerateSimulationGrid( self ):
        self.X = np.linspace (
            - self.width / 2,
              self.width / 2,
              self.horizontalResolution,
              endpoint = False
              )
        self.Z = np.linspace ( 
            0,
            self.depth,
            self.verticalResolution,
            endpoint = True
        )
        self.x, self.z = np.meshgrid( self.X, self.Z )

    # generates the central finite difference matrices
    def GenerateDifferentiationMatrices( self ):
        D1 = np.zeros(( self.horizontalResolution, self.horizontalResolution ))
        D2 = np.zeros(( self.horizontalResolution, self.horizontalResolution ))

        cx=0
        while cx < self.horizontalResolution:
            D1[ cx, ( cx+1 ) % self.horizontalResolution ] =  1
            D1[ cx, ( cx-1 ) % self.horizontalResolution ] = -1

            D2[ cx, ( cx-1 ) % self.horizontalResolution ] =  1
            D2[ cx, ( cx )   % self.horizontalResolution ] = -2
            D2[ cx, ( cx+1 ) % self.horizontalResolution ] =  1

            cx = cx + 1

        self.D1 = D1 / (2 * self.dx)
        self.D2 = D2 / (self.dx ** 2)

    def GenerateCoefficientMatrices( self ):
        
        A_Rotation = np.eye( self.horizontalResolution ) * ( ( self.f ** 2 ) * ( self.dt ** 2 ) / 4 )
        self.A = np.array( [
            A_Rotation - ( self.WaveSpeed( mode ) * ( self.dt ** 2 ) * self.D2 / 4 )
            for mode in self.modes
        ] )

        for a in range( len( self.modes ) ):
            for i in range( self.horizontalResolution ):
                x = ( i - ( self.horizontalResolution / 2 ) ) * ( self.width / self.horizontalResolution )

                alpha = self.EvaluateSponge( x )

                self.A[ a ][ i ][ i ] += ( 1 + self.dt * alpha ) ** 2
                self.B[ a ][ i ][ i ] +=   1 + self.dt * alpha
        
        self.Ainv = np.array( [
            np.linalg.inv( A )
            for A in self.A
        ] )


    def Simulate( self ) -> List[ G ]:
        self.data = [ self.initialData ]

        for t in range( ceil( self.time / self.dt ) ):
            self.data.append( self.SimulationStep( self.data[ t ] ) )

        return self.data

    # saves every timestep as an image and saves a gif
    def Animate( self, path: str, gif: bool = True ):
        pass

    # saves a single image
    def Image( self, path: str ):
        pass

    def WaveSpeed( self, mode ):
        return sqrt( ( ( self.N * self.depth ) ** 2 )
                   / ( ( ( mode * pi ) ** 2 ) + self.speedFactor ) )

    def EvaluateSponge( self, x ):
        alpha = 0
        value = abs( x ) - ( self.width / 2 ) - self.spongeWidth
        if value > 0:
            alpha = sin( 0.5 * pi * value / self.spongeWidth ) ** 2

        return self.spongeStrength * alpha