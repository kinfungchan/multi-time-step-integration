import numpy as np
"""
This module is to apply Boundary conditions to the domain
"""

class VelBoundaryConditions:
    def __init__(self, indexes: list, velocities: list):
        self.indexes = indexes
        self.velocities = velocities

    def velbc(t, L, E, rho):
        sinePeriod = (L / 2) * np.sqrt(rho/E)
        freq = 1 / sinePeriod
        if t >= sinePeriod * 0.5:
            return 0
        else:
            return 0.01 * np.sin(2 * np.pi * freq * t)
        
    def velbcHighFreq(t, L, E, rho):
        sinePeriod = (L / 2) * np.sqrt(rho/E)
        highFreqPeriod = (L / 100) * np.sqrt(rho/E)
        freq = 1 / sinePeriod
        highFreq = 1 / highFreqPeriod
        if t >= sinePeriod * 0.5:
            return 0
        else:
            return 0.01 * np.sin(2 * np.pi * freq * t) + 0.001 * np.sin(2 * np.pi * highFreq * t)
        
    def velbcSquareWave(t, L, E, rho):
        sinePeriod = (L / 2) * np.sqrt(rho/E)
        if t >= sinePeriod * 0.5:
            return 0
        else:
            return 0.01

class AccelBoundaryConditions:
    def __init__(self, indexes: list, accelerations: list):
        self.indexes = indexes
        self.accelerations = accelerations