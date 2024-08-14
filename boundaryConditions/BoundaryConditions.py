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
        
    def velbcSquare(t, L, E, rho):
        sinePeriod = (L / 2) * np.sqrt(rho/E)
        if t >= sinePeriod * 0.5:
            return 0
        else:
            return 0.01
        
    def velbcGaussWP(t, L, E, rho, sigma, n_peaks=10):
        sinePeriod = (L / 2) * np.sqrt(rho/E)
        # freq = 1 / sinePeriod

        mu = sinePeriod / 2  # Centre of the Gaussian
        A = 0.01  # amplitude
        oscillation = np.cos(2 * np.pi * n_peaks * (t - mu) / sinePeriod)
        gaussian_pulse = A * np.exp(-(t - mu)**2 / (2 * sigma**2)) * oscillation

        if t >= mu + 3 * sigma: # 3 sigma is the approximate width of the Gaussian
            return 0
        else:
            return gaussian_pulse

class AccelBoundaryConditions:
    def __init__(self, indexes: list, accelerations: list):
        self.indexes = indexes
        self.accelerations = accelerations