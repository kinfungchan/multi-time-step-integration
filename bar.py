import numpy as np

"""
In this module we define the Bar_1D class which 
contains the properties of the bar
"""
class Bar_1D:
    def __init__(self):
        # Large Domain
        self.E_L = 0.0005 * 10**9 
        self.rho_L = 8000
        self.length_L = 100 * 10**-3 
        self.area_L = 1 
        self.num_elem_L = 1200
        # Small Domain
        self.E_S = 0.02 * 10**9 
        self.rho_S = self.rho_L
        self.length_S = 2 * 100 * 10**-3 
        self.area_S = self.area_L
        self.num_elem_S = 2400
        self.safety_Param = 0.5

