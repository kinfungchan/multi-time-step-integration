import numpy as np

"""
In this module we define the Bar_1D class which 
contains the properties of the bar
"""
class Bar_1D:
    def __init__(self):
        # Large Domain
        self.E_L = 0.02 * 10**9 # 0.02GPa
        self.rho_L = 8000
        self.length_L = 50 * 10**-3 # 50mm
        self.area_L = 1 # 1m^2
        self.num_elem_L = 300
        # Small Domain
        self.E_S = 200.0 * 10**9 # 200GPa High Heterogeneity
        self.rho_S = self.rho_L
        self.length_S = 2 * 50 * 10**-3 
        self.area_S = self.area_L
        self.num_elem_S = 600
        # Safety Parameter
        self.safety_Param = 0.5

class Bar_1D_HighHet:
    def __init__(self):
        # Large Domain
        self.E_L = 0.02 * 10**9 # 0.02GPa
        self.rho_L = 8000
        self.length_L = (50 * 10**-3) - (50 * 10**-3) / 300 
        self.area_L = 1 # 1m^2
        self.num_elem_L = 299
        # Small Domain
        self.E_S = 200.0 * 10**9 # 200GPa High Heterogeneity
        self.rho_S = self.rho_L
        self.length_S = (2 * 50 * 10**-3) + (50 * 10**-3) / 300
        self.area_S = self.area_L
        self.num_elem_S = 601
        # Safety Parameter
        self.safety_Param = 0.5

class Bar_1D_HighHetUnstable:
    '''
    Highly Heterogeneous Bar with Unstable Solution with Proposed Method
    '''
    def __init__(self):
        # Large Domain
        self.E_L = 0.02 * 10**9 # 0.02GPa
        self.rho_L = 8000
        self.length_L = 50 * 10**-3 # 50mm
        self.area_L = 1 # 1m^2
        self.num_elem_L = 300
        # Small Domain
        self.E_S = 200 * 10**9 # 200GPa High Heterogeneity
        self.rho_S = self.rho_L
        self.length_S = 2 * 50 * 10**-3 
        self.area_S = self.area_L
        self.num_elem_S = 600
        # Safety Parameter
        self.safety_Param = 0.5