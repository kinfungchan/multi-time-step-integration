import numpy as np
from SingleDomain import Domain
from Cho_MTS import Visualise_MTS
from BoundaryConditions import  VelBoundaryConditions as vbc
import imageio
import os

"""
In this notebook we look to reimplement CDM Multistep Time Integration
for One-Dimensional Heterogeneous Solids from the following paper


Reference: Cho, S. S., Kolman, R., González, J. A., & Park, K. C. (2019).
Explicit multistep time integration for discontinuous elastic stress wave
propagation in heterogeneous solids. International Journal for Numerical
Methods in Engineering, 118(5), 276-302

This script attends to the Numerical Example in Section 4.1 of the paper

"""


def solve_Interface_EOM(BMB_L, BMB_S, L_L, L_S, Bat_L, Bat_S):
    """
    Solve the Interface Equations of Motion
    """
    # Set up 3x3 Matrix
    iEOM = np.empty((3, 3))
    iEOM[0, 0] = BMB_L; iEOM[0, 1] = 0; iEOM[0, 2] = L_L
    iEOM[1, 0] = 0; iEOM[1, 1] = BMB_S; iEOM[1, 2] = L_S
    iEOM[2, 0] = L_L; iEOM[2, 1] = L_S; iEOM[2, 2] = 0

    # Set up 3x1 Vector for Unconstrained Accelerations
    a = np.empty(3)
    a[0] = Bat_L
    a[1] = Bat_S
    a[2] = 0

    # Solve for Lagrange Multipliers and Frame Acceleration
    x = np.linalg.solve(iEOM, a)
    Lambda_L = x[0]
    Lambda_S = x[1]
    a_f = x[2]

    # Return Lagrange Multipliers and Frame Acceleration
    return Lambda_L, Lambda_S, a_f

class Multistep:
    """
    Constructor for the One Dimensional Domain class

    :param Large: Large Domain
    :param Small: Small Domain

    """
    def __init__(self, L_Domain: Domain, S_Domain: Domain, m):

        self.Large = L_Domain
        self.Small = S_Domain

        self.m = m # Number of Small Steps

        # Large Domain Interface
        self.B_L = np.zeros(self.Large.n_nodes) # Boolean Vectors for Extracting Interface DOFs for each domain
        self.L_L = 1 # Boolean Vectors for Extracting Interface DOFs for global acc and disp
        self.B_L[-1] = 1

        # Small Domain Interface
        self.B_S = np.zeros(self.Small.n_nodes) # Boolean Vectors for Extracting Interface DOFs for each domain
        self.L_S = 1 # Boolean Vectors for Extracting Interface DOFs for global acc and disp
        self.B_S[0] = 1

        # Frame
        self.a_f = np.zeros(1) 
        self.v_f = np.zeros(1)
        self.u_f = np.zeros(1)

    def multistep_CDM(self):
        """
        Integrate the Domain using the PFPB Scheme
        """
        invM_L = np.linalg.inv(self.Large.M)
        invM_S = np.linalg.inv(self.Small.M)
        BMB_S = np.dot(np.transpose(self.B_S), np.dot(invM_S, self.B_S)) # B_S^T * M_S^-1 * B_S
        BMB_L = np.dot(np.transpose(self.B_L), np.dot(invM_L, self.B_L)) # B_S^T * M_S^-1 * B_S

        k = 0

        while (k < self.m):
            '''
            Update of Small Domain
            '''
            self.Small.element_update()

            if (self.Small.t == 0):
                self.Small.a = np.linalg.solve(self.Small.M, self.Small.f_ext - np.dot(self.Small.K, self.Small.u))

            # Compute Lagrange Multipliers and Frame Acceleration
            ut_njS_S = self.Small.u + self.Small.dt * self.Small.v + self.Small.a * (0.5 - self.Small.beta) * self.Small.dt**2
            at_njS_S = np.linalg.solve(self.Small.M, self.Small.f_ext - np.dot(self.Small.K, ut_njS_S))            

            # Extract Last 3x3 in Large M, Large K and Last 3x1 in Large f_ext
            E1E2_LargeM = self.Large.M[-3:, -3:]
            E1E2_Largef_ext = self.Large.f_ext[-3:]
            E1E2_LargeK = self.Large.K[-3:, -3:]
            E1E2_ut_njS_L = self.Large.u[-3:] + self.Large.dt * self.Large.v[-3:] + self.Large.a[-3:] * (0.5 - self.Large.beta) * self.Large.dt**2
            E1E2_at_njS_L = np.linalg.solve(E1E2_LargeM, E1E2_Largef_ext - np.dot(E1E2_LargeK, E1E2_ut_njS_L)) 
            E1E2_B_L = self.B_L[-3:]
            
            Bat_njS_S = np.dot(np.transpose(self.B_S), at_njS_S)
            Bat_njS_L = np.dot(np.transpose(E1E2_B_L), E1E2_at_njS_L)

            Lambda_njS_L, Lambda_njS_S, a_njS_f = solve_Interface_EOM(BMB_L, BMB_S, self.L_L, self.L_S, Bat_njS_L , Bat_njS_S)    

            if (self.Small.t == 0):
                self.Small.a = np.linalg.solve(self.Small.M, self.Small.f_ext - np.dot(self.Small.K, self.Small.u)) 

            # Calculation of Predictors
            u_k1 = ut_njS_S
            v_k1 = self.Small.v + self.Small.a * (1 - self.Small.gamma) * self.Small.dt
        
            # Solution of Linear Problem
            # Explicit Method
            if (self.Small.beta == 0.0):
                a_k1 = at_njS_S - np.dot(invM_S, (self.B_S * Lambda_njS_S))   

            # Calculation of Correctors
            u_k1 = u_k1 
            v_k1 = v_k1 + a_k1 * self.Small.gamma * self.Small.dt

            # Update State Variables        
            self.Small.u = u_k1
            self.Small.v = v_k1
            self.Small.a = a_k1
            self.Small.t = self.Small.t + self.Small.dt
            self.Small.n += 1

            # Update Frame
            self.a_f = a_njS_f
            self.v_f = self.Small.v[0]
            self.u_f = self.Small.u[0]

            # Update Small Loop Counter
            k += 1
            
        '''
        Update of Large Domain
        '''
        self.Large.element_update()
        
        if (self.Large.t == 0):
            self.Large.a = np.linalg.solve(self.Large.M, self.Large.f_ext - np.dot(self.Large.K, self.Large.u))
            self.Large.assemble_vbcs(self.Large.t)
        
        self.Large.a[0] = 0.0
        u_k1 = self.Large.u + self.Large.dt * self.Large.v + self.Large.a * (0.5 - self.Large.beta) * self.Large.dt**2
        v_k1 = self.Large.v + self.Large.a * (1 - self.Large.gamma) * self.Large.dt
       
        # Solution of Linear Problem
        # Explicit Method
        if (self.Large.beta == 0.0):
            a_k1 = np.linalg.solve(self.Large.M, self.Large.f_ext - np.dot(self.Large.K, u_k1))
            a_k1[0] = 0.0
            a_k1 -= np.dot(invM_L, (self.B_L * Lambda_njS_L))            

        u_k1 = u_k1
        v_k1 = v_k1 + a_k1 * self.Large.gamma * self.Large.dt

        # Update State Variables        
        self.Large.u = u_k1
        self.Large.v = v_k1
        self.Large.assemble_vbcs(self.Large.t)
        self.Large.a = a_k1

        self.Large.u[-1] = self.u_f
        self.Large.v[-1] = self.v_f

        self.Large.t = self.Large.t + self.Large.dt
        self.Large.n += 1

        # Check for Time Equivalence
        if abs(self.Large.t - self.Small.t) > 1e-10:
            print("Time Discrepancy")
            print("Large Time:", full_Domain.Large.t, "Small Time:", full_Domain.Small.t)
            exit()


if __name__ == '__main__':
    # Initialise Domains
    # Large Domain
    E_L = 0.02 * 10**9 # 0.02GPa
    E_s = 0.18 * 10**9 # Integer Time Step Ratio = 3
    rho_L = 8000 # 8000kg/m^3
    length_L = 50 * 10**-3 # 50mm
    length_S = 2 * 50 * 10**-3 # 100mm
    area_L = 1 # 1m^2
    num_elements_L = 300
    num_elements_S = 600
    safety_Param = 0.5
    def vel(t): return vbc.velbcSquare(t, 2 * length_L, E_L, rho_L)
    velboundaryConditions = vbc(list([0]), list([vel]))
    # Large Domain
    Domain_L = Domain('Large', E_L, rho_L, length_L, area_L, num_elements_L, safety_Param, velboundaryConditions)
    Domain_L.compute_mass_matrix()
    Domain_L.compute_stiffness_matrix()
    # Small Domain
    Domain_S = Domain('Small', E_s, rho_L, length_S, area_L, num_elements_S, safety_Param, None)
    Domain_S.compute_mass_matrix()
    Domain_S.compute_stiffness_matrix()
    # Multistep Combined Domains
    full_Domain = Multistep(Domain_L, Domain_S, 3)

    # Visualisation
    bar = Visualise_MTS(full_Domain)

    # Integrate over time
    while Domain_L.t < 0.0015:
        full_Domain.multistep_CDM()
        print("Time: ", Domain_L.t)
        if Domain_L.n % 10 == 0: 
            bar.plot_accel()
            bar.plot_vel()
            bar.plot_disp()
            bar.plot_stress()

    bar.create_gif('FEM1DAccel.gif', bar.filenames_accel)
    bar.create_gif('FEM1DVel.gif', bar.filenames_vel)
    bar.create_gif('FEM1DDisp.gif', bar.filenames_disp)
    bar.create_gif('FEM1DStress.gif', bar.filenames_stress)

