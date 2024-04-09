import numpy as np
from SingleDomain import Domain
from Cho_MTS import Visualise_MTS
from BoundaryConditions import  VelBoundaryConditions as vbc

"""
In this notebook we look to reimplement Asynchronous Direct Time
Integration for One-Dimensional Heterogeneous Solids from the following paper


Reference: Dvořák, R., Kolman, R., Mračko, M., Kopačka, J., Fíla, T., Jiroušek, 
O., Falta, J., Neuhäuserová, M., Rada, V., Adámek, V. and González, J.A., 2023. 
Energy-conserving interface dynamics with asynchronous direct time integration 
employing arbitrary time steps. Computer Methods in Applied Mechanics and 
Engineering, 413, p.116110.

Dvorák, Radim, et al. "ASYNCHRONOUSLY IN TIME INTEGRATED INTERFACE DYNAMICS 
PROBLEM WHILE MAINTAINING ZERO INTERFACE ENERGY.", pp9-10 2023.

"""

class Multistep:
    """
    Constructor for the One Dimensional Domain class

    :param Large: Large Domain
    :param Small: Small Domain

    """
    def __init__(self, L_Domain: Domain, S_Domain: Domain, m):

        self.Large = L_Domain
        self.Small = S_Domain
        self.invM_L = np.linalg.inv(self.Large.M)
        self.invM_S = np.linalg.inv(self.Small.M)

        self.m = m # Number of Small Steps

        # Large Domain Interface
        self.B_L = np.zeros(self.Large.n_nodes) # Boolean Vectors for Extracting Interface DOFs for each domain
        self.L_L = 1 # Boolean Vectors for Extracting Interface DOFs for global acc and disp
        self.B_L[-1] = 1

        # Small Domain Interface
        self.B_S = np.zeros(self.Small.n_nodes) # Boolean Vectors for Extracting Interface DOFs for each domain
        self.L_S = 1 # Boolean Vectors for Extracting Interface DOFs for global acc and disp
        self.B_S[0] = 1

        self.BMB_S = np.dot(np.transpose(self.B_S), np.dot(self.invM_S, self.B_S)) # B_S^T * M_S^-1 * B_S
        self.BMB_L = np.dot(np.transpose(self.B_L), np.dot(self.invM_L, self.B_L)) # B_S^T * M_S^-1 * B_S
        self.M_f = (self.L_S * (1 / self.BMB_S) *  self.L_S) + (self.L_L * (1 / self.BMB_L) *  self.L_S)
        self.invM_f = 1 / self.M_f

        # Region
        self.Lambda_n1r_L = 0.0
        self.a_r_L = np.zeros(3)
        self.v_r_L = np.zeros(3)
        self.u_r_L = np.zeros(3)
        self.t_r_L = 0.0

        self.Lambda_n1r_S = 0.0
        self.a_r_S = np.zeros(3)
        self.v_r_S = np.zeros(3)
        self.u_r_S = np.zeros(3)
        self.t_r_S = 0.0

        # Frames
        self.a_f = np.zeros(1) 
        self.v_f = np.zeros(1)
        self.u_f = np.zeros(1)
        self.t_f = 0.0
        self.n_f = 0

        # Clocks
        self.t_s_act = 0.0 # Last known solution Time
        self.t_s_new = self.t_s_act + self.Small.dt # Time of interest
        self.t_L_act = 0.0 # Last known solution Time
        self.t_L_new = self.t_L_act + self.Large.dt # Time of interest
        self.dt_f = 0.0 # Interface time step

    def solve_subframes(self):
        self.dt_f = min(self.t_s_new - self.t_f, self.t_L_new - self.t_f) # 2.2 

        # Extract Last 3x3 in Large M, Large K and Last 3x1 in Large f_ext
        M_S_r = self.Small.M[:3, :3]
        f_ext_S_r = self.Small.f_ext[:3]
        K_S_r = self.Small.K[:3, :3]
        M_L_r = self.Large.M[-3:, -3:]
        f_ext_L_r = self.Large.f_ext[-3:]
        K_L_r = self.Large.K[-3:, -3:]

        # Step 1.1 Predict subframe kinematics
        ut_njS_S_r = self.u_r_S + self.dt_f * self.Small.v[:3] + self.Small.a[:3] * (0.5 - self.Small.beta) * self.dt_f**2
        ut_njS_L_r = self.Large.u[-3:] + self.Large.dt * self.Large.v[-3:] + self.Large.a[-3:] * (0.5 - self.Large.beta) * self.Large.dt**2
        
        self.v_r_S += self.a_r_S * (1 - self.Small.gamma) * self.dt_f 
        self.v_r_L += self.a_r_L * (1 - self.Large.gamma) * self.dt_f

        # Step 1.2 Evaluate Acceleration of both interface regions
        at_njS_S_r = np.linalg.solve(M_S_r, f_ext_S_r - np.dot(K_S_r, ut_njS_S_r))
        at_njS_L_r = np.linalg.solve(M_L_r, f_ext_L_r - np.dot(K_L_r, ut_njS_L_r)) 

        B_S_r = self.B_S[:3]
        B_L_r = self.B_L[-3:]        
        Bat_njS_S = np.dot(np.transpose(B_S_r), at_njS_S_r)
        Bat_njS_L = np.dot(np.transpose(B_L_r), at_njS_L_r)

        f_n1_f = (self.L_S * (1 / self.BMB_S) *  Bat_njS_S) + (self.L_L * (1 / self.BMB_L) *  Bat_njS_L)  # Summation of Internal forces # 2.(c)
        a_n1_f = self.invM_f * f_n1_f # Evaluate Frame Acceleration Explicitly # 2.(d)
        v_n1_f = np.dot(np.transpose(self.B_S), self.Small.v) + (np.dot(np.transpose(self.B_S), self.Small.a) * (1 - self.Small.gamma) * self.dt_f)
        v_n1_f += a_n1_f * self.Small.gamma * self.dt_f  # 2.(e)

        # Update Frame
        self.a_f = a_n1_f
        self.v_f = v_n1_f
        self.u_f = np.dot(np.transpose(B_S_r), ut_njS_S_r)

        ## Solution of the Interface Regions (2 Elements)
        # Compute Lagrange Multipliers Explicitly
        self.Lambda_n1r_L = (1 / self.BMB_L) * (Bat_njS_L - a_n1_f) # 3.(b)
        self.Lambda_n1r_S = -self.Lambda_n1r_L

        # Solution of Small Region
        self.u_r_S = ut_njS_S_r
        self.a_r_S = at_njS_S_r - np.dot(np.linalg.inv(M_S_r), (B_S_r * self.Lambda_n1r_S)) 
        self.v_r_S += self.a_r_S * self.Small.gamma * self.dt_f 
        self.t_r_S += self.dt_f

        # Solution of Large Region
        self.u_r_L = ut_njS_L_r
        self.a_r_L = at_njS_L_r - np.dot(np.linalg.inv(M_L_r), (B_L_r * self.Lambda_n1r_L))
        self.v_r_L += self.a_r_L * self.Large.gamma * self.dt_f
        self.t_r_L += self.dt_f
        # 3.(e) avoid drifting here too
        self.u_r_L[-1] = self.u_f
        self.v_r_L[-1] = self.v_f

        # Update Frame
        self.t_f = self.t_f + self.dt_f
        self.n_f += 1

    def solve_subdomains(self, Domain: Domain, Lambda_n1r, invM_r, B_r):
        Domain.element_update()
        
        if (Domain.t == 0):
            Domain.a = np.linalg.solve(Domain.M, Domain.f_ext - np.dot(Domain.K, Domain.u))
            Domain.assemble_vbcs(Domain.t)        
        if (Domain.label == 'Large'):
            Domain.a[0] = 0.0

        # Compute Unconstrained Kinematics for Subdomains Time Step
        ut_n1_r = Domain.u + Domain.dt * Domain.v + Domain.a * (0.5 - Domain.beta) * Domain.dt**2
        vt_n1_r = Domain.v + Domain.a * (1 - Domain.gamma) * Domain.dt
        at_n1_r = np.linalg.solve(Domain.M, Domain.f_ext - np.dot(Domain.K, ut_n1_r))

        # Solution of Linear Problem
        # Explicit Method        
        if (Domain.label == 'Large'):
            at_n1_r[0] = 0.0        
        a_n1_r = at_n1_r - np.dot(invM_r, (B_r * Lambda_n1r))            
        v_n1_r = vt_n1_r + a_n1_r  * Domain.gamma * Domain.dt

        # Update State Variables        
        Domain.u = ut_n1_r 
        Domain.v = v_n1_r
        Domain.assemble_vbcs(Domain.t)
        Domain.a = a_n1_r 

        # Correct Drifting
        Domain.u = np.dot(np.identity(Domain.n_nodes) - np.outer(B_r, np.transpose(B_r)) , Domain.u) # I - BB^T * u
        Domain.u += np.dot(B_r, self.u_f) # + BL * u_f
        Domain.v = np.dot(np.identity(Domain.n_nodes) - np.outer(B_r, np.transpose(B_r)) , Domain.v) # I - BB^T * v
        Domain.v += np.dot(B_r, self.v_f) # + BL * v_f

        # Reset Region 
        self.u_r_S[0] = self.u_f
        self.v_r_S[0] = self.v_f
        self.a_r_S[0] = self.a_f
        self.u_r_L[-1] = self.u_f
        self.v_r_L[-1] = self.v_f
        self.a_r_L[-1] = self.a_f

        Domain.t = Domain.t + Domain.dt
        Domain.n += 1

    def Dvorak_multistep(self):
        """
        Integrate the Domain using the CDM Scheme
        """

        while (self.t_s_new <= self.t_L_new + 1e-12):    

            # Solution of Solvable Subframes
            self.solve_subframes()

            # Solution of Small Solvable Subdomain
            self.solve_subdomains(self.Small, self.Lambda_n1r_S, self.invM_S, self.B_S)
            self.t_s_act = self.Small.t 
            self.t_s_new = self.t_s_act + self.Small.dt 
        
        # Solution of Solvable Subframes
        self.solve_subframes()

        # Solution of Large Solvable Subdomain   
        self.solve_subdomains(self.Large, self.Lambda_n1r_L, self.invM_L, self.B_L)
        self.t_L_act = self.Large.t 
        self.t_L_new = self.t_L_act + self.Large.dt

    def time_equivalence(self): # This should be done for regions, as subdomains do not need to be time equivalent
        # Check for Time Equivalence
        if abs(self.Large.t - self.Small.t) > 1e-12:
            print("Time Discrepancy")
            print("Large Time:", full_Domain.Large.t, "Small Time:", full_Domain.Small.t)
            exit()

if __name__ == '__main__':
    # Initialise Domains
    # Large Domain
    E_L = 0.02 * 10**9 # 0.02GPa
    rho_L = 8000 # 8000kg/m^3
    E_S = 0.18 * 10**9 # Integer Time Step Ratio = 3    
    # E_S = ((450/np.pi)**2) * rho_L # Non Integer Time Step Ratio = 2.86
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
    Domain_S = Domain('Small', E_S, rho_L, length_S, area_L, num_elements_S, safety_Param, None)
    Domain_S.compute_mass_matrix()
    Domain_S.compute_stiffness_matrix()
    # Multistep Combined Domains
    full_Domain = Multistep(Domain_L, Domain_S, 3)

    # Visualisation
    bar = Visualise_MTS(full_Domain)

    # Integrate over time
    while Domain_L.t < 0.0015:
        full_Domain.Dvorak_multistep()
        full_Domain.time_equivalence()
        print("Time: ", Domain_L.t)
        if Domain_L.n % 10 == 0: 
            bar.plot_accel()
            bar.plot_vel()
            bar.plot_disp()
            bar.plot_stress()

            # bar.plot_interface_accel()
            # bar.plot_interface_vel()
            # bar.plot_interface_disp()

    bar.create_gif('DvoFEM1DAccel.gif', bar.filenames_accel)
    bar.create_gif('DvoFEM1DVel.gif', bar.filenames_vel)
    bar.create_gif('DvoFEM1DDisp.gif', bar.filenames_disp)
    bar.create_gif('DvoFEM1DStress.gif', bar.filenames_stress)
    
    # bar.create_gif('DvoFEM1DAccel_r.gif', bar.filenames_accel_r)
    # bar.create_gif('DvoFEM1DVel_r.gif', bar.filenames_vel_r)
    # bar.create_gif('DvoFEM1DDisp_r.gif', bar.filenames_disp_r)


