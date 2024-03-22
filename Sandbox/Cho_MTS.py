import numpy as np
import matplotlib.pyplot as plt
from SingleDomain import Domain
from SingleDomain import Visualise_Monolithic
from BoundaryConditions import  VelBoundaryConditions as vbc
import imageio
import os

"""
In this notebook we look to reimplement Explicit Multistep Time Integration
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
        self.B_L[self.Large.n_nodes - 1] = 1

        # Small Domain Interface
        self.B_S = np.zeros(self.Small.n_nodes) # Boolean Vectors for Extracting Interface DOFs for each domain
        self.L_S = 1 # Boolean Vectors for Extracting Interface DOFs for global acc and disp
        self.B_S[0] = 1

        # Frame
        self.a_f = np.zeros(1) 
        self.v_f = np.zeros(1)
        self.u_f = np.zeros(1)

        # Time Integration Constants
        self.theta = 0.5 # Average Displacement for Push-Forward Pullback
        # For Total Lagrangian Formulation beta terms remain constant
        self.alpha_L = self.Large.dt / self.Large.dt_C
        self.beta1_L = (self.alpha_L / 6) * (3 * self.alpha_L + self.theta - (self.theta * self.alpha_L ** 2))
        self.beta2_L = self.theta * (self.alpha_L / 6) * (self.alpha_L ** 2 - 1)
        self.alpha_S = self.Small.dt / self.Small.dt_C
        self.beta1_S = (self.alpha_S / 6) * (3 * self.alpha_S + self.theta - (self.theta * self.alpha_S ** 2))
        self.beta2_S = self.theta * (self.alpha_S / 6) * (self.alpha_S ** 2 - 1) 

    def multistep_pfpb(self):
        """
        Integrate the Domain using the PFPB Scheme
        """
        invM_L = np.linalg.inv(self.Large.M)
        invM_S = np.linalg.inv(self.Small.M)
        BMB_S = np.dot(np.transpose(self.B_S), np.dot(invM_S, self.B_S)) # B_S^T * M_S^-1 * B_S
        BMB_L = np.dot(np.transpose(self.B_L), np.dot(invM_L, self.B_L)) # B_S^T * M_S^-1 * B_S

        k = 0
        prev_u_S = self.Small.u
        prev_v_S = self.Small.v
        prev_a_S = self.Small.a

        while (k < self.m):
            '''
            Update of Small Domain
            '''
            self.Small.element_update()
            
            if (self.Small.t == 0):
                self.Small.a = np.linalg.solve(self.Small.M, self.Small.f_ext - np.dot(self.Small.K, self.Small.u))

            ## Push-forward Step for Small Courant Timestep nC_S
            # Find Unconstrained values for small Courant timestep (tilda)
            ut_nCS_S = self.Small.u + self.Small.dt_C * self.Small.v + self.Small.a * (0.5 * self.Small.dt_C**2)
            at_nCS_S = np.linalg.solve(self.Small.M, self.Small.f_ext - np.dot(self.Small.K, ut_nCS_S))
            ut_nCS_L = self.Large.u + self.Small.dt_C * self.Large.v + self.Large.a * (0.5 * self.Small.dt_C**2)
            at_nCS_L = np.linalg.solve(self.Large.M, self.Large.f_ext - np.dot(self.Large.K, ut_nCS_L))
            
            # Calculate the Lagrange Multipliers on Small Push-Forward Step (Courant Step)
            Bat_nCS_S = np.dot(np.transpose(self.B_S), at_nCS_S)
            Bat_nCS_L = np.dot(np.transpose(self.B_L), at_nCS_L)
            Lambda_nCS_L, Lambda_nCS_S, a_nCS_f = solve_Interface_EOM(BMB_L, BMB_S, self.L_L, self.L_S, Bat_nCS_L , Bat_nCS_S)

            # Calculating constrained acceleration
            a_nCS_S = at_nCS_S - np.dot(invM_S, self.B_S * Lambda_nCS_S)
            a_nCS_L = at_nCS_L - np.dot(invM_L, self.B_L * Lambda_nCS_L)

            ## Pullback Step for actual Small Timestep nj_S 
            # Find Unconstrained values for actual small timestep
            ut_njS_S = self.Small.u + (self.Small.dt * self.Small.v) + (self.beta1_S * (self.Small.dt_C)**2 * self.Small.a) + (self.beta2_S * (self.Small.dt_C)**2 * at_nCS_S)
            at_njS_S = np.linalg.solve(self.Small.M, self.Small.f_ext - np.dot(self.Small.K, ut_njS_S))
            ut_njS_L = self.Large.u + (self.Small.dt * self.Large.v) + (self.beta1_S * (self.Small.dt_C)**2 * self.Large.a) + (self.beta2_S * (self.Small.dt_C)**2 * at_nCS_L)
            at_njS_L = np.linalg.solve(self.Large.M, self.Large.f_ext - np.dot(self.Large.K, ut_njS_L))

            # Calculate the Lagrange Multipliers on Pullback Step
            Bat_njS_S = np.dot(np.transpose(self.B_S), at_njS_S)
            Bat_njS_L = np.dot(np.transpose(self.B_L), at_njS_L)
            Lambda_njS_L, Lambda_njS_S, a_njS_f = solve_Interface_EOM(BMB_L, BMB_S, self.L_L, self.L_S, Bat_njS_L , Bat_njS_S)

            # Update Small Domain
            self.Small.u = self.Small.u + (self.Small.dt * self.Small.v) + (self.beta1_S * (self.Small.dt_C)**2 * self.Small.a) + (self.beta2_S * (self.Small.dt_C)**2 * a_nCS_S)
            self.Small.a = at_njS_S - np.dot(invM_S, self.B_S * Lambda_njS_S) # = np.linalg.solve(self.M, self.f_ext - np.dot(self.K, self.u)) - np.dot(invM_S, np.dot(self.B_S, self.Lambda_nj_S))
            self.Small.v = self.Small.v + self.Small.dt * ((1 - self.Small.gamma) * self.Small.a + self.Small.gamma * self.Small.a) # Use of old a here?
            self.Small.t = self.Small.t + self.Small.dt
            self.Small.n += 1

            # Update Small Frames
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
        
        ## Push-forward Step for Large Courant Timestep nC_L
        self.Large.a[0] = 0.0
        # Find Unconstrained values for large Courant timestep (tilda)
        ut_nCL_S = prev_u_S + self.Large.dt_C * prev_v_S + prev_a_S * (0.5 * self.Large.dt_C**2)
        at_nCL_S = np.linalg.solve(self.Small.M, self.Small.f_ext - np.dot(self.Small.K, ut_nCL_S))
        ut_nCL_L = self.Large.u + self.Large.dt_C * self.Large.v + self.Large.a * (0.5 * self.Large.dt_C**2)
        at_nCL_L = np.linalg.solve(self.Large.M, self.Large.f_ext - np.dot(self.Large.K, ut_nCL_L))
        at_nCL_L[0] = 0.0

        # Calculate the Lagrange Multipliers on Small Push-Forward Step (Courant Step)
        Bat_nCL_S = np.dot(np.transpose(self.B_S), at_nCL_S)
        Bat_nCL_L = np.dot(np.transpose(self.B_L), at_nCL_L)
        Lambda_nCL_L, Lambda_nCL_S, a_nCL_f = solve_Interface_EOM(BMB_L, BMB_S, self.L_L, self.L_S, Bat_nCL_L , Bat_nCL_S)

        # Calculating constrained acceleration
        a_nCL_S = at_nCS_S - np.dot(invM_S, self.B_S * Lambda_nCL_S)    
        a_nCL_L = at_nCS_L - np.dot(invM_L, self.B_L * Lambda_nCL_L)
        a_nCL_L[0] = 0.0

        ## Pullback Step for Large Timestep nj_L
        # Find Unconstrained values for actual small timestep
        ut_njL_S = prev_u_S + (self.Large.dt * prev_v_S) + (self.beta1_S * (self.Large.dt_C)**2 * prev_a_S) + (self.beta2_S * (self.Large.dt_C)**2 * at_nCL_S)
        at_njL_S = np.linalg.solve(self.Small.M, self.Small.f_ext - np.dot(self.Small.K, ut_njL_S))
        ut_njL_L = self.Large.u + (self.Large.dt * self.Large.v) + (self.beta1_S * (self.Large.dt_C)**2 * self.Large.a) + (self.beta2_S * (self.Large.dt_C)**2 * at_nCL_L)
        at_njL_L = np.linalg.solve(self.Large.M, self.Large.f_ext - np.dot(self.Large.K, ut_njL_L))
        at_njL_L[0] = 0.0

        # Calculate the Lagrange Multipliers on Pullback Step
        Bat_njL_S = np.dot(np.transpose(self.B_S), at_njL_S)
        Bat_njL_L = np.dot(np.transpose(self.B_L), at_njL_L)
        Lambda_njL_L, Lambda_njL_S, a_njL_f = solve_Interface_EOM(BMB_L, BMB_S, self.L_L, self.L_S, Bat_njL_L , Bat_njL_S)

        # Update Large Domain
        self.Large.u = self.Large.u + (self.Large.dt * self.Large.v) + (self.beta1_S * (self.Large.dt_C)**2 * self.Large.a) + (self.beta2_S * (self.Large.dt_C)**2 * a_nCL_L)
        # self.Large.u[300] = self.u_f        
        self.Large.a = at_njL_L - np.dot(invM_L, self.B_L * Lambda_njL_L) 
        self.Large.a[0] = 0.0
        # self.Large.a[300] = self.a_f
        self.Large.v = self.Large.v + self.Large.dt * ((1 - self.Large.gamma) * self.Large.a + self.Large.gamma * self.Large.a) # Use of old a here?
        self.Large.assemble_vbcs(self.Large.t)    
        self.Large.t = self.Large.t + self.Large.dt
        self.Large.n += 1

        # Prevent Drifting with Small Frame
        self.Large.u[self.Large.n_nodes - 1] = self.u_f
        self.Large.v[self.Large.n_nodes - 1] = self.v_f

        # Check for Time Equivalence
        if abs(self.Large.t - self.Small.t) > 1e-10:
            print("Time Discrepancy")
            print("Large Time:", full_Domain.Large.t, "Small Time:", full_Domain.Small.t)
            exit()

class Visualise_MTS:

    def __init__(self, Domain: Multistep):

        self.domain = Domain
        self.filenames_accel = []
        self.filenames_vel = []
        self.filenames_disp = []
        self.filenames_stress = []    

    def plot(self, variable_L, variable_S, position_L, position_S, title, xlabel, ylabel, filenames):
        filenames.append(f'FEM1D_{title}{self.domain.Large.n}.png')
        plt.style.use('ggplot')
        plt.plot(position_L, variable_L)
        plt.plot([position + self.domain.Large.length for position in position_S], variable_S)  # Convert self.domain.Large.length to a list
        plt.title(title,fontsize=12)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(["Time: " + format(self.domain.Large.t * 1e6, ".1f") + "us"])
        plt.savefig(f'FEM1D_{title}{self.domain.Large.n}.png')
        plt.close()

    def plot_accel(self):
        self.plot(self.domain.Large.a, self.domain.Small.a, self.domain.Large.position, self.domain.Small.position, "Acceleration", "Domain Position (m)", "Acceleration (m/s^2)", self.filenames_accel)

    def plot_vel(self):
        self.plot(self.domain.Large.v, self.domain.Small.v, self.domain.Large.position, self.domain.Small.position,  "Velocity", "Domain Position (m)", "Velocity (m/s)", self.filenames_vel)

    def plot_disp(self):
        self.plot(self.domain.Large.u, self.domain.Small.u, self.domain.Large.position, self.domain.Small.position,  "Displacement", "Domain Position (m)", "Displacement (m)", self.filenames_disp)

    def plot_stress(self):
        self.plot(self.domain.Large.stress, self.domain.Small.stress, self.domain.Large.midposition, self.domain.Small.midposition, "Stress", "Domain Position (m)", "Stress (Pa)", self.filenames_stress)

    def create_gif(self, gif_name, filenames):
        with imageio.get_writer(gif_name, mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        for filename in set(filenames):
            os.remove(filename)

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
        full_Domain.multistep_pfpb()
        print("Time: ", Domain_L.t)
        if Domain_L.n % 40 == 0: 
            bar.plot_accel()
            bar.plot_vel()
            bar.plot_disp()
            bar.plot_stress()

    bar.create_gif('FEM1DAccel.gif', bar.filenames_accel)
    bar.create_gif('FEM1DVel.gif', bar.filenames_vel)
    bar.create_gif('FEM1DDisp.gif', bar.filenames_disp)
    bar.create_gif('FEM1DStress.gif', bar.filenames_stress)

