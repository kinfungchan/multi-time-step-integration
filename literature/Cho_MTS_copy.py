import numpy as np
from singleDomain import Domain
from Cho_PFPB import Visualise_MTS
from boundaryConditions.BoundaryConditions import  VelBoundaryConditions as vbc
from utils.Utils import exportCSV
import matplotlib.pyplot as plt

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

        # List of Time Histories
        self.steps_2El = np.array([0.0])
        self.steps_L = np.array([0.0])
        self.steps_S = np.array([0.0])

        # Minimum Time Step 
        self.min_dt = np.inf
        self.el_steps = 0

        self.Lambda_njS_L = np.array([0.0])
        self.Lambda_njS_S = np.array([0.0])
        self.dW_Link_L = np.array([0.0])
        self.dW_Link_S = np.array([0.0])
        self.t_sync = np.array([0.0])
        self.a_drift = np.array([0.0])
        self.v_drift = np.array([0.0])
        self.u_drift = np.array([0.0])

    def Cho_multistep(self):
        """
        Integrate the Domain using the CDM Scheme
        """
        invM_L = np.linalg.inv(self.Large.M)
        invM_S = np.linalg.inv(self.Small.M)
        BMB_S = np.dot(np.transpose(self.B_S), np.dot(invM_S, self.B_S)) # B_S^T * M_S^-1 * B_S
        BMB_L = np.dot(np.transpose(self.B_L), np.dot(invM_L, self.B_L)) # B_S^T * M_S^-1 * B_S

        k = 0
        dW_Link_S = 0.0
        dW_Link_L = 0.0

        while (k < self.m):
            '''
            Update of Small Domain
            '''
            self.Small.element_update()

            if (self.Small.t == 0):
                self.Small.a = np.linalg.solve(self.Small.M, self.Small.f_ext - np.dot(self.Small.K, self.Small.u))

            # Compute Unconstrained Kinematics for Small Time Step
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
            # Compute Lagrange Multipliers and Frame Acceleration
            Lambda_njS_L, Lambda_njS_S, a_njS_f = solve_Interface_EOM(BMB_L, BMB_S, self.L_L, self.L_S, Bat_njS_L , Bat_njS_S)    
            
            self.Lambda_njS_L = np.append(self.Lambda_njS_L, Lambda_njS_L)
            self.Lambda_njS_S = np.append(self.Lambda_njS_S, Lambda_njS_S)      
            dW_Link_S += 0.5 * (self.Small.u[0] - self.Small.u_prev[0]) * (Lambda_njS_S + self.Lambda_njS_S[-2])      

            if (self.Small.t == 0):
                self.Small.a = np.linalg.solve(self.Small.M, self.Small.f_ext - np.dot(self.Small.K, self.Small.u)) 

            # Calculation of Predictors
            vt_njS_S = self.Small.v + self.Small.a * (1 - self.Small.gamma) * self.Small.dt
        
            # Solution of Linear Problem
            # Explicit Method
            a_njS_S = at_njS_S - np.dot(invM_S, (self.B_S * Lambda_njS_S))   

            # Calculation of Correctors
            v_njS_S = vt_njS_S + a_njS_S * self.Small.gamma * self.Small.dt

            # Update State Variables   
            self.Small.u_prev = np.copy(self.Small.u)
            self.Small.v_prev = np.copy(self.Small.v)
            self.Small.a_prev = np.copy(self.Small.a)

            self.Small.u = ut_njS_S
            self.Small.v = v_njS_S
            self.Small.a = a_njS_S
            self.Small.t = self.Small.t + self.Small.dt
            self.Small.n += 1
            self.el_steps += self.Small.n_elems + 2 # +two layer elements of L-domain adjacent to interface
            self.steps_2El = np.append(self.steps_2El, self.Small.dt)
            self.steps_S = np.append(self.steps_S, self.Small.dt)

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

        dW_Link_L += 0.5 * (self.Large.u[-1] - self.Large.u_prev[-1]) * (Lambda_njS_L + self.Lambda_njS_L[-2])
        
        self.Large.a[0] = 0.0
        # Compute Unconstrained Kinematics for Large Time Step
        ut_n1L_L = self.Large.u + self.Large.dt * self.Large.v + self.Large.a * (0.5 - self.Large.beta) * self.Large.dt**2
        vt_n1L_L = self.Large.v + self.Large.a * (1 - self.Large.gamma) * self.Large.dt
       
        # Solution of Linear Problem
        # Explicit Method
        at_n1L_L = np.linalg.solve(self.Large.M, self.Large.f_ext - np.dot(self.Large.K, ut_n1L_L))
        at_n1L_L[0] = 0.0
        a_n1L_L = at_n1L_L - np.dot(invM_L, (self.B_L * Lambda_njS_L))            

        v_n1L_L = vt_n1L_L + a_n1L_L  * self.Large.gamma * self.Large.dt

        # Update State Variables        
        self.Large.u_prev = np.copy(self.Large.u)
        self.Large.v_prev = np.copy(self.Large.v)
        self.Large.a_prev = np.copy(self.Large.a)

        self.Large.u = ut_n1L_L 
        self.Large.v = v_n1L_L
        self.Large.assemble_vbcs(self.Large.t)
        self.Large.a = a_n1L_L 

        # self.Large.u[-1] = self.u_f
        # self.Large.v[-1] = self.v_f
        self.a_drift = np.append(self.a_drift, self.Large.a[-1] - self.Small.a[0])
        self.v_drift = np.append(self.v_drift, self.Large.v[-1] - self.Small.v[0])
        self.u_drift = np.append(self.u_drift, self.Large.u[-1] - self.Small.u[0])

        self.Large.t = self.Large.t + self.Large.dt
        self.Large.n += 1
        self.el_steps += self.Large.n_elems
        self.steps_L = np.append(self.steps_L, self.Large.dt)

        self.t_sync = np.append(self.t_sync, self.Large.t)
        self.dW_Link_L = np.append(self.dW_Link_L, dW_Link_L)
        self.dW_Link_S = np.append(self.dW_Link_S, dW_Link_S)

        # Update Minimum Time Step
        self.min_dt = min(self.min_dt, self.Small.dt, self.Large.dt)

def ChoCoupling():
    # Initialise Domains
    # Large Domain
    E_L = 0.02 * 10**9 # 0.02GPa
    rho_L = 8000 # 8000kg/m^3
    # E_S = 0.18 * 10**9 # Integer Time Step Ratio = 3    
    E_S = (np.pi/0.02)**2 * rho_L # Non Integer Time Step Ratio = pi
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
    m_int = np.ceil(Domain_L.dt / Domain_S.dt)
    Domain_S.dt = Domain_L.dt / m_int
    full_Domain = Multistep(Domain_L, Domain_S, m_int)

    # Visualisation
    bar = Visualise_MTS(full_Domain)

    # Integrate over time
    while Domain_L.t < 0.0016:
        full_Domain.Cho_multistep()
        print("Time: ", Domain_L.t)
        if Domain_L.n % 500 == 0: 
            bar.plot_accel()
            bar.plot_vel()
            bar.plot_disp()
            bar.plot_stress()

        if Domain_L.n % 900 == 0:
            exportCSV('Square_Cho_v_L2.csv', 'Square_Cho_v_S2.csv', Domain_L, Domain_S)

    # plot_dW_Link
    plt.plot(full_Domain.t_sync, full_Domain.dW_Link_L + full_Domain.dW_Link_S, label='Total')
    # plt.plot(full_Domain.t_sync, full_Domain.dW_Link_L, label='Large')
    # plt.plot(full_Domain.t_sync, full_Domain.dW_Link_S, label='Small')
    plt.xlabel('Time (s)')
    plt.ylabel('dW_Link')
    plt.title('dW_Link')
    plt.legend()
    plt.show()

    # plot_drift
    plt.plot(full_Domain.t_sync, full_Domain.a_drift, label='Acceleration')
    plt.xlabel('Time (s)')
    plt.ylabel('Drift')
    plt.title('Acceleration Drift')
    plt.legend()
    plt.show()

    plt.plot(full_Domain.t_sync, full_Domain.v_drift, label='Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Drift')
    plt.title('Velocity Drift')
    plt.legend()
    plt.show()

    plt.plot(full_Domain.t_sync, full_Domain.u_drift, label='Displacement')
    plt.xlabel('Time (s)')
    plt.ylabel('Drift')
    plt.title('Displacement Drift')
    plt.legend()
    plt.show()

    bar.create_gif('FEM1DAccel.gif', bar.filenames_accel)
    bar.create_gif('FEM1DVel.gif', bar.filenames_vel)
    bar.create_gif('FEM1DDisp.gif', bar.filenames_disp)
    bar.create_gif('FEM1DStress.gif', bar.filenames_stress)

    # Print Minimum Time Step for Whole Domain
    print("Minimum Time Step for Whole Domain: ", full_Domain.min_dt)
    # Print Total Number of Integration Steps 
    print("Number of Integration Steps: ", full_Domain.el_steps)
    # Print First 10 Time Steps on Large and Small
    print("Time Steps: ", full_Domain.steps_L[:10])
    print("Time Steps: ", full_Domain.steps_2El[:10])
    print("Time Steps: ", full_Domain.steps_S[:10])

if __name__ == '__main__':
    ChoCoupling()
    