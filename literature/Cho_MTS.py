import numpy as np
from literature.singleDomain import Domain
from boundaryConditions.BoundaryConditions import VelBoundaryConditions as vbc
from database import History
from utils.Visualise import Plot, Animation
from utils.Paper import Outputs

"""
In this notebook we look to reimplement CDM Multistep Time Integration
for One-Dimensional Heterogeneous Solids from the following paper


Reference: Cho, S. S., Kolman, R., González, J. A., & Park, K. C. (2019).
Explicit multistep time integration for discontinuous elastic stress wave
propagation in heterogeneous solids. International Journal for Numerical
Methods in Engineering, 118(5), 276-302
s
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

class Cho_MTS:
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

    def Cho_multistep(self):
        """
        Integrate the Domain using the CDM Scheme
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
        self.Large.u = ut_n1L_L 
        self.Large.v = v_n1L_L
        self.Large.assemble_vbcs(self.Large.t)
        self.Large.a = a_n1L_L 

        self.Large.u[-1] = self.u_f
        self.Large.v[-1] = self.v_f

        self.Large.t = self.Large.t + self.Large.dt
        self.Large.n += 1
        self.el_steps += self.Large.n_elems
        self.steps_L = np.append(self.steps_L, self.Large.dt)

        # Update Minimum Time Step
        self.min_dt = min(self.min_dt, self.Small.dt, self.Large.dt)

def ChoCoupling(bar):
    # Initialise Domains
    def vel(t): return vbc.velbcSquare(t, 2 * bar.length_L, bar.E_L, bar.rho_L)
    velboundaryConditions = vbc(list([0]), list([vel]))

    # Large Domain
    Domain_L = Domain('Large', bar.E_L, bar.rho_L, bar.length_L, bar.area_L,
                       bar.num_elem_L, bar.safety_Param, velboundaryConditions)
    Domain_L.compute_mass_matrix()
    Domain_L.compute_stiffness_matrix()

    # Small Domain
    Domain_S = Domain('Small', bar.E_S, bar.rho_S, bar.length_S, bar.area_L,
                       bar.num_elem_S, bar.safety_Param, None)
    Domain_S.compute_mass_matrix()
    Domain_S.compute_stiffness_matrix()

    # Multistep Combined Domains
    m_int = np.ceil(Domain_L.dt / Domain_S.dt)
    Domain_S.dt = Domain_L.dt / m_int
    full_Domain = Cho_MTS(Domain_L, Domain_S, m_int)

    # Intialise History
    hst_L = History(Domain_L.position, Domain_L.n_nodes, Domain_L.n_elems)
    hst_S = History(Domain_S.position, Domain_S.n_nodes, Domain_S.n_elems)    

    # Visualisation
    plot = Plot()
    animate = Animation(plot)
    sq_L = np.zeros((3, full_Domain.Large.n_nodes))
    sq_S = np.zeros((3, full_Domain.Small.n_nodes))
    pos_L = full_Domain.Large.position
    pos_S = full_Domain.Small.position + full_Domain.Large.L

    # Integrate over time
    while Domain_L.t <= 0.0016:
        full_Domain.Cho_multistep()

        # History Data
        hst_L.append_timestep(Domain_L.t, Domain_L.position, Domain_L.a, Domain_L.v, Domain_L.u, Domain_L.stress, Domain_L.strain)
        hst_S.append_timestep(Domain_S.t, Domain_S.position, Domain_S.a, Domain_S.v, Domain_S.u, Domain_S.stress, Domain_S.strain)

        print("Time: ", Domain_L.t)
        if Domain_L.n % 40 == 0: 
            animate.save_single_plot(2, [full_Domain.Large.position, [position + full_Domain.Large.L for position in full_Domain.Small.position]],
                                     [full_Domain.Large.a, full_Domain.Small.a],
                                     "Acceleration", "Domain Position (m)", "Acceleration (m/s^2)",
                                     [None, None], [None, None],
                                     animate.filenames_accel, full_Domain.Large.n,
                                     ["Large", "Small"])
            animate.save_single_plot(2, [full_Domain.Large.position, [position + full_Domain.Large.L for position in full_Domain.Small.position]],
                                     [full_Domain.Large.v, full_Domain.Small.v],
                                     "Velocity", "Domain Position (m)", "Velocity (m/s)",
                                     [None, None], [None, None],
                                     animate.filenames_vel, full_Domain.Large.n,
                                     ["Large", "Small"])
            animate.save_single_plot(2, [full_Domain.Large.position, [position + full_Domain.Large.L for position in full_Domain.Small.position]],
                                     [full_Domain.Large.u, full_Domain.Small.u],
                                     "Displacement", "Domain Position (m)", "Displacement (m)",
                                     [None, None], [None, None],
                                     animate.filenames_disp, full_Domain.Large.n,
                                     ["Large", "Small"])
            animate.save_single_plot(2, [full_Domain.Large.midposition, [position + full_Domain.Large.L for position in full_Domain.Small.midposition]],
                                     [full_Domain.Large.stress, full_Domain.Small.stress],
                                     "Stress", "Domain Position (m)", "Stress (Pa)",
                                     [None, None], [None, None],
                                     animate.filenames_stress, full_Domain.Large.n,
                                     ["Large", "Small"])

        if Domain_L.n % 600 == 0: # 0.00100
            sq_L[0] = full_Domain.Large.v
            sq_S[0] = full_Domain.Small.v
        if Domain_L.n % 750 == 0: # 0.00125
            sq_L[1] = full_Domain.Large.v
            sq_S[1] = full_Domain.Small.v
        if Domain_L.n % 900 == 0: # 0.00150
            sq_L[2] = full_Domain.Large.v
            sq_S[2] = full_Domain.Small.v

    animate.save_MTS_gifs("Cho")

    # Print Minimum Time Step for Whole Domain
    steps = [full_Domain.steps_L, full_Domain.steps_2El, full_Domain.steps_S]
    domains = ['$\Omega_L^{Cho}$', '$\Omega_{2EL}^{Cho}$', '$\Omega_S^{Cho}$']

    print("Minimum Time Step for Whole Domain: ", full_Domain.min_dt)
    # Print Total Number of Integration Steps 
    print("Number of Integration Steps: ", full_Domain.el_steps)

    outputs = Outputs(domains, steps, sq_L, sq_S, pos_L, pos_S)
    return outputs