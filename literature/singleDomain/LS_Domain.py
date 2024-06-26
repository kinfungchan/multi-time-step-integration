import numpy as np
from boundaryConditions.BoundaryConditions import  VelBoundaryConditions as vbc

"""
The integrate_nb() method reimplements the Newmark Beta Method for
explicit and implicit time integration of the structural equation of motion

The integrate_pfpb() method implements the pushforward-pullback global
method for the time integration of the structural equation of motion

Reference: Cho, S. S., Kolman, R., González, J. A., & Park, K. C. (2019).
Explicit multistep time integration for discontinuous elastic stress wave
propagation in heterogeneous solids. International Journal for Numerical
Methods in Engineering, 118(5), 276-302

This script attends to the Numerical Example in Section 4.1 of the paper
It solves for two domains (Large and Small) with a SINGLE time step

"""

class Domain:
    """
    Constructor for the One Dimensional Domain class

    :param label: Large or Small
    :param young: Young's Modulus
    :param density: Density
    :param length: Length of the domain
    :num_elements: Number of elements
    :param Co: Courant Number

    """
    def __init__(self, label, young_L, young_S, density, length_L, length_S, area, num_elements_L, num_elements_S, safety_Param, v_bc: vbc):
        self.label = label
        self.E_L = young_L
        self.E_S = young_S
        self.rho = density
        self.L_L = length_L
        self.L_S = length_S
        self.A = area
        # self.n_elems = num_elements * 2
        self.n_elems_L = num_elements_L
        self.n_elems_S = num_elements_S
        self.C = 1.0
        self.n_nodes = self.n_elems_L + self.n_elems_S + 1
        self.position, self.dx = np.linspace(0, length_L, self.n_nodes, retstep=True)
        self.midposition = [self.position[n] + 0.5 * self.dx for n in range(0, len(self.position)-1)]

        self.position_L, self.dx_L = np.linspace(0, self.L_L, self.n_elems_L + 1, retstep=True)
        self.position_S, self.dx_S = np.linspace(self.L_L, self.L_L + self.L_S, self.n_elems_S + 1, retstep=True)
        self.position = np.concatenate((self.position_L, self.position_S[1:]))

        self.midposition_L = [self.position_L[n] + 0.5 * self.dx_L for n in range(0, len(self.position_L)-1)]
        self.midposition_S = [self.position_S[n] + 0.5 * self.dx_S for n in range(0, len(self.position_S)-1)]
        self.midposition = np.concatenate((self.midposition_L, self.midposition_S))

        self.v_bc = v_bc
        self.n = 0
        self.t = 0.0
        self.a = np.zeros(self.n_nodes) # Acceleration Field
        self.v = np.zeros(self.n_nodes) # Velocity Field
        self.u = np.zeros(self.n_nodes) # Displacement Field
        self.stress = np.zeros(self.n_elems_L + self.n_elems_S) # Stress Field
        self.strain = np.zeros(self.n_elems_L + self.n_elems_S) # Strain Field    
        self.dt = safety_Param * self.dx_S * np.sqrt(self.rho / self.E_S) # Time Step with Safety Parameter
        self.dt_C =  self.C * self.dx_S * np.sqrt(self.rho / self.E_S) # Time Step at Courant = 1.0

        self.K = np.zeros((self.n_nodes, self.n_nodes)) # Stiffness Matrix
        self.M = np.zeros((self.n_nodes, self.n_nodes)) # Mass Matrix
        self.f_ext = np.zeros(self.n_nodes) # External Force Vector

        self.beta = 0.0 # Coefficients for the Newmark Scheme
        self.gamma = 0.5

        self.min_dt = np.inf
        self.el_steps = 0

    def compute_mass_matrix(self):
        '''
        Lumped Mass Matrix
        '''
        nodal_mass_L = 0.5 * self.rho * self.A * self.dx_L
        nodal_mass_S = 0.5 * self.rho * self.A * self.dx_S
        for i in range(self.n_elems_L + 1):
            if i > 0 and i < self.n_nodes - 1:
                self.M[i, i] = 2 * nodal_mass_L
            else:
                self.M[i, i] = nodal_mass_L
        for i in range(self.n_elems_L, self.n_nodes):
            if i > self.n_elems_L and i < self.n_nodes - 1:
                self.M[i, i] = 2 * nodal_mass_S
            else:
                self.M[i, i] += nodal_mass_S
        print("Lumped Mass Matrix")
        
    def compute_stiffness_matrix(self):
        '''
        Stiffness Matrix
        '''
        stiffness_element_L = self.E_L * self.A / self.dx_L
        for i in range(self.n_elems_L + 1):
            self.K[i, i] += stiffness_element_L
            if i != (self.n_elems_L) + 1 - 1:
                self.K[i, i + 1] -= stiffness_element_L
                self.K[i + 1, i] -= stiffness_element_L
            if i > 0 and i < (self.n_elems_L):
                self.K[i, i] += stiffness_element_L
        stiffness_element_S = self.E_S * self.A / self.dx_S
        for i in range(self.n_elems_L, self.n_nodes):
            self.K[i, i] += stiffness_element_S
            if i != self.n_nodes - 1:
                self.K[i, i + 1] -= stiffness_element_S
                self.K[i + 1, i] -= stiffness_element_S
            if i > (self.n_elems_L) and i < self.n_nodes - 1:
                self.K[i, i] += stiffness_element_S
        print("Stiffness Matrix")
        print(self.K)

    def element_update(self):
        self.strain = np.concatenate((np.diff(self.u[:self.n_elems_L+1]) / self.dx_L, np.diff(self.u[self.n_elems_L:]) / self.dx_S))
        for i in range(self.n_elems_L):
            self.stress[i] = self.strain[i] * self.E_L
        for i in range(self.n_elems_L, self.n_elems_L + self.n_elems_S):
            self.stress[i] = self.strain[i] * self.E_S

    def assemble_vbcs(self, t):
        if (self.v_bc):
            for counter in range(0, len(self.v_bc.indexes)):
                self.v[self.v_bc.indexes[counter]] = self.v_bc.velocities[counter](t)

    def integrate_nb(self):
        """
        Function to solve discretised structural equation of motion using the Newmark Beta Method

        """      
        if (self.t == 0):
            self.a = np.linalg.solve(self.M, self.f_ext - np.dot(self.K, self.u))
            self.assemble_vbcs(self.t)
        # Calculation of Predictors
        self.a[0] = 0.0
        u_k1 = self.u + self.dt * self.v + self.a * (0.5 - self.beta) * self.dt**2
        v_k1 = self.v + self.a * (1 - self.gamma) * self.dt
        
        # Solution of Linear Problem
        # Explicit Method
        if (self.beta == 0.0):
            a_k1 = np.linalg.solve(self.M, self.f_ext - np.dot(self.K, u_k1))
            a_k1[0] = 0.0
        # Implicit Method
        else:        
            LHS = self.M + (self.K * self.beta * self.dt**2) 
            RHS = self.f_ext - np.dot(self.K, u_k1)
            a_k1 = np.linalg.solve(LHS, RHS) # Requires inverse of the K matrix  
            a_k1[0] = 0.0
        # Calculation of Correctors
        u_k1 = u_k1 + a_k1 * self.beta * self.dt**2
        v_k1 = v_k1 + a_k1 * self.gamma * self.dt

        # Update State Variables        
        self.u = u_k1
        self.v = v_k1
        self.assemble_vbcs(self.t)
        self.a = a_k1
        self.t = self.t + self.dt 
        self.n += 1
        self.min_dt = min(self.min_dt, self.dt)

    def integrate_pfpb(self):
        '''
        Function to solve the discretised structural equation of motion using the
        pushforward-pullback method
 
        Time stepping with one time step and two domains
        '''
        if (self.t == 0):
            self.a = np.linalg.solve(self.M, self.f_ext - np.dot(self.K, self.u))
            self.assemble_vbcs(self.t)
        # Push-forward Step
        self.a[0] = 0.0
        u_nC = self.u + self.dt_C * self.v + self.a * (0.5 * self.dt_C**2)
        # v_nC = self.v + self.dt_C * self.a # velocity at n+C is not needed
        a_nC = np.linalg.solve(self.M, self.f_ext - np.dot(self.K, u_nC))    
        a_nC[0] = 0.0
        # Weight Coefficients (maintain in Loop for when dt changes in Updated Lagrangian)
        alpha = self.dt / self.dt_C
        theta = 0.5 # 0.5 for Average Displacement , 0 for CDM
        beta_1 = (alpha / 6) * (3 * alpha + theta - (theta * alpha ** 2))
        beta_2 = theta * (alpha / 6) * (alpha ** 2 - 1)
        self.n += 1
        # Pullback Step
        self.u = self.u + (self.dt * self.v) + (beta_1 * (self.dt_C)**2 * self.a) + (beta_2 * (self.dt_C)**2 * a_nC)
        self.a = np.linalg.solve(self.M, self.f_ext - np.dot(self.K, self.u))
        self.a[0] = 0.0
        self.v = self.v + self.dt * ((1 - self.gamma) * self.a + self.gamma * self.a)  
        self.assemble_vbcs(self.t)    
        self.t = self.t + self.dt
        self.n += 1
        self.el_steps += (self.n_elems_L + self.n_elems_S)
        self.min_dt = min(self.min_dt, self.dt)

