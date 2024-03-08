import numpy as np
import matplotlib.pyplot as plt
from BoundaryConditions import  VelBoundaryConditions as vbc
import imageio
import os

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
    def __init__(self, label, young_L, young_S, density, length, area, num_elements, safety_Param, v_bc: vbc):
        self.label = label
        self.E_L = young_L
        self.E_S = young_S
        self.rho = density
        self.length = length
        self.A = area
        self.n_elems = num_elements * 2
        self.C = 1.0
        self.n_nodes = self.n_elems + 1
        self.position, self.dx = np.linspace(0, length, self.n_nodes, retstep=True)
        self.midposition = [self.position[n] + 0.5 * self.dx for n in range(0, len(self.position)-1)]
        self.v_bc = v_bc
        self.n = 0
        self.t = 0.0
        self.a = np.zeros(self.n_nodes) # Acceleration Field
        self.v = np.zeros(self.n_nodes) # Velocity Field
        self.u = np.zeros(self.n_nodes) # Displacement Field
        self.stress = np.zeros(self.n_elems) # Stress Field
        self.strain = np.zeros(self.n_elems) # Strain Field    
        self.dt = safety_Param * self.dx * np.sqrt(self.rho / self.E_S) # Time Step with Safety Parameter
        self.dt_C =  self.C * self.dx * np.sqrt(self.rho / self.E_S) # Time Step at Courant = 1.0

        self.K = np.zeros((self.n_nodes, self.n_nodes)) # Stiffness Matrix
        self.M = np.zeros((self.n_nodes, self.n_nodes)) # Mass Matrix
        self.f_ext = np.zeros(self.n_nodes) # External Force Vector

        self.Lamda = np.zeros((self.n_nodes, self.n_nodes)) # Localised Lagrange Multipliers
        self.B = np.zeros(self.n_nodes) # Boolean Vectors for Extracting Interface DOFs for each domain
        self.L = np.zeros(self.n_nodes) # Boolean Vectors for Extracting Interface DOFs for global acc and disp

        self.beta = 0.25 # Coefficients for the Newmark Scheme
        self.gamma = 0.5

    def compute_mass_matrix(self):
        '''
        Lumped Mass Matrix
        '''
        nodal_mass = 0.5 * self.rho * self.A * self.dx
        for i in range(self.n_nodes):
            if i > 0 and i < self.n_nodes - 1:
                self.M[i, i] = 2 * nodal_mass
            else:
                self.M[i, i] = nodal_mass
        print("Lumped Mass Matrix")
        
    def compute_stiffness_matrix(self):
        '''
        Stiffness Matrix
        '''
        stiffness_element_L = self.E_L * self.A / self.dx
        for i in range((self.n_elems // 2) + 1):
            self.K[i, i] += stiffness_element_L
            if i != (self.n_elems // 2) + 1 - 1:
                self.K[i, i + 1] -= stiffness_element_L
                self.K[i + 1, i] -= stiffness_element_L
            if i > 0 and i < (self.n_elems // 2):
                self.K[i, i] += stiffness_element_L
        stiffness_element_S = self.E_S * self.A / self.dx
        for i in range((self.n_elems // 2) + 1, self.n_nodes):
            self.K[i, i] += stiffness_element_S
            if i != self.n_nodes - 1:
                self.K[i, i + 1] -= stiffness_element_S
                self.K[i + 1, i] -= stiffness_element_S
            if i > 0 and i < self.n_nodes - 1:
                self.K[i, i] += stiffness_element_S
        print("Stiffness Matrix")
        print(self.K)

    def element_update(self):
        self.strain = (np.diff(self.u) / self.dx) 
        for i in range(self.n_elems // 2):
            self.stress[i] = self.strain[i] * self.E_L
        for i in range((self.n_elems // 2) + 1, self.n_elems):
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
        # Pullback Step
        self.u = self.u + (self.dt * self.v) + (beta_1 * (self.dt_C)**2 * self.a) + (beta_2 * (self.dt_C)**2 * a_nC)
        self.a = np.linalg.solve(self.M, self.f_ext - np.dot(self.K, self.u))
        self.a[0] = 0.0
        self.v = self.v + self.dt * ((1 - self.gamma) * self.a + self.gamma * self.a)  
        self.assemble_vbcs(self.t)    
        self.t = self.t + self.dt
        self.n += 1

class Visualise_Monolithic:

    def __init__(self, Large: Domain):

        self.domain = Large
        self.filenames_accel = []
        self.filenames_vel = []
        self.filenames_disp = []
        self.filenames_stress = []

    def plot(self, variable, position, title, xlabel, ylabel, filenames):
        filenames.append(f'FEM1D_{title}{self.domain.n}.png')
        plt.style.use('ggplot')
        plt.plot(position, variable)
        plt.title(title,fontsize=12)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(["Time: " + format(self.domain.t * 1e6, ".1f") + "us"])
        plt.savefig(f'FEM1D_{title}{self.domain.n}.png')
        plt.close()
        
    def plot_accel(self):
        self.plot(self.domain.a, self.domain.position, "Acceleration", "Domain Position (m)", "Acceleration (m/s^2)", self.filenames_accel)

    def plot_vel(self):
        self.plot(self.domain.v, self.domain.position, "Velocity", "Domain Position (m)", "Velocity (m/s)", self.filenames_vel)

    def plot_disp(self):
        self.plot(self.domain.u, self.domain.position, "Displacement", "Domain Position (m)", "Displacement (m)", self.filenames_disp)

    def plot_stress(self):
        self.plot(self.domain.stress, self.domain.midposition, "Stress", "Domain Position (m)", "Stress (Pa)", self.filenames_stress)

    def create_gif(self, gif_name, filenames):
        with imageio.get_writer(gif_name, mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        for filename in set(filenames):
            os.remove(filename)

if __name__ == '__main__':
    # Initialise Domain
    E_L = 0.02 * 10**9 # 0.02GPa
    E_s = 200 * 10**9 # 200GPa
    rho = 8000 # 8000kg/m^3
    length = 50 * 10**-3 # 50mm
    area = 1 # 1m^2
    num_elements = 300
    safety_Parameter = 0.5
    def vel(t): return vbc.velbcSquare(t, length, E_L, rho)
    velboundaryConditions = vbc(list([0]), list([vel]))

    Domain = Domain('CDM', E_L, E_s, rho, length, area, num_elements, safety_Parameter, velboundaryConditions)
    Domain.compute_mass_matrix()
    Domain.compute_stiffness_matrix()

    bar = Visualise_Monolithic(Domain)

    # Integrate over time
    while Domain.t < 0.001:
        Domain.element_update()
        # Domain.integrate_nb()
        Domain.integrate_pfpb()
        print("Time: ", Domain.t)
        if Domain.n % 5000 == 0:
            bar.plot_accel()
            bar.plot_vel()
            bar.plot_disp()
            bar.plot_stress()
    
    bar.create_gif('FEM1DAccel.gif', bar.filenames_accel)
    bar.create_gif('FEM1DVel.gif', bar.filenames_vel)
    bar.create_gif('FEM1DDisp.gif', bar.filenames_disp)
    bar.create_gif('FEM1DStress.gif', bar.filenames_stress)

