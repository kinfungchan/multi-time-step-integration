import numpy as np
import matplotlib.pyplot as plt
from BoundaryConditions import  VelBoundaryConditions as vbc
import imageio
import os

"""
In this notebook we look to reimplement the multistep time integration for 
two-dimensional heterogeneous solids from the following paper

Reference: Cho, S. S., Kolman, R., González, J. A., & Park, K. C. (2019).
Explicit multistep time integration for discontinuous elastic stress wave
propagation in heterogeneous solids. International Journal for Numerical
Methods in Engineering, 118(5), 276-302

This script attends to the Numerical Example in Section 4.1 of the paper

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
    def __init__(self, label, young, density, length, area, num_elements, Courant, v_bc: vbc):
        self.label = label
        self.E = young
        self.rho = density
        self.length = length
        self.A = area
        self.n_elems = num_elements
        self.C = Courant
        self.n_nodes = num_elements + 1
        self.position, self.dx = np.linspace(0, length, self.n_nodes, retstep=True)
        self.midposition = [self.position[n] + 0.5 * self.dx for n in range(0, len(self.position)-1)]
        self.v_bc = v_bc
        self.n = 0
        self.t = 0.0
        self.a = np.zeros(self.n_nodes) # Acceleration Field

        self.a_n = np.zeros(self.n_nodes) 
        self.a_n1 = np.zeros(self.n_nodes)        

        self.v = np.zeros(self.n_nodes) # Velocity Field
        self.u = np.zeros(self.n_nodes) # Displacement Field
        self.stress = np.zeros(self.n_elems) # Stress Field
        self.strain = np.zeros(self.n_elems) # Strain Field        
        self.dt_C =  self.C * self.dx * np.sqrt(self.rho / self.E) # Time Step

        self.K = np.zeros((self.n_nodes, self.n_nodes)) # Stiffness Matrix
        self.M = np.zeros((self.n_nodes, self.n_nodes)) # Mass Matrix
        self.f_ext = np.zeros(self.n_nodes) # External Force Vector

        self.Lamda = np.zeros((self.n_nodes, self.n_nodes)) # Localised Lagrange Multipliers
        self.B = np.zeros(self.n_nodes) # Boolean Vectors for Extracting Interface DOFs for each domain
        self.L = np.zeros(self.n_nodes) # Boolean Vectors for Extracting Interface DOFs for global acc and disp

        self.alpha = 0.0 # Ratio of Time Step and Critical Time Step
        self.beta = 0.0 # Coefficients for the Newmark Scheme
        self.beta2 = 0.0 
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
        print(self.M)
        
    def compute_stiffness_matrix(self):
        '''
        Stiffness Matrix
        '''
        stiffness_element = self.E * self.A / self.dx
        for i in range(self.n_nodes):
            self.K[i, i] += stiffness_element
            if i != self.n_nodes - 1:
                self.K[i, i + 1] -= stiffness_element
                self.K[i + 1, i] -= stiffness_element
            if i > 0 and i < self.n_nodes - 1:
                self.K[i, i] += stiffness_element
        print("Stiffness Matrix")
        print(self.K)

    def element_update(self):
        self.stress = np.zeros(self.n_elems)
        self.strain = np.zeros(self.n_elems)
        self.strain = (np.diff(self.u) / self.dx)
        self.stress = self.E * self.strain

    def assemble_vbcs(self, t):
        if (self.v_bc):
            for counter in range(0, len(self.v_bc.indexes)):
                self.v[self.v_bc.indexes[counter]] = self.v_bc.velocities[counter](t)

    def integrate(self):
        """
        Function to solve discretised structural equation of motion using the Newmark Beta Method

        """
        self.assemble_vbcs(self.t)
        if self.t == 0:
            self.a_n = np.linalg.solve(self.M, self.f_ext - np.dot(self.K, self.u))
        else:
            self.a_n = self.a_n1

        # Compute Displacement 
        u_n1 = self.u + (self.dt_C * self.v) + (0.5 * self.dt_C**2 * (((1 - 2 * self.beta) * self.a_n) + (2 * self.beta * self.a_n1)))
        # Compute Acceleration
        self.a_n1 = np.linalg.solve(self.M, self.f_ext - np.dot(self.K, u_n1)) 
        # Compute Velocity
        v_n1 = self.v + ((1 - self.gamma) * self.dt_C * self.a_n) + (self.gamma * self.dt_C * self.a_n1)   

        # Update State Variables
        self.v = v_n1    
        self.u = u_n1
        self.t = self.t + self.dt_C
        self.n += 1

class Visualise_Monolithic:

    def __init__(self, Large: Domain):

        self.domain = Large
        self.filenames_vel = []
        self.filenames_disp = []
        self.filenames_stress = []

    def plot_vel(self):
        self.filenames_vel.append(f'FEM1D_vel{self.domain.n}.png')
        plt.style.use('ggplot')
        plt.plot(self.domain.position, self.domain.v)
        plt.title(f"Graph of Velocity against Position for a Square Wave Excitation",fontsize=12)
        plt.xlabel("Domain Position (mm)")
        plt.ylabel("Velocity (mm/ms)")
        plt.savefig(f'FEM1D_vel{self.domain.n}.png')
        plt.close()

    def plot_disp(self):
        self.filenames_disp.append(f'FEM1D_disp{self.domain.n}.png')
        plt.style.use('ggplot')
        plt.plot(self.domain.position, self.domain.u)
        plt.title(f"Graph of Displacement against Position for a Square Wave Excitation",fontsize=12)
        plt.xlabel("Domain Position (mm)")
        plt.ylabel("Displacement (mm)")
        plt.savefig(f'FEM1D_disp{self.domain.n}.png')
        plt.close()

    def plot_stress(self):
        self.filenames_stress.append(f'FEM1D_stress{self.domain.n}.png')
        plt.style.use('ggplot')
        plt.plot(self.domain.midposition, self.domain.stress)
        plt.title(f"Element Stress for a Square Wave Excitation",fontsize=12)
        plt.xlabel("Domain Position (mm)")
        plt.ylabel("Stress (GPa)")
        plt.savefig(f'FEM1D_stress{self.domain.n}.png')
        plt.close()

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
    rho_L = 8000 # 8000kg/m^3
    length_L = 50 * 10**-3 # 50mm
    area_L = 1 # 1m^2
    num_elements_L = 300
    Courant_L = 0.5
    def vel(t): return vbc.velbcSquare(t, length_L, E_L, rho_L)
    velboundaryConditions = vbc(list([0]), list([vel]))

    Domain_L = Domain('Large', E_L, rho_L, length_L, area_L, num_elements_L, Courant_L, velboundaryConditions)
    Domain_L.compute_mass_matrix()
    Domain_L.compute_stiffness_matrix()

    bar = Visualise_Monolithic(Domain_L)

    # Integrate over time
    while Domain_L.t < 0.001:
        Domain_L.element_update()
        Domain_L.integrate()
        print("Time: ", Domain_L.t)
        if Domain_L.n % 10 == 0:
            bar.plot_vel()
            bar.plot_disp()
            bar.plot_stress()
    
    bar.create_gif('FEM1DVel.gif', bar.filenames_vel)
    bar.create_gif('FEM1DDisp.gif', bar.filenames_disp)
    bar.create_gif('FEM1DStress.gif', bar.filenames_stress)

    # # Small Domain
    # E_S = 200 * 10**9 # 200GPa
    # rho_S = 8000 # 8000kg/m^3
    # length_S = 50 * 10**-3 # 50mm
    # num_elements_S = 300
    # Courant_S = 0.5

