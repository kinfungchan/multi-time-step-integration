import numpy as np
import matplotlib.pyplot as plt
from literature.BoundaryConditions import  VelBoundaryConditions as vbc
import imageio
import os


"""
In this notebook we look to reimplement the Monolithic time integration for
two-dimensional heterogeneous solids from the following paper


Reference: Park, K. C., Lim, S. J., & Huh, H. (2012). A method for computation of
discontinuous wave propagation in heterogeneous solids: basic algorithm description
and application to one‐dimensional problems. International Journal for Numerical
Methods in Engineering, 91(6), 622-643.

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
    def __init__(self, label, young, density, length, area, num_elements, safety_Param, v_bc: vbc):
        self.label = label
        self.E = young
        self.rho = density
        self.L = length
        self.A = area
        self.n_elems = num_elements
        self.C = 1.0
        self.n_nodes = num_elements + 1
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
        self.dt = safety_Param * self.dx * np.sqrt(self.rho / self.E)    
        self.dt_C =  self.C * self.dx * np.sqrt(self.rho / self.E) # Time Step

        self.K = np.zeros((self.n_nodes, self.n_nodes)) # Stiffness Matrix
        self.M = np.zeros((self.n_nodes, self.n_nodes)) # Mass Matrix
        self.f_ext = np.zeros(self.n_nodes) # External Force Vector

        self.beta = 0.0 # Coefficients for the Newmark Scheme, 0 for CDM
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
        self.strain = (np.diff(self.u) / self.dx)
        self.stress = self.strain * self.E

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
        global pushforward-pullback method

        Time stepping for one domain
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
        theta = 0.0 # 0.5 for Average Displacement , 0 for CDM
        beta_1 = (alpha / 6) * (3 * alpha + theta - (theta * alpha ** 2))
        beta_2 = theta * (alpha / 6) * (alpha ** 2 - 1)

        # Pullback Step
        self.u = self.u + (self.dt * self.v) + (beta_1 * (self.dt_C)**2 * self.a) + (beta_2 * (self.dt_C)**2 * a_nC)
        self.a = np.linalg.solve(self.M, self.f_ext - np.dot(self.K, self.u))
        self.a[0] = 0.0
        self.v = self.v + self.dt * ((1 - self.gamma) * self.a + self.gamma * self.a) # Should this used an old a too?
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
    # Initialise Domains
    # Large Domain
    E_L = 0.02 * 10**9 # 0.02GPa
    rho_L = 8000 # 8000kg/m^3
    length_L = 50 * 10**-3 # 50mm
    area_L = 1 # 1m^2
    num_elements_L = 300
    safety_Param = 0.5
    def vel(t): return vbc.velbcSquare(t, length_L, E_L, rho_L)
    velboundaryConditions = vbc(list([0]), list([vel]))

    Domain_L = Domain('Large', E_L, rho_L, length_L, area_L, num_elements_L, safety_Param, velboundaryConditions)
    Domain_L.compute_mass_matrix()
    Domain_L.compute_stiffness_matrix()

    bar = Visualise_Monolithic(Domain_L)

    # Integrate over time
    while Domain_L.t < 0.001:
        Domain_L.element_update()
        Domain_L.integrate_nb()
        # Domain_L.integrate_pfpb()
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


