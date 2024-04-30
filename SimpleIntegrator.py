import numpy as np
import matplotlib.pyplot as plt
from BoundaryConditions import  VelBoundaryConditions as vbc
from BoundaryConditions import  AccelBoundaryConditions as abc
import imageio
import os
"""
This module Integrates in Time a 1D Domain using the following algorithm 
from Belytschko´s Non  Linear Finite Elements for Continua and Structures
Box 2.5
The element formulation is in the same book Example 2.1

1. Set Initial Conditions
2. get f_n
3. Compute accelerations a_n =  1/M * f_n
4. Update Nodal Velocities v_(n+0.5) = v_(n+0.5-alpha) + alpha*dt*a_n
    alpha is 0.5 when n = 0 and 1 otherwise
5. Enforce essential boundary conditions on node I. v_(n + 0.5) = v_bc(x, t_n+0.5)
6. Update displacements. u_(n+1) = u_n + v_(n + 0.5) * dT
7. Update counter and time 
8. Output if not over go to 2

Module get f

1. Gather element nodal displacements and velocities u_n and v_n+0.5
2. if n = 0 go to 5
3. compute measure of deformation
4. compute stress by constitutive equation
5. Compute internal nodal forces by relevant equation
6. Compute external nodal force and f = fext - fint
7. Scatter to global matrix
"""


class SimpleIntegrator:
    """
    Constructor for the Simple integrator
    :param formulation: Total or Updated Lagrangian
    :param young:     Young´s modulus
    :param density:   Density
    :param length:     length of bar
    :param A:     Area of bar
    :param num_elems: #elements in bar
    :param tfinal: final time of computation
    :param Co: Courant number
    :param v_bc: velocity boundary condition
    """

    def __init__(self, formulation, young, density, length, A, num_elems, tfinal, v_bc: vbc, a_bc: abc, Co):
        self.formulation = formulation
        self.E = young
        # self.rho = density
        self.L = length
        self.n_elem = num_elems
        self.tfinal = tfinal
        self.Co = Co
        self.n_nodes = num_elems + 1
        self.position, self.dx = np.linspace(0, length, self.n_nodes, retstep=True)
        self.midposition = [self.position[n] + 0.5 * self.dx for n in range(0, len(self.position)-1)]
        self.v_bc = v_bc
        self.a_bc = a_bc
        self.n = 0
        self.t = 0
        self.a = np.zeros(self.n_nodes)
        self.v = np.zeros(self.n_nodes)
        self.u = np.zeros(self.n_nodes)
        self.rho = np.ones(self.n_elem) * density
        self.stress = np.zeros(self.n_elem)
        self.strain = np.zeros(self.n_elem)
        self.bulk_viscosity_stress = np.zeros(self.n_elem)
        self.f_int = np.zeros(self.n_nodes)
        self.dt = Co * self.dx * np.sqrt(min(self.rho) / self.E)
        nodalMass = 0.5 * min(self.rho) * self.dx
        self.mass = np.ones(self.n_nodes) * nodalMass
        self.mass[1:-1] *= 2
        self.elMass = np.ones(self.n_elem) * 2 * nodalMass
        self.kinetic_energy = []
        self.internal_energy = []
        self.tot_energy = []
        self.timestamps = []
        self.a_prev = np.zeros(self.n_nodes)
        self.v_prev = np.zeros(self.n_nodes)
        self.u_prev = np.zeros(self.n_nodes)
        self.f_int_prev = np.zeros(self.n_nodes)
        self.t_prev = 0


    def assemble_internal(self):
        if (self.formulation == "updated"):
            tempdx = [self.position[n]-self.position[n-1] for n in range(1, len(self.position))] # Updated Lagrangian
            self.midposition = [self.position[n] + 0.5 * tempdx[n] for n in range(0, len(self.position)-1)]
            self.rho = self.elMass / tempdx
            self.dt = self.Co * min(tempdx) * np.sqrt(min(self.rho) / self.E) # Updated Lagrangian
            self.strain = (np.diff(self.u) / self.dx) 
            self.stress = self.strain * self.E

            # Bulk Viscosity
            D = (np.diff(self.v) / tempdx) # Deformation Gradient            
            c = np.sqrt(self.E / self.rho) # Speed of sound
            C0 = 0.0 # Bulk Viscosity Quadratic Coefficient
            C1 = 0.06 # Bulk Viscosity Linear Coefficient
            BV_quad = C0 * self.dx * D**2
            BV_lin = C1 * c * D
            self.bulk_viscosity_stress =  self.rho * tempdx * (BV_quad - BV_lin)
            # Include bulk viscosity term in stress update
            self.stress -= self.bulk_viscosity_stress  # Add bulk viscosity term

            self.f_int[1:-1] = -np.diff(self.stress)
            self.f_int[0] += -self.stress[0]
            self.f_int[-1] += self.stress[-1]

        if (self.formulation == "total"):
            self.strain = (np.diff(self.u) / self.dx) 
            self.stress = self.strain * self.E

            # Bulk Viscosity
            D = (np.diff(self.v) / self.dx) # Deformation Gradient            
            c = np.sqrt(self.E / min(self.rho)) # Speed of sound
            C0 = 0.0 # Bulk Viscosity Quadratic Coefficient
            C1 = 0.06 # Bulk Viscosity Linear Coefficient
            BV_quad = C0 * self.dx * D**2
            BV_lin = C1 * c * D
            self.bulk_viscosity_stress =  min(self.rho) * self.dx * (BV_quad - BV_lin)
            # Include bulk viscosity term in stress update
            self.stress -= self.bulk_viscosity_stress  # Add bulk viscosity term

            self.f_int[1:-1] = -np.diff(self.stress)
            self.f_int[0] += -self.stress[0]
            self.f_int[-1] += self.stress[-1]

    def assemble_vbcs(self, t):
        if (self.v_bc):
            for counter in range(0, len(self.v_bc.indexes)):
                self.v[self.v_bc.indexes[counter]] = self.v_bc.velocities[counter](t)

    def assemble_abcs(self):
        if (self.a_bc):
            for counter in range(0, len(self.a_bc.indexes)):
                self.a[self.a_bc.indexes[counter]] = self.a_bc.accelerations[counter]()

    def save_prev(self):
        self.a_prev, self.v_prev, self.u_prev = np.copy(self.a), np.copy(self.v), np.copy(self.u)
        self.f_int_prev = np.copy(self.f_int)
        self.t_prev = np.copy(self.t)

    def single_tstep_integrate(self):
        self.save_prev()
        self.a = -self.f_int / self.mass
        self.assemble_abcs() 
        if self.n == 0:
            self.v += 0.5 * self.a * self.dt
        else:
            self.v += self.a * self.dt
        self.assemble_vbcs(self.t + 0.5 * self.dt)
        self.u += self.v * self.dt
        if (self.formulation == "updated"):
            self.position += self.u # Updated Lagrangian
        self.n += 1
        self.compute_energy()
        self.t += self.dt
        self.f_int.fill(0)

    def compute_energy(self):
        self.timestamps.append(self.t)
        ke = 0
        ie = 0
        for i in range(self.n_nodes):
            ke += 0.5 * self.mass[i] * self.v[i]**2
        for j in range(self.n_elem):
            ie += ((0.5 * self.stress[j]**2) / self.E) * self.dx
        self.kinetic_energy.append(ke)
        self.internal_energy.append(ie)
        self.tot_energy.append(ke+ie)
        

class Visualise_Monolithic:

    def __init__(self, totalLagrangian: SimpleIntegrator, updatedLagrangian: SimpleIntegrator):

        self.total = totalLagrangian
        self.updated = updatedLagrangian
        self.filenames_accel = []
        self.filenames_vel = []
        self.filenames_disp = []
        self.filenames_stress = []
        self.filenames_bulk_viscosity_stress = []

    def plot(self, totVariable, updVariable, totPosition, updPosition, title, xlabel, ylabel, filenames):
        filenames.append(f'FEM1D_{title}{self.total.n}.png')
        plt.style.use('ggplot')
        plt.plot(totPosition, totVariable, "--")
        plt.plot(updPosition, updVariable)
        plt.title(title,fontsize=12)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend([f"Tot Lagr at T_t = {self.total.t:.8f}", f"Upd Lagr at T_l = {self.updated.t:.8f}"])
        plt.savefig(f'FEM1D_{title}{self.total.n}.png')
        plt.close()

    def plot_accel(self):
        self.plot(self.total.a, self.updated.a, self.total.position, self.updated.position, "Acceleration", "Domain Position (m)", "Acceleration (m/s^2)", self.filenames_accel)

    def plot_vel(self):
        self.plot(self.total.v, self.updated.v, self.total.position, self.updated.position, "Velocity", "Domain Position (m)", "Velocity (m/s)", self.filenames_vel)

    def plot_disp(self):
        self.plot(self.total.u, self.updated.u, self.total.position, self.updated.position, "Displacement", "Domain Position (m)", "Displacement (m)", self.filenames_disp)

    def plot_stress(self):
        self.plot(self.total.stress, self.updated.stress, self.total.midposition, self.updated.midposition, "Element Stress", "Domain Position (m)", "Stress (Pa)", self.filenames_stress)

    def plot_bulk_viscosity_stress(self):
        self.plot(self.total.bulk_viscosity_stress, self.updated.bulk_viscosity_stress, self.total.midposition, self.updated.midposition, "Element Bulk Viscosity Stress", "Domain Position (m)", "Stress (Pa)", self.filenames_bulk_viscosity_stress)

    def plot_energy(self):
        plt.style.use('ggplot')
        plt.locator_params(axis='both', nbins=4)
        plt.plot(self.total.timestamps, self.total.kinetic_energy, "--")
        plt.plot(self.updated.timestamps, self.updated.kinetic_energy)
        plt.plot(self.total.timestamps, self.total.internal_energy, "--")
        plt.plot(self.updated.timestamps, self.updated.internal_energy)
        plt.plot(self.total.timestamps, self.total.tot_energy, "--")
        plt.plot(self.updated.timestamps, self.updated.tot_energy)
        plt.title(f"Elastic Energies for a Half Sine Excitation",fontsize=12)
        plt.xlabel("Time (s)")
        plt.ylabel("Energy (N)")
        plt.legend([f"Total Lagrangian KE", "Updated Lagrangian KE","Total Lagrangian IE", "Updated Lagrangian","Total Tot Lagrangian Total Energy", "Updated Tot Lagrangian Total Energy"])
        plt.savefig(f'FEM1D_enbal.png')
        plt.close()

    def create_gif(self, gif_name, filenames):
        with imageio.get_writer(gif_name, mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        for filename in set(filenames):
            os.remove(filename)

"""
Example from N.Bombace Thesis 2018

Model now matches DARCoMS 1D Model 

In this example we use the simple integrator to simulate a velocity applied BC
on the first node of the mesh

"""
def monolithic():
    n_elem = 300
    E = 0.02 * 10**9
    rho = 8000
    L = 50 * 10**-3
    propTime = 0.5*L * np.sqrt(rho / E)
    # def vel(t): return vbc.velbc(t, L, E, rho)
    def vel(t): return vbc.velbcSquareWave(t, L, E, rho)
    velboundaryConditions = vbc(list([0]), list([vel]))
    tot_formulation = "total"
    upd_formulation = "updated"
    tot_bar = SimpleIntegrator(tot_formulation, E, rho, L, 1, n_elem, 2*propTime, velboundaryConditions, None, Co=0.9)
    upd_bar = SimpleIntegrator(upd_formulation, E, rho, L, 1, n_elem, 2*propTime, velboundaryConditions, None, Co=0.9)
    bar = Visualise_Monolithic(tot_bar, upd_bar) 
    while tot_bar.t <= tot_bar.tfinal:
        upd_bar.assemble_internal()
        upd_bar.single_tstep_integrate()
        tot_bar.assemble_internal()
        tot_bar.single_tstep_integrate()
        if tot_bar.n % 20 == 0:
            print(f"Time: {tot_bar.t} s")
            bar.plot_accel()
            bar.plot_vel()
            bar.plot_disp()
            bar.plot_stress()
            bar.plot_bulk_viscosity_stress()
    bar.plot_energy()
    # The evolution of Velocity, Displacement and Stress is plotted in the following gifs
    bar.create_gif('Updated_and_Total_1DFEM_accel.gif', bar.filenames_accel)
    bar.create_gif('Updated_and_Total_1DFEM_vel.gif', bar.filenames_vel)
    bar.create_gif('Updated_and_Total_1DFEM_disp.gif', bar.filenames_disp)
    bar.create_gif('Updated_and_Total_1DFEM_stress.gif', bar.filenames_stress)
    bar.create_gif('Updated_and_Total_1DFEM_bulk_viscosity_stress.gif', bar.filenames_bulk_viscosity_stress)

if __name__ == "__main__":
    monolithic()
