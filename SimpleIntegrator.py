import numpy as np
import matplotlib.pyplot as plt
from BoundaryConditions import  BoundaryConditions as bc
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

    def __init__(self, formulation, young, density, length, A, num_elems, tfinal, v_bc: bc, Co=1.0, stressComputer=None, intForcesComputer = None):
        self.formulation = formulation
        self.E = young
        self.rho = density
        self.L = length
        self.n_elem = num_elems
        self.tfinal = tfinal
        self.Co = Co
        self.n_nodes = num_elems + 1
        self.position, self.dx = np.linspace(0, length, self.n_nodes, retstep=True)
        self.midposition = [self.position[n] + 0.5 * self.dx for n in range(0, len(self.position)-1)]
        self.v_bc = v_bc
        self.n = 0
        self.t = 0
        self.v = np.zeros(self.n_nodes)
        self.u = np.zeros(self.n_nodes)
        self.stress = np.zeros(self.n_elem)
        self.strain = np.zeros(self.n_elem)
        self.f_int = np.zeros(self.n_nodes)
        self.dt = Co * self.dx * np.sqrt(self.rho / self.E)
        nodalMass = 0.5 * self.rho * self.dx
        self.mass = np.ones(self.n_nodes) * nodalMass
        self.mass[1:-1] *= 2
        self.stressComputer = stressComputer
        self.intForcesComputer=None

    def assemble_internal(self):
        if (self.formulation == "updated"):
            tempdx = [self.position[n]-self.position[n-1] for n in range(1, len(self.position))] # Updated Lagrangian
            self.midposition = [self.position[n] + 0.5 * tempdx[n] for n in range(0, len(self.position)-1)]
            self.dt = self.Co * min(tempdx) * np.sqrt(self.rho / self.E) # Updated Lagrangian
            self.strain = np.zeros(self.n_elem)
            if (self.stressComputer):
                self.stress = self.stressComputer()
            else:
                self.strain = (np.diff(self.v) / tempdx) # Strain Measure is Rate of Deformation
                self.stress += self.strain * self.dt * self.E # Updated Lagrangian

            self.f_int[1:-1] = -np.diff(self.stress)
            self.f_int[0] += -self.stress[0]
            self.f_int[-1] += self.stress[-1]

        if (self.formulation == "total"):
            self.stress = np.zeros(self.n_elem)
            self.strain = np.zeros(self.n_elem)
            if (self.stressComputer):
                self.stress = self.stressComputer()
            else:
                self.strain = (np.diff(self.u) / self.dx) 
                self.stress = self.strain * self.E

            self.f_int[1:-1] = -np.diff(self.stress)
            self.f_int[0] += -self.stress[0]
            self.f_int[-1] += self.stress[-1]

    def assemble_bcs(self, t):
        if (self.v_bc) :
            for counter in range(0, len(self.v_bc.indexes)):
                self.v[self.v_bc.indexes[counter]] = self.v_bc.velocities[counter](t)

    def single_tstep_integrate(self):
        a = -self.f_int / self.mass
        if self.n == 0:
            self.v += 0.5 * a * self.dt
        else:
            self.v += a * self.dt
        self.assemble_bcs(self.t + 0.5 * self.dt)
        self.u += self.v * self.dt
        if (self.formulation == "updated"):
            self.position += self.u # Updated Lagrangian
        self.n += 1
        self.t += self.dt
        self.f_int.fill(0)

class Plotting:

    def __init__(self, totalLagrangian: SimpleIntegrator, updatedLagrangian: SimpleIntegrator):

        self.total = totalLagrangian
        self.updated = updatedLagrangian
        self.filenames_vel = []
        self.filenames_disp = []
        self.filenames_stress = []

    def plot_vel(self):
        self.filenames_vel.append(f'FEM1D_vel{self.total.n}{self.updated.n}.png')
        plt.style.use('ggplot')
        plt.plot(self.total.position, self.total.v, "--")
        plt.plot(self.updated.position, self.updated.v)
        plt.title(f"Graph of Velocity against Position for a Half Sine Excitation",fontsize=12)
        plt.xlabel("Domain Position (mm)")
        plt.ylabel("Velocity (mm/ms)")
        plt.legend([f"Total Lagrangian", "Updated Lagrangian"])
        plt.savefig(f'FEM1D_vel{self.total.n}{self.updated.n}.png')
        plt.close()

    def plot_disp(self):
        self.filenames_disp.append(f'FEM1D_disp{self.total.n}{self.updated.n}.png')
        plt.style.use('ggplot')
        plt.plot(self.total.position, self.total.u, "--")
        plt.plot(self.updated.position, self.updated.u)
        plt.title(f"Graph of Displacement against Position for a Half Sine Excitation",fontsize=12)
        plt.xlabel("Domain Position (mm)")
        plt.ylabel("Displacement (mm)")
        plt.legend([f"Total Lagrangian", "Updated Lagrangian"])
        plt.savefig(f'FEM1D_disp{self.total.n}{self.updated.n}.png')
        plt.close()

    def plot_stress(self):
        self.filenames_stress.append(f'FEM1D_stress{self.total.n}{self.updated.n}.png')
        plt.style.use('ggplot')
        plt.plot(self.total.midposition, self.total.stress, "--")
        plt.plot(self.updated.midposition, self.updated.stress)
        plt.title(f"Element Stress for a Half Sine Excitation (Compression)",fontsize=12)
        plt.xlabel("Domain Position (mm)")
        plt.ylabel("Stress (GPa)")
        plt.legend([f"Total Lagrangian", "Updated Lagrangian"])
        plt.savefig(f'FEM1D_stress{self.total.n}{self.updated.n}.png')
        plt.close()


"""
Example from Bombace Thesis

Model now matches DARCoMS 1D Model 

In this example we use the simple integrator to simulate a velocity applied BC
on the first node of the mesh

"""

def velbc(t, L, E, rho):
    sinePeriod = (L / 2) * np.sqrt(rho/E)
    freq = 1 / sinePeriod
    if t >= sinePeriod * 0.5:
        return 0
    else:
        return 0.01 * np.sin(2 * np.pi * freq * t)

if __name__ == "__main__":
    n_elem = 375
    E = 207
    rho = 7.83e-6
    L = 1
    propTime = 0.5*L * np.sqrt(rho / E)
    def vel(t): return velbc(t, L, E, rho)
    boundaryConditions = bc(list([0]), list([vel]))
    tot_formulation = "total"
    upd_formulation = "updated"
    upd_bar = SimpleIntegrator(upd_formulation, E, rho, L, 1, n_elem, 2*propTime, boundaryConditions, Co=1.0)
    tot_bar = SimpleIntegrator(tot_formulation, E, rho, L, 1, n_elem, 2*propTime, boundaryConditions, Co=1.0)
    bar = Plotting(tot_bar, upd_bar)
    while upd_bar.t <= upd_bar.tfinal:
        upd_bar.assemble_internal()
        upd_bar.single_tstep_integrate()
        tot_bar.assemble_internal()
        tot_bar.single_tstep_integrate()
        bar.plot_vel()
        bar.plot_disp()
        bar.plot_stress()

    with imageio.get_writer('Updated_and_Total_1DFEM_vel.gif', mode='I') as writer:
        for filename in bar.filenames_vel:
            image = imageio.imread(filename)
            writer.append_data(image)
    for filename in set(bar.filenames_vel):
        os.remove(filename)

    with imageio.get_writer('Updated_and_Total_1DFEM_disp.gif', mode='I') as writer:
        for filename in bar.filenames_disp:
            image = imageio.imread(filename)
            writer.append_data(image)
    for filename in set(bar.filenames_disp):
        os.remove(filename)

    with imageio.get_writer('Updated_and_Total_1DFEM_stress.gif', mode='I') as writer:
        for filename in bar.filenames_stress:
            image = imageio.imread(filename)
            writer.append_data(image)
    for filename in set(bar.filenames_stress):
        os.remove(filename)