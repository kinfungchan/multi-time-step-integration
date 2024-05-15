import numpy as np
from BoundaryConditions import  VelBoundaryConditions as vbc
from BoundaryConditions import  AccelBoundaryConditions as abc
from Visualise import Plot, Animation

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
        self.v_fulldt = np.zeros(self.n_nodes)
        self.a_tilda = np.zeros(self.n_nodes)

    def apply_bulk_viscosity(self, v, dx, rho, E, stress):
        D = (np.diff(v) / dx) # Deformation Gradient            
        c = np.sqrt(E / rho) # Speed of sound
        C1 = 0.06 # Bulk Viscosity Linear Coefficient
        BV_lin = C1 * c * D
        bulk_viscosity_stress =  rho * dx * (-BV_lin)
        # Include bulk viscosity term in stress update
        stress -= bulk_viscosity_stress  # Add bulk viscosity term

    def assemble_internal(self):
        if (self.formulation == "updated"):
            tempdx = [self.position[n]-self.position[n-1] for n in range(1, len(self.position))] # Updated Lagrangian
            self.midposition = [self.position[n] + 0.5 * tempdx[n] for n in range(0, len(self.position)-1)]
            self.rho = self.elMass / tempdx
            self.dt = self.Co * min(tempdx) * np.sqrt(min(self.rho) / self.E) # Updated Lagrangian
            self.strain = (np.diff(self.u) / self.dx) 
            self.stress = self.strain * self.E

            # Bulk Viscosity
            self.apply_bulk_viscosity(self.v, tempdx, self.rho, self.E, self.stress)

            self.f_int[1:-1] = -np.diff(self.stress)
            self.f_int[0] += -self.stress[0]
            self.f_int[-1] += self.stress[-1]

        if (self.formulation == "total"):
            self.strain = (np.diff(self.u) / self.dx) 
            self.stress = self.strain * self.E

            # Bulk Viscosity
            # self.apply_bulk_viscosity(self.v, self.dx, min(self.rho), self.E, self.stress)

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
        self.a_tilda = np.copy(self.a)
        self.assemble_abcs() 
        if self.n == 0:
            self.v += 0.5 * self.a * self.dt
            self.v_fulldt = self.v + 0.5 * self.a * self.dt
        else:
            self.v += self.a * self.dt
            self.v_fulldt = self.v + 0.5 * self.a * self.dt
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

    plot = Plot()
    animate = Animation(plot)

    while tot_bar.t <= tot_bar.tfinal:
        upd_bar.assemble_internal()
        upd_bar.single_tstep_integrate()
        tot_bar.assemble_internal()
        tot_bar.single_tstep_integrate()
        if tot_bar.n % 20 == 0:
            print(f"Time: {tot_bar.t} s")

            animate.save_single_plot(2, [upd_bar.position, tot_bar.position],
                                     [upd_bar.a, tot_bar.a],
                                     "Acceleration", "Domain Position (m)", "Acceleration (m/s^2)",
                                     animate.filenames_accel, upd_bar.n,
                                     ["Updated", "Total"])
            animate.save_single_plot(2, [upd_bar.position, tot_bar.position],
                                     [upd_bar.v, tot_bar.v],
                                     "Velocity", "Domain Position (m)", "Velocity (m/s)",
                                     animate.filenames_vel, upd_bar.n,
                                     ["Updated", "Total"])
            animate.save_single_plot(2, [upd_bar.position, tot_bar.position],
                                     [upd_bar.u, tot_bar.u],
                                     "Displacement", "Domain Position (m)", "Displacement (m)",
                                     animate.filenames_disp, upd_bar.n,
                                     ["Updated", "Total"])
            animate.save_single_plot(2, [upd_bar.midposition, tot_bar.midposition],
                                     [upd_bar.stress, tot_bar.stress],
                                     "Stress", "Domain Position (m)", "Stress (Pa)",
                                     animate.filenames_stress, upd_bar.n,
                                     ["Updated", "Total"])
            animate.save_single_plot(2, [upd_bar.midposition, tot_bar.midposition],
                                     [upd_bar.bulk_viscosity_stress, tot_bar.bulk_viscosity_stress],
                                     "Stress_BV", "Domain Position (m)", "Bulk Viscosity Stress (Pa)",
                                     animate.filenames_bv, upd_bar.n,
                                     ["Updated", "Total"])
    # Energy Balance
    plot.plot(6, [tot_bar.timestamps, upd_bar.timestamps, tot_bar.timestamps,
                  upd_bar.timestamps, tot_bar.timestamps, upd_bar.timestamps], 
                 [tot_bar.kinetic_energy, upd_bar.kinetic_energy, tot_bar.internal_energy,
                  upd_bar.internal_energy, tot_bar.tot_energy, upd_bar.tot_energy], 
                 "Energy Balance", "Time (s)", "Energy (J)", 
                 ["Total KE", "Updated KE", "Total IE", "Updated IE", "Total KE+IE", "Updated KE+IE"],
                 [None, None], [None, None], True)

    animate.save_monolithic_gifs()

if __name__ == "__main__":
    monolithic()
