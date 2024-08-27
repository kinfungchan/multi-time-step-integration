import numpy as np
from boundaryConditions.BoundaryConditions import  VelBoundaryConditions as vbc
from boundaryConditions.BoundaryConditions import  AccelBoundaryConditions as abc
from database import History
from utils.Visualise import Plot, Animation

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
        # Each element has its own Young´s Modulus and Density
        if isinstance(young, (list, np.ndarray)):
            self.E = np.array(young)
        else:
            self.E = np.full(num_elems, young)        
        if isinstance(density, (list, np.ndarray)):
            self.rho = np.array(density)
        else:
            self.rho = np.full(num_elems, density)
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
        self.stress = np.zeros(self.n_elem)
        self.strain = np.zeros(self.n_elem)
        self.bulk_viscosity_stress = np.zeros(self.n_elem)
        self.f_int = np.zeros(self.n_nodes)
        self.dt = Co * min(self.dx * np.sqrt(self.rho / self.E))
        self.dt_0 = self.dt
        self.dt_min_fac = 0.01
        nodalMass = 0.5 * np.concatenate(([self.rho[0]], self.rho[:-1] + self.rho[1:], [self.rho[-1]])) * self.dx
        self.mass = np.ones(self.n_nodes) * nodalMass
        self.elMass = self.mass[:-1] + self.mass[1:]
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
        D = (np.diff(v) / dx)      
        c = np.sqrt(E / rho) 
        C1 = 0.06 
        BV_lin = C1 * c * D
        self.bulk_viscosity_stress =  rho * dx * (-BV_lin)
        stress -= self.bulk_viscosity_stress 

    def assemble_internal(self):
        if (self.formulation == "updated"):
            tempdx = [self.position[n]-self.position[n-1] for n in range(1, len(self.position))] 
            self.midposition = [self.position[n] + 0.5 * tempdx[n] for n in range(0, len(self.position)-1)]
            self.rho = self.elMass / tempdx
            # Check dt does not drop below 1% of initial value
            # Avoid computing sqrt if min(self.rho)
            if ((self.Co * min(tempdx) * np.sqrt(min(self.rho) / self.E)) < self.dt_min_fac * self.dt_0):
                self.dt = self.dt_min_fac * self.dt_0
            else:
                self.dt = self.Co * min(tempdx * np.sqrt(self.rho / self.E))  
            self.strain = (np.diff(self.u) / self.dx) 
            self.stress = self.strain * self.E

            self.apply_bulk_viscosity(self.v, tempdx, self.rho, self.E, self.stress)

            self.f_int[1:-1] = -np.diff(self.stress)
            self.f_int[0] += -self.stress[0]
            self.f_int[-1] += self.stress[-1]

        if (self.formulation == "total"):
            self.strain = (np.diff(self.u) / self.dx) 
            self.stress = self.strain * self.E

            self.apply_bulk_viscosity(self.v, self.dx, min(self.rho), self.E, self.stress)

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
            self.position += self.u
        self.n += 1
        self.t += self.dt
        self.f_int.fill(0)

def monolithic():
    n_elem = 900
    rho = 8000
    E_L = 0.02 * 10**9  # 0.02 GPa
    # E_s = 200.0 * 10**9  # 200.0 GPa
    E_s = (np.pi/0.02)**2 * rho # Non Integer Time Step Ratio = pi
    L = 150 * 10**-3 # 100mm
    # Initialise with default material properties
    young = np.full(n_elem, E_L)
    density = np.full(n_elem, rho)
    # Overwrite last 150 elements with E_s for High Heterogeneity
    young[-600:] = E_s

    propTime = 0.5 * L * np.sqrt(rho / E_L)
    def vel(t): return vbc.velbcSquare(t, 2 * (L / 3) , E_L, rho)
    velboundaryConditions = vbc(list([0]), list([vel]))
    tot_formulation = "total"
    tot_bar = SimpleIntegrator(tot_formulation, young, density, L, 1, n_elem, propTime, velboundaryConditions, None, Co=0.9)

    # Intialise History
    hst = History(tot_bar.position, tot_bar.n_nodes, tot_bar.n_elem) 
    
    # Initilise Plotting
    plot = Plot()
    animate = Animation(plot)

    while tot_bar.t <= 0.0016:
        tot_bar.assemble_internal()
        tot_bar.single_tstep_integrate()

        # History Data
        hst.append_timestep(tot_bar.t, tot_bar.position,
                            tot_bar.a, tot_bar.v, tot_bar.u, 
                            tot_bar.stress, tot_bar.strain)
        print("Time: ", tot_bar.t)
        # Plotting and Saving Figures
        if (tot_bar.n % 200 == 0): # Determine frequency of Output Plots
            print("Time: ", tot_bar.t)
            animate.save_single_plot(1, [tot_bar.position],
                                     [tot_bar.a],
                                     "Acceleration", "Domain Position (m)", "Acceleration (m/s^2)",
                                     [None, None], [None, None],
                                     animate.filenames_accel, tot_bar.n,
                                     ["Mono"])
            animate.save_single_plot(1, [tot_bar.position],
                                     [tot_bar.v],
                                     "Velocity", "Domain Position (m)", "Velocity (m/s)",
                                     [None, None], [-0.015, 0.015],
                                     animate.filenames_vel, tot_bar.n,
                                     ["Mono"])
            animate.save_single_plot(1, [tot_bar.position],
                                     [tot_bar.u],
                                     "Displacement", "Domain Position (m)", "Displacement (m)",
                                     [None, None], [None, None],
                                     animate.filenames_disp, tot_bar.n,
                                     ["Mono"])
            animate.save_single_plot(1, [tot_bar.midposition],
                                     [tot_bar.stress],
                                     "Stress", "Domain Position (m)", "Stress (Pa)",
                                     [None, None], [None, None],
                                     animate.filenames_stress, tot_bar.n,
                                     ["Mono"])
    animate.save_MTS_gifs("Monolithic")

    # Write History to CSV
    hst.write_to_csv("Mono_Conv900Elem_Co09")



