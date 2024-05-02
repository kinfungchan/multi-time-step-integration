from SimpleIntegrator import SimpleIntegrator
from BoundaryConditions import VelBoundaryConditions as vbc
from BoundaryConditions import AccelBoundaryConditions as abc
from Stability import Stability
from Sandbox import exportCSV, writeCSV, vHalftoCSV
import numpy as np
import matplotlib.pyplot as plt
import imageio 
import os

"""
This module implements the subcycling algorithm with interface constant acceleration from:

K.F. Chan, N. Bombace, D. Sap, D. Wason, and N. Petrinic (2023),
A Multi-Time Stepping Algorithm for the Modelling of Heterogeneous Structures with Explicit Time Integration, 
I.J. Num. Meth. in Eng., 2023;00:1–6.

"""
class MultiTimeStep:

    """
    Constructor for the subcycling class
    It accepts two domains a large and a small one
    They are both SimpleIntegrators, and they ratio is a non-integer number:
    LARGE     |   SMALL
    *----*----*--*--*--*--*
    """
    def __init__(self, largeDomain: SimpleIntegrator, smallDomain: SimpleIntegrator, stability: Stability):
        
        self.large = largeDomain
        self.small = smallDomain
        # Interface
        self.mass_Gamma = self.large.mass[-1] + self.small.mass[0]
        self.f_int_Gamma = self.large.f_int[-1] + self.small.f_int[0]
        # Time step Ratio Computations
        self.large_tTrial = 0.0
        self.small_tTrial = 0.0
        self.nextTimeStepRatio = 0.0 
        self.currTimeStepRatio = 0.0
        # Reduction Factor Values
        self.large_alpha = 0.0
        self.small_alpha = 0.0
        # Time Step History
        self.steps_L = np.array([0.0])
        self.steps_S = np.array([0.0])
        
        # Stability Calculations
        self.stability = stability

        # Minimum Time Step 
        self.min_dt = np.inf
        self.el_steps = 0

    def calc_timestep_ratios(self):
        self.small_tTrial = self.small.t + self.small.dt
        self.large_tTrial = self.large.t + self.large.dt
        self.currTimeStepRatio = (self.small.t - self.large.t) / (self.large_tTrial - self.large.t)
        self.nextTimeStepRatio = ((self.small.t + self.small.dt) - self.large.t) / (self.large_tTrial - self.large.t)

    def accelCoupling(self): 
        return -self.f_int_Gamma / self.mass_Gamma

    def update_small_domain(self):
        self.small.a_bc.indexes.append(0)
        self.small.a_bc.accelerations.append(self.accelCoupling)
        self.small.single_tstep_integrate()

        self.steps_S = np.append(self.steps_S, self.small.dt)
        self.el_steps += self.small.n_elem
        self.small.assemble_internal()
        self.calc_timestep_ratios()

    def integrate(self):
    
        if ((self.large.t == 0) and (self.small.t == 0)):
            self.small.assemble_internal()
            self.large.assemble_internal()
            self.f_int_Gamma = self.large.f_int[-1] + self.small.f_int[0]
            self.calc_timestep_ratios()

        while (self.nextTimeStepRatio <= 1 or (self.currTimeStepRatio <= 1 and self.nextTimeStepRatio <= 1.000001)):
            # Integrate Small Domain
            self.update_small_domain()
            self.stability.t_small = np.append(self.stability.t_small, self.small.t)

        # Compute Pullback Values
        self.alpha_L = 1 - ((self.large_tTrial - self.small.t)/(self.large_tTrial - self.large.t))
        self.alpha_s = 1 - ((self.small_tTrial - self.large_tTrial)/(self.small_tTrial - self.small.t))

        if (self.alpha_L >= self.alpha_s):
            self.large.dt = self.alpha_L * self.large.dt
        elif (self.alpha_s > self.alpha_L):
            self.small.dt = self.alpha_s * self.small.dt
            self.update_small_domain()
            
        # Integrate Large Domain
        self.large.a_bc.indexes.append(-1)
        self.large.a_bc.accelerations.append(self.accelCoupling)
        self.large.single_tstep_integrate()

        # Enforce continuity
        self.large.u[-1] = self.small.u[0]
        self.large.v[-1] = self.small.v[0]

        self.min_dt = min(self.min_dt, self.small.dt, self.large.dt)
        self.steps_L = np.append(self.steps_L, self.large.dt)
        self.el_steps += self.large.n_elem
        self.stability.calc_drift(self.large.a[-1], self.small.a[0], self.large.v[-1], self.small.v[0], self.large.u[-1], self.small.u[0], self.large.t)
        self.stability.calc_KE(self.large.mass[-1], self.large.v[-1], self.small.mass[0], self.small.v[0])

        self.large.assemble_internal()        
        self.calc_timestep_ratios()

        self.f_int_Gamma = self.large.f_int[-1] + self.small.f_int[0]  

        lm_L, lm_s, a_diff = self.stability.LagrangeMultiplierEquiv(self.large.mass[-1], self.small.mass[0], 
                                                                    self.large.f_int[-1], self.small.f_int[0],
                                                                    -self.f_int_Gamma / self.mass_Gamma)       
        self.stability.lm_L = np.append(self.stability.lm_L, lm_L)
        self.stability.lm_s = np.append(self.stability.lm_s, lm_s)
        self.stability.a_diff = np.append(self.stability.a_diff, a_diff)
        self.stability.calc_Work(lm_L, lm_s, self.large.u[-1], self.small.u[0])
    
class Visualise_MultiTimestep:

    def __init__(self, upd_fullDomain: MultiTimeStep):

        self.updated = upd_fullDomain
        self.filenames_accel = []
        self.filenames_vel = []
        self.filenames_disp = []
        self.filenames_stress = []

    def plot(self, variable_L, variable_S, position_L, position_S, title, xlabel, ylabel, filenames):
        filenames.append(f'FEM1D_{title}{self.updated.large.n}.png')
        plt.style.use('ggplot')
        plt.plot(position_L, variable_L)
        plt.plot([position + self.updated.large.L for position in position_S], variable_S) 
        plt.title(title,fontsize=12)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(["Time: " + format(self.updated.large.t * 1e6, ".1f") + "us"])
        plt.savefig(f'FEM1D_{title}{self.updated.large.n}.png')
        plt.close()

    def plot_accel(self):
        self.plot(self.updated.large.a, self.updated.small.a, self.updated.large.position, self.updated.small.position, "Acceleration", "Domain Position (m)", "Acceleration (m/s^2)", self.filenames_accel)

    def plot_vel(self):
        self.plot(self.updated.large.v, self.updated.small.v, self.updated.large.position, self.updated.small.position,  "Velocity", "Domain Position (m)", "Velocity (m/s)", self.filenames_vel)

    def plot_disp(self):
        self.plot(self.updated.large.u, self.updated.small.u, self.updated.large.position, self.updated.small.position, "Displacement", "Domain Position (m)", "Displacement (m)", self.filenames_disp)

    def plot_stress(self):
        self.plot(self.updated.large.stress, self.updated.small.stress, self.updated.large.midposition, self.updated.small.midposition, "Stress", "Domain Position (m)", "Stress (Pa)", self.filenames_stress)

    def create_gif(self, gif_name, filenames):
        with imageio.get_writer(gif_name, mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        for filename in set(filenames):
            os.remove(filename)

    def plot_time_steps(self):
        x = ['L', 'S']
        n_steps = 10 # Number of Steps to Plot
        data = np.empty((n_steps, 2))
        for i in range(n_steps):
            data[i] = np.array([self.updated.steps_L[i], 
                                self.updated.steps_S[i]])
            
        plt.figure(figsize=(10, 6))
        for i in range(1, n_steps):        
            plt.bar(x, data[i], bottom=np.sum(data[:i], axis=0), color=plt.cm.tab10(i), label=f'Local Step {i}')

        plt.ylabel('Time (s)')
        plt.title('Time Steps taken for New Multi-step')
        plt.legend()
        plt.show()

def newCoupling(vel_csv, stability_plots):
    # Utilise same element size, drive time step ratio with Co.
    nElemLarge = 300
    E_L = 0.02e9
    # E_s = 0.18e9    
    rho = 8000
    E_s = (np.pi/0.02)**2 * rho # Non Integer Time Step Ratio = pi
    # c = sp.c * 1e-8
    # E_s = (c/0.02)**2 * rho # Non Integer Time Step Ratio Speed of Light 
    Courant = 0.5
    Length = 50e-3
    propTime = 1.75 * Length * np.sqrt(rho / E_L)    
    def vel(t): return vbc.velbcSquareWave(t, 2 * Length , E_L, rho)
    accelBCs_L = abc(list(),list())
    accelBCs_s = abc(list(),list())
    upd_largeDomain = SimpleIntegrator("total", E_L, rho, Length, 1, nElemLarge, propTime, vbc([0], [vel]), accelBCs_L, Co=Courant)
    upd_smallDomain = SimpleIntegrator("total", E_s, rho, Length * 2, 1, nElemLarge * 2, propTime, None, accelBCs_s, Co=Courant)
    stability = Stability()
    upd_fullDomain = MultiTimeStep(upd_largeDomain, upd_smallDomain, stability)
    plotfullDomain = Visualise_MultiTimestep(upd_fullDomain)
    
    # Solve Loop
    while(upd_fullDomain.large.t <= 0.0016):
        upd_fullDomain.integrate()
        print("Time: ", upd_fullDomain.large.t)
        if (upd_fullDomain.large.n % 500 == 0):
            plotfullDomain.plot_accel()
            plotfullDomain.plot_vel()
            plotfullDomain.plot_disp()
            plotfullDomain.plot_stress()

        # Export to CSV
        if (vel_csv):
            vHalftoCSV(upd_fullDomain.large.t, upd_fullDomain.large.v, upd_fullDomain.small.v,
                   upd_fullDomain.large.t_prev, upd_fullDomain.large.v_prev, upd_fullDomain.small.v_prev, 
                   upd_fullDomain.large.position, upd_fullDomain.small.position, upd_fullDomain.large.L)

    plotfullDomain.create_gif('Updated_Multi-time-step_accel.gif', plotfullDomain.filenames_accel)
    plotfullDomain.create_gif('Updated_Multi-time-step.gif', plotfullDomain.filenames_vel)
    plotfullDomain.create_gif('Updated_Multi-time-step_disp.gif', plotfullDomain.filenames_disp)
    plotfullDomain.create_gif('Updated_Multi-time-step_stress.gif', plotfullDomain.filenames_stress)

    if (stability_plots):
        stability.plot_LMEquiv()
        stability.plot_drift()
        stability.plot_Work()
        stability.plot_KE()

    # Print Minimum Time Step for Whole Domain
    plotfullDomain.plot_time_steps()
    print("Minimum Time Step for Large Domain: ", upd_fullDomain.min_dt)
    # Print Total Number of Integration Steps on Large 
    print("Number of Integration Steps: ", upd_fullDomain.el_steps)
    # Print First 10 Time Steps on Large and Small
    print("Time Steps: ", upd_fullDomain.steps_L[:10])
    print("Time Steps: ", upd_fullDomain.steps_S[:10])

if __name__ == "__main__":
    newCoupling(True, True) # Export CSV, Stability Plots

    
