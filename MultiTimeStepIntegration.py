from SimpleIntegrator import SimpleIntegrator
from BoundaryConditions import VelBoundaryConditions as vbc
from BoundaryConditions import AccelBoundaryConditions as abc
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
    def __init__(self, largeDomain: SimpleIntegrator, smallDomain: SimpleIntegrator):
        
        self.large = largeDomain
        self.small = smallDomain
        self.large.mass[-1] += self.small.mass[0] 
        # Time step Ratio Computations
        self.large_tTrial = 0.0
        self.small_tTrial = 0.0
        self.nextTimeStepRatio = 0.0 
        self.currTimeStepRatio = 0.0
        # Pullback Values
        self.large_alpha = 0.0
        self.small_alpha = 0.0
        # Time Step History
        self.steps_L = np.array([0.0])
        self.steps_Gamma = np.array([0.0])
        self.steps_S = np.array([0.0])
        # Calculate Drifting
        self.a_drift = np.array([0.0])
        self.u_drift = np.array([0.0])
        self.v_drift = np.array([0.0])
        self.t_sync = np.array([0.0])

    def calc_timestep_ratios(self):
        self.small_tTrial = self.small.t + self.small.dt
        self.large_tTrial = self.large.t + self.large.dt
        self.currTimeStepRatio = (self.small.t - self.large.t) / (self.large_tTrial - self.large.t)
        self.nextTimeStepRatio = ((self.small.t + self.small.dt) - self.large.t) / (self.large_tTrial - self.large.t)

    def update_small_domain(self):
        largeForce = self.large.f_int[-1]
        largeMass = self.large.mass[-1]
        def accelCoupling(): return -largeForce / largeMass

        self.small.a_bc.indexes.append(0)
        self.small.a_bc.accelerations.append(accelCoupling)
        self.steps_Gamma = np.append(self.steps_Gamma, self.small.dt)

        self.small.single_tstep_integrate()
        self.steps_S = np.append(self.steps_S, self.small.dt)
        self.small.assemble_internal()
        self.calc_timestep_ratios()

    def integrate(self):
    
        if ((self.large.t == 0) and (self.small.t == 0)):
            self.small.assemble_internal()
            self.large.assemble_internal()
            self.large.f_int[-1] += self.small.f_int[0]
            self.calc_timestep_ratios()

        while (self.nextTimeStepRatio <= 1 or (self.currTimeStepRatio <= 1 and self.nextTimeStepRatio <= 1.000001)):
            # Integrate Small Domain
            self.update_small_domain()

        # Compute Pullback Values
        self.alpha_L = 1 - ((self.large_tTrial - self.small.t)/(self.large_tTrial - self.large.t))
        self.alpha_s = 1 - ((self.small_tTrial - self.large_tTrial)/(self.small_tTrial - self.small.t))

        if (self.alpha_L >= self.alpha_s):
            self.large.dt = self.alpha_L * self.large.dt
        elif (self.alpha_s > self.alpha_L):
            self.small.dt = self.alpha_s * self.small.dt
            self.update_small_domain()

        # Integrate Large Domain
        self.large.single_tstep_integrate()
        # Enforce continuity
        self.large.u[-1] = self.small.u[0]
        self.large.v[-1] = self.small.v[0]

        self.steps_L = np.append(self.steps_L, self.large.dt)
        self.a_drift =  np.append(self.a_drift, self.large.a[-1] - self.small.a[0])
        self.u_drift =  np.append(self.u_drift, self.large.u[-1] - self.small.u[0])
        self.v_drift =  np.append(self.v_drift, self.large.v[-1] - self.small.v[0])
        self.t_sync =  np.append(self.t_sync, self.large.t)
        self.large.assemble_internal()        
        self.calc_timestep_ratios()

        self.large.f_int[-1] += self.small.f_int[0]

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
        x = ['L', 'Interface', 'S']
        n_steps = 10 # Number of Steps to Plot
        data = np.empty((n_steps, 3))
        for i in range(n_steps):
            data[i] = np.array([self.updated.steps_L[i], 
                                self.updated.steps_Gamma[i],
                                self.updated.steps_S[i]])
            
        plt.figure(figsize=(10, 6))
        for i in range(1, n_steps):        
            plt.bar(x, data[i], bottom=np.sum(data[:i], axis=0), color=plt.cm.tab10(i), label=f'Local Step {i}')

        plt.ylabel('Time (s)')
        plt.title('Time Steps taken for New Multi-step')
        plt.legend()
        plt.show()

    def plot_drift(self):
        # Plot X Axis Sync Times and Y Axis Drift
        plt.plot(self.updated.t_sync, self.updated.a_drift)
        plt.xlabel("Time (ms)")
        plt.ylabel("Acceleration Drift (m/s^2)")
        plt.title("Acceleration Drift between Large and Small Domains")
        plt.show()

        plt.plot(self.updated.t_sync, self.updated.u_drift)
        plt.xlabel("Time (ms)")
        plt.ylabel("Displacement Drift (mm)")
        plt.title("Displacement Drift between Large and Small Domains")
        plt.show()

        plt.plot(self.updated.t_sync, self.updated.v_drift)
        plt.xlabel("Time (ms)")
        plt.ylabel("Velocity Drift (mm/ms)")
        plt.title("Velocity Drift between Large and Small Domains")
        plt.show()

def newCoupling():
    # Utilise same element size, drive time step ratio with Co.
    nElemLarge = 300
    E_L = 0.02e9
    # E_s = 0.18e9    
    rho = 8000
    E_s = (np.pi/0.02)**2 * rho # Non Integer Time Step Ratio = pi
    Courant = 0.5
    Length = 50e-3
    propTime = 1.75 * Length * np.sqrt(rho / E_L)    
    def vel(t): return vbc.velbcSquareWave(t, 2 * Length , E_L, rho)
    accelBoundaryCondtions = abc(list(),list())
    upd_largeDomain = SimpleIntegrator("updated", E_L, rho, Length, 1, nElemLarge, propTime, vbc([0], [vel]), None, Co=Courant)
    upd_smallDomain = SimpleIntegrator("updated", E_s, rho, Length * 2, 1, nElemLarge * 2, propTime, None, accelBoundaryCondtions, Co=Courant)
    upd_fullDomain = MultiTimeStep(upd_largeDomain, upd_smallDomain)
    plotfullDomain = Visualise_MultiTimestep(upd_fullDomain)
    # Solve Loop
    while(upd_fullDomain.large.t <= upd_fullDomain.large.tfinal):
        upd_fullDomain.integrate()
        print("Time: ", upd_fullDomain.large.t)
        if (upd_fullDomain.large.n % 5 == 0):
            plotfullDomain.plot_accel()
            plotfullDomain.plot_vel()
            plotfullDomain.plot_disp()
            plotfullDomain.plot_stress()

    plotfullDomain.create_gif('Updated_Multi-time-step_accel.gif', plotfullDomain.filenames_accel)
    plotfullDomain.create_gif('Updated_Multi-time-step.gif', plotfullDomain.filenames_vel)
    plotfullDomain.create_gif('Updated_Multi-time-step_disp.gif', plotfullDomain.filenames_disp)
    plotfullDomain.create_gif('Updated_Multi-time-step_stress.gif', plotfullDomain.filenames_stress)

    plotfullDomain.plot_time_steps()
    plotfullDomain.plot_drift()

if __name__ == "__main__":
    newCoupling()
