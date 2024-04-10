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
        self.large_tpredicted = 0.0
        self.small_tpredicted = 0.0
        self.nextTimeStepRatio = 0.0 
        self.currTimeStepRatio = 0.0
        # Pullback Values
        self.large_alpha = 0.0
        self.small_alpha = 0.0
        # Time Step History
        self.steps_L = np.array([0.0])
        self.steps_S = np.array([0.0])

    def calc_timestep_ratios(self):
        self.small_tpredicted = self.small.t + self.small.dt
        self.large_tpredicted = self.large.t + self.large.dt
        self.currTimeStepRatio = (self.small.t - self.large.t) / (self.large_tpredicted - self.large.t)
        self.nextTimeStepRatio = ((self.small.t + self.small.dt) - self.large.t) / (self.large_tpredicted - self.large.t)

    def integrate(self):
    
        if ((self.large.t == 0) and (self.small.t == 0)):
            self.small.assemble_internal()
            self.large.assemble_internal()
            self.large.f_int[-1] += self.small.f_int[0]
            self.calc_timestep_ratios()

        while (self.nextTimeStepRatio <= 1 or (self.currTimeStepRatio <= 1 and self.nextTimeStepRatio <= 1.000001)):
            largeForce = self.large.f_int[-1]
            largeMass = self.large.mass[-1]
            def accelCoupling(): return -largeForce / largeMass

            self.small.a_bc.indexes.append(0)
            self.small.a_bc.accelerations.append(accelCoupling)

            self.small.single_tstep_integrate()
            self.steps_S = np.append(self.steps_S, self.small.dt)
            self.small.assemble_internal()
            self.calc_timestep_ratios()

        # Compute Pullback Values
        self.alpha_L = 1 - ((self.large_tpredicted - self.small.t)/(self.large_tpredicted - self.large.t))
        self.alpha_s = 1 - ((self.small_tpredicted - self.large_tpredicted)/(self.small_tpredicted - self.small.t))

        if (self.alpha_L >= self.alpha_s):
            self.large.dt = self.alpha_L * self.large.dt
        elif (self.alpha_s > self.alpha_L):
            self.small.dt = self.alpha_s * self.small.dt

        self.large.single_tstep_integrate()
        self.steps_L = np.append(self.steps_L, self.large.dt)
        self.large.assemble_internal()        
        self.calc_timestep_ratios()

        self.large.f_int[-1] += self.small.f_int[0]

class Visualise_MultiTimestep:

    def __init__(self, upd_fullDomain: MultiTimeStep):

        self.updated = upd_fullDomain
        self.filenames = []

    def plot(self):
        self.filenames.append(f'FEM1D{self.updated.large.n}.png')
        plt.style.use('ggplot')
        plt.plot(self.updated.large.position, self.updated.large.v, "--")
        plt.plot(self.updated.small.position + self.updated.large.L , self.updated.small.v, "--")
        plt.title(f"1D Wave Propagation through Heterogeneous Media", fontsize=9)
        plt.xlabel("Domain Position (mm)", fontsize=9)
        plt.ylabel("Velocity (mm/ms)", fontsize=9)
        plt.legend([f"Updated Large Time Step Domain", "Updated Small Time Step Domain"])
        plt.savefig(f'FEM1D{self.updated.large.n}.png')
        plt.close()

    def create_gif(self):
        with imageio.get_writer('Updated_Multi-time-step.gif', mode='I') as writer:
            for filename in self.filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        for filename in set(self.filenames):
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
    upd_smallDomain = SimpleIntegrator("updated", E_s, rho, Length * 2, 1, nElemLarge, propTime, None, accelBoundaryCondtions, Co=Courant)
    upd_fullDomain = MultiTimeStep(upd_largeDomain, upd_smallDomain)
    plotfullDomain = Visualise_MultiTimestep(upd_fullDomain)
    # Solve Loop
    while(upd_fullDomain.large.t <= upd_fullDomain.large.tfinal):
        upd_fullDomain.integrate()
        if (upd_fullDomain.large.n % 5 == 0):
            plotfullDomain.plot()
    plotfullDomain.create_gif()
    plotfullDomain.plot_time_steps()

if __name__ == "__main__":
    newCoupling()
