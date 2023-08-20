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
    They are both SimpleIntegrators, and they ratio is an integer number:
             LARGE      |   SMALL
    *----*----*----*----*--*--*--*--*
    """
    def __init__(self, largeDomain: SimpleIntegrator, smallDomain: SimpleIntegrator):
        
        self.large = largeDomain
        self.small = smallDomain
        self.large.mass[-1] += self.small.mass[0] #CHANGE when swapping domain for paper numerical e.g.
        # Time step Ratio Computations
        self.large_tpredicted = 0.0
        self.small_tpredicted = 0.0
        # Pullback Values
        self.large_alpha = 0.0
        self.small_alpha = 0.0

    """
    First we implement with a Large Pullback without the need 
    for time step ratios (solely for this case - not general)
    """

    def integrate(self):
    
        if ((self.large.t == 0) and (self.small.t == 0)):
            self.small.assemble_internal()
            self.large.assemble_internal()
            self.large.f_int[-1] += self.small.f_int[0]
            if (self.large.formulation != "updated"):
                self.small.f_int.fill(0) # CHECK
            # acceleration_L = accelCoupling(self.large.f_int[-1] , self.large.mass[-1])
            # Compute time step ratios
            self.small_tpredicted = self.small.t + self.small.dt
            self.large_tpredicted = self.large.t + self.large.dt

        # we could pass in a flag to single_tstep_integrate
        largeForce = self.large.f_int[-1]
        largeMass = self.large.mass[-1]
        def accelCoupling(): return -largeForce / largeMass

        self.small.a_bc.indexes.append(0)
        self.small.a_bc.accelerations.append(accelCoupling)

        self.small.single_tstep_integrate()
        self.small.assemble_internal()
        self.small_tpredicted = self.small.t + self.small.dt

        self.small.single_tstep_integrate()
        self.small.assemble_internal()
        self.small_tpredicted = self.small.t + self.small.dt

        # Compute Pullback Values
        self.alpha_L = 1 - ((self.large_tpredicted - self.small.t)/(self.large_tpredicted - self.large.t))
        self.alpha_s = 1 - ((self.small_tpredicted - self.large_tpredicted)/(self.small_tpredicted - self.small.t))

        if (self.alpha_L >= self.alpha_s):
            self.large.dt = self.alpha_L * self.large.dt
        elif (self.alpha_s > self.alpha_L):
            self.small.dt = self.alpha_s * self.small.dt
            print("Error: Shouldn't be hitting this atm")

        self.large.single_tstep_integrate()
        self.large.assemble_internal()        
        self.large_tpredicted = self.large.t + self.large.dt

        self.large.f_int[-1] += self.small.f_int[0]
        # if (self.large.formulation != "updated"):
        #         self.small.f_int.fill(0) # CHECK

        # calc_timestep_ratios()

    def calc_timestep_ratios(self):
        pass
    # Place where we have predicted time time step

class Visualise_MultiTimestep:

    def __init__(self, upd_fullDomain: MultiTimeStep):

        self.updated = upd_fullDomain
        self.filenames = []

    def plot(self):
        self.filenames.append(f'FEM1D{self.updated.large.n}.png')
        plt.style.use('ggplot')
        plt.plot(self.updated.large.position, self.updated.large.v, "--")
        plt.plot(self.updated.small.position + 0.5, self.updated.small.v, "--")
        plt.title(f"Graph of Velocity against Position for a Half Sine Excitation (Compression)", fontsize=9)
        plt.xlabel("Domain Position (mm)", fontsize=9)
        plt.ylabel("Velocity (mm/ms)", fontsize=9)
        plt.legend([f"Updated Large Domain", "Updated Small Domain"])
        plt.savefig(f'FEM1D{self.updated.large.n}.png')
        plt.close()

    def create_gif(self):
        with imageio.get_writer('Updated_Multi-time-step.gif', mode='I') as writer:
            for filename in self.filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        for filename in set(self.filenames):
            os.remove(filename)


def velbc(t, L, E, rho):
    sinePeriod = (L / 2) * np.sqrt(rho/E)
    freq = 1 / sinePeriod
    if t >= sinePeriod * 0.5:
        return 0
    else:
        return -0.01 * np.sin(2 * np.pi * freq * t) #force higher - original 0.01


def newCoupling():
    # Utilise same element size, drive time step ratio with Co.
    nElemLarge = 250 
    refinementFactor = 2.1 
    E = 207
    rho = 7.83e-6
    L = 1
    Courant = 0.9
    propTime = 1 * L * np.sqrt(rho / E)
    def vel(t): return velbc(t, L, E, rho)
    accelBoundaryCondtions = abc(list(),list())
    upd_largeDomain = SimpleIntegrator("updated", E, rho, L * 0.5, 1, nElemLarge, propTime, None, None, Co=Courant)
    upd_smallDomain = SimpleIntegrator("updated", E, rho, L * 0.5, 1, nElemLarge, propTime, vbc([nElemLarge], [vel]), accelBoundaryCondtions, Co=Courant/refinementFactor)
    upd_fullDomain = MultiTimeStep(upd_largeDomain, upd_smallDomain)
    plotfullDomain = Visualise_MultiTimestep(upd_fullDomain)
    # Solve Loop
    while(upd_fullDomain.large.t <= upd_fullDomain.large.tfinal):
        upd_fullDomain.integrate()
        plotfullDomain.plot()
    plotfullDomain.create_gif()


if __name__ == "__main__":
    newCoupling()
