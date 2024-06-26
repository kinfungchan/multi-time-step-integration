from SimpleIntegrator import SimpleIntegrator
from boundaryConditions.BoundaryConditions import VelBoundaryConditions as vbc
import numpy as np
import matplotlib.pyplot as plt
import imageio 
import os

"""
This module implements the subcycling algorithm with interface linear interpolated velocities from:

Belytschko, T., Yen, H.-J. &; Mullen, R. (1979).
Mixed Methods for Time Integration. Computer Methods in Applied Mechanics
and Engineering, 17/18, pp. 259-275

"""

class Subcycling:

    """
    Constructor for the subcycling class
    It accepts two domains a large and a small one
    They are both SimpleIntegrators, and they ratio is an integer number:
             LARGE      |   SMALL
    *----*----*----*----*--*--*--*--*
    """
    def __init__(self, coupling, largeDomain: SimpleIntegrator, smallDomain: SimpleIntegrator, scaleRatio):

        self.coupling = coupling
        self.large = largeDomain
        self.small = smallDomain
        if (self.coupling == "stable"):
            self.large.mass[-1] += self.small.mass[0]
        self.scaleRatio = scaleRatio


    def integrate(self):
        self.small.assemble_internal()
        self.large.f_int[-1] += self.small.f_int[0]
        if (self.large.formulation != "updated"):
            self.small.f_int.fill(0)
        prevV = self.large.v[-1]
        prevT = self.large.t
        self.large.assemble_internal()
        self.large.single_tstep_integrate()
        currV = self.large.v[-1]
        acceleration = (currV - prevV) / self.large.dt
        smallPrevV = prevV + acceleration * 0.5 * (self.large.dt - self.small.dt)
        def velCoupling(t): return smallPrevV + acceleration * (t - prevT)
        if not self.small.v_bc:
            self.small.v_bc = vbc([0], [velCoupling])
        else:
            self.small.v_bc.indexes.append(0)
            self.small.v_bc.velocities.append(velCoupling)
        counter = 0
        while(counter < self.scaleRatio):
            if ((self.large.formulation == "total") or (self.large.formulation == "updated" and counter != 0)):
                self.small.assemble_internal()
            self.small.single_tstep_integrate()
            counter += 1

class Visualise_Subcycling:

    def __init__(self, tot_fullDomain: Subcycling, upd_fullDomain: Subcycling):

        self.total = tot_fullDomain
        self.updated = upd_fullDomain
        self.filenames = []

    def plot(self):
        self.filenames.append(f'FEM1D{self.total.large.n}{self.updated.large.n}.png')
        plt.style.use('ggplot')
        plt.plot(self.total.large.position, self.total.large.v)
        plt.plot(self.total.small.position + 0.5, self.total.small.v)
        plt.plot(self.updated.large.position, self.updated.large.v, "--")
        plt.plot(self.updated.small.position + 0.5, self.updated.small.v, "--")
        plt.title(f"Graph of Velocity against Position for a Half Sine Excitation", fontsize=9)
        plt.xlabel("Domain Position (mm)", fontsize=9)
        plt.ylabel("Velocity (mm/ms)", fontsize=9)
        plt.legend([f"Total Large Domain", "Total Small Domain", "Updated Large Domain", "Updated Small Domain"])
        plt.savefig(f'FEM1D{self.total.large.n}{self.updated.large.n}.png')
        plt.close()

    def create_gif(self):
        with imageio.get_writer('Updated_and_Total_SpuriousWave.gif', mode='I') as writer:
            for filename in self.filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        for filename in set(self.filenames):
            os.remove(filename)

"""
The stable coupling integrates the bar with an exchange of mass with the SAME space discretization
using two different timesteps
"""
def BelytschkoCoupling():
    nElemLarge = 250
    refinementFactor = 1
    E = 207
    rho = 7.83e-6
    L = 1
    Courant = 0.9
    propTime = 1 * L * np.sqrt(rho / E)
    def vel(t): return vbc.velbc(t, L, E, rho)
    tot_largeDomain = SimpleIntegrator("total", E, rho, L * 0.5, 1, nElemLarge, propTime, None, None, Co=Courant)
    tot_smallDomain = SimpleIntegrator("total", E, rho, L * 0.5, 1, nElemLarge, propTime, vbc([nElemLarge], [vel]), None, Co=Courant/refinementFactor)
    tot_fullDomain = Subcycling("stable", tot_largeDomain, tot_smallDomain, refinementFactor)
    upd_largeDomain = SimpleIntegrator("updated", E, rho, L * 0.5, 1, nElemLarge, propTime, None, None, Co=Courant)
    upd_smallDomain = SimpleIntegrator("updated", E, rho, L * 0.5, 1, nElemLarge, propTime, vbc([nElemLarge], [vel]), None, Co=Courant/refinementFactor)
    upd_fullDomain = Subcycling("stable", upd_largeDomain, upd_smallDomain, refinementFactor)
    fullDomain = Visualise_Subcycling(tot_fullDomain, upd_fullDomain)
    while(upd_fullDomain.large.t <= upd_fullDomain.large.tfinal):
        upd_fullDomain.integrate()
        tot_fullDomain.integrate()
        fullDomain.plot()
    fullDomain.create_gif()

if __name__ == "__main__":
    BelytschkoCoupling()



