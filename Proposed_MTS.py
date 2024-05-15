from SimpleIntegrator import SimpleIntegrator
from BoundaryConditions import VelBoundaryConditions as vbc
from BoundaryConditions import AccelBoundaryConditions as abc
import numpy as np
from Visualise import Plot, Animation

"""
This module implements the subcycling algorithm with interface constant acceleration from:

K.F. Chan, N. Bombace, D. Sap, D. Wason, and N. Petrinic (2024),
A Multi-Time Stepping Algorithm for the Modelling of Heterogeneous Structures with Explicit Time Integration, 
I.J. Num. Meth. in Eng., 2024;00:1–6.

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
        # Tolerance
        self.tol = 1e-6

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
        self.small.assemble_internal()
        self.calc_timestep_ratios()

    def integrate(self):    
        if ((self.large.t == 0) and (self.small.t == 0)):
            self.small.assemble_internal()
            self.large.assemble_internal()
            self.f_int_Gamma = self.large.f_int[-1] + self.small.f_int[0]
            self.calc_timestep_ratios()

        while (self.nextTimeStepRatio <= 1 or (self.currTimeStepRatio <= 1 and self.nextTimeStepRatio <= 1 + self.tol)):
            # Integrate Small Domain
            self.update_small_domain()

        # Compute Reduction Factors 
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
        self.large.assemble_internal()        
        self.calc_timestep_ratios()

        # Interface Internal Force Summation
        self.f_int_Gamma = self.large.f_int[-1] + self.small.f_int[0]  

def proposedCoupling():
    nElemLarge = 300
    E_L = 0.02e9
    rho = 8000
    E_s = (np.pi/0.02)**2 * rho # Non Integer Time Step Ratio = pi
    Courant = 0.5
    Length = 50e-3
    propTime = 1.75 * Length * np.sqrt(rho / E_L) 
    def vel(t): return vbc.velbcSquareWave(t, 2 * Length , E_L, rho)
    accelBCs_L = abc(list(),list())
    accelBCs_s = abc(list(),list())
    upd_largeDomain = SimpleIntegrator("total", E_L, rho, Length, 1, nElemLarge, propTime, vbc([0], [vel]), accelBCs_L, Co=Courant)
    upd_smallDomain = SimpleIntegrator("total", E_s, rho, Length * 2, 1, nElemLarge * 2, propTime, None, accelBCs_s, Co=Courant)
    upd_fullDomain = MultiTimeStep(upd_largeDomain, upd_smallDomain)
    
    # Initilise Plotting
    plot = Plot()
    animate = Animation(plot)

    # Solve Loop
    while(upd_fullDomain.large.t <= 0.0016):
        upd_fullDomain.integrate()
        
        # Plotting and Saving Figures
        print("Time: ", upd_fullDomain.large.t)
        if (upd_fullDomain.large.n % 10 == 0):
            animate.save_single_plot(2, [upd_fullDomain.large.position, [position + upd_fullDomain.large.L for position in upd_fullDomain.small.position]],
                                     [upd_fullDomain.large.a, upd_fullDomain.small.a],
                                     "Acceleration", "Domain Position (m)", "Acceleration (m/s^2)",
                                     animate.filenames_accel, upd_fullDomain.large.n,
                                     ["Large", "Small"])
            animate.save_single_plot(2, [upd_fullDomain.large.position, [position + upd_fullDomain.large.L for position in upd_fullDomain.small.position]],
                                     [upd_fullDomain.large.v, upd_fullDomain.small.v],
                                     "Velocity", "Domain Position (m)", "Velocity (m/s)",
                                     animate.filenames_vel, upd_fullDomain.large.n,
                                     ["Large", "Small"])
            animate.save_single_plot(2, [upd_fullDomain.large.position, [position + upd_fullDomain.large.L for position in upd_fullDomain.small.position]],
                                     [upd_fullDomain.large.u, upd_fullDomain.small.u],
                                     "Displacement", "Domain Position (m)", "Displacement (m)",
                                     animate.filenames_disp, upd_fullDomain.large.n,
                                     ["Large", "Small"])
            animate.save_single_plot(2, [upd_fullDomain.large.midposition, [position + upd_fullDomain.large.L for position in upd_fullDomain.small.midposition]],
                                     [upd_fullDomain.large.stress, upd_fullDomain.small.stress],
                                     "Stress", "Domain Position (m)", "Stress (Pa)",
                                     animate.filenames_stress, upd_fullDomain.large.n,
                                     ["Large", "Small"])
    animate.save_MTS_gifs()

if __name__ == "__main__":
    proposedCoupling()

    
