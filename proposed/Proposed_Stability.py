from proposed.SimpleIntegrator import SimpleIntegrator
from boundaryConditions.BoundaryConditions import VelBoundaryConditions as vbc
from boundaryConditions.BoundaryConditions import AccelBoundaryConditions as abc
from proposed.Stability import Stability
from proposed.Energy import SubdomainEnergy
from literature import vHalftoCSV
from utils.Visualise import Plot, Animation
import numpy as np

"""
This module implements the subcycling algorithm with interface constant acceleration from:

K.F. Chan, N. Bombace, D. Sap, D. Wason, and N. Petrinic (2024),
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

        k = 0
        dW_Link_s = 0.0 
        dW_Link_L = 0.0

        while (self.nextTimeStepRatio <= 1 or (self.currTimeStepRatio <= 1 and self.nextTimeStepRatio <= 1.000001)):
            # Integrate Small Domain
            self.update_small_domain()
            k+=1
            self.stability.energy_s.calc_energy_balance_subdomain(self.small.n_nodes, self.small.n_elem, self.small.mass, 
                                                                  self.small.v, self.small.stress, self.small.bulk_viscosity_stress,
                                                                  self.small.E, self.small.dx)

            self.stability.t_small = np.append(self.stability.t_small, self.small.t)
            if (k < 3):
                lm_L_dts, lm_s_dts, f_int_L_dts = self.stability.f_int_L_equiv(self.large.mass[-1], self.small.mass[0],
                                                                            self.small.f_int[0], self.accelCoupling())
                dW_Link_s += 0.5 * (self.small.u[0] - self.small.u_prev[0]) * (self.stability.lm_s_dts[-1] + self.stability.lm_s_dts[-2])

        # Compute Pullback Values
        self.alpha_L = 1 - ((self.large_tTrial - self.small.t)/(self.large_tTrial - self.large.t))
        self.alpha_s = 1 - ((self.small_tTrial - self.large_tTrial)/(self.small_tTrial - self.small.t))

        if (self.alpha_L >= self.alpha_s):
            self.large.dt = self.alpha_L * self.large.dt
        elif (self.alpha_s > self.alpha_L):
            self.small.dt = self.alpha_s * self.small.dt
            self.update_small_domain()
            k+=1
            
        # Integrate Large Domain
        self.large.a_bc.indexes.append(-1)
        self.large.a_bc.accelerations.append(self.accelCoupling)
        self.large.single_tstep_integrate()

        # Enforce continuity
        # self.large.u[-1] = self.small.u[0]
        # self.large.v[-1] = self.small.v[0]

        # Comparison for other MTS Methods
        self.min_dt = min(self.min_dt, self.small.dt, self.large.dt)
        self.steps_L = np.append(self.steps_L, self.large.dt)
        self.el_steps += self.large.n_elem
        self.stability.calc_drift(self.large.a[-1], self.small.a[0], self.large.v[-1], self.small.v[0], self.large.u[-1], self.small.u[0], self.large.t)

        self.large.assemble_internal()        
        self.calc_timestep_ratios()

        self.f_int_Gamma = self.large.f_int[-1] + self.small.f_int[0]  

        self.stability.energy_L.calc_energy_balance_subdomain(self.large.n_nodes, self.large.n_elem, self.large.mass,
                                                              self.large.v, self.large.stress, self.large.bulk_viscosity_stress,
                                                              self.large.E, self.large.dx)
        
        lm_L_dts, lm_s_dts, f_int_L_dts = self.stability.f_int_L_equiv(self.large.mass[-1], self.small.mass[0],
                                                                            self.small.f_int[0], self.accelCoupling())
        dW_Link_s += 0.5 * (self.small.u[0] - self.small.u_prev[0]) * (self.stability.lm_s_dts[-1] + self.stability.lm_s_dts[-2])
        self.stability.dW_Link_s = np.append(self.stability.dW_Link_s, dW_Link_s)


        # Stability Calculations over Large Time Step
        lm_L, lm_s, a_f = self.stability.LagrangeMultiplierEquiv(self.large.mass[-1], self.small.mass[0], 
                                                                 self.large.f_int[-1], self.small.f_int[0],
                                                                 (-self.f_int_Gamma / self.mass_Gamma))    
        self.stability.a_diff = np.append(self.stability.a_diff, (np.sqrt((a_f - (-self.f_int_Gamma / self.mass_Gamma))) ** 2))

        dW_Link_L += 0.5 * (self.large.u[-1] - self.large.u_prev[-1]) * (self.stability.lm_L[-1] + self.stability.lm_L[-2])  
        self.stability.dW_Link_L = np.append(self.stability.dW_Link_L, dW_Link_L)

        # Evaluate W_Gamma 
        self.stability.calc_dW_Gamma_dtL("Small", self.small.mass[0], a_f, self.stability.a_f[-2], self.small.f_int[0], self.stability.f_int_s_prev_dtL, 
                                       self.small.u[0], self.stability.u_s_prev_dtL)
        self.stability.calc_dW_Gamma_dtL("Large", self.large.mass[-1], a_f, self.stability.a_f[-2], self.large.f_int[-1], self.large.f_int_prev[-1],
                                       self.large.u[-1], self.large.u_prev[-1])
        
        self.stability.f_int_s_prev_dtL = np.copy(self.small.f_int[0])
        self.stability.u_s_prev_dtL = np.copy(self.small.u[0])

def proposedCouplingStability(vel_csv, stability_plots):
    # Utilise same element size, drive time step ratio with Co.
    nElemLarge = 300
    E_L = 0.02e9 
    rho = 8000
    E_s = (np.pi/0.02)**2 * rho # Non Integer Time Step Ratio = pi
    Courant = 0.5
    Length = 50e-3
    propTime = 1.75 * Length * np.sqrt(rho / E_L)    
    def vel(t): return vbc.velbcSquare(t, 2 * Length , E_L, rho)
    accelBCs_L = abc(list(),list())
    accelBCs_s = abc(list(),list())
    upd_largeDomain = SimpleIntegrator("total", E_L, rho, Length, 1, nElemLarge, propTime, vbc([0], [vel]), accelBCs_L, Co=Courant)
    upd_smallDomain = SimpleIntegrator("total", E_s, rho, Length * 2, 1, nElemLarge * 2, propTime, None, accelBCs_s, Co=Courant)

    energy_L = SubdomainEnergy()
    energy_s = SubdomainEnergy()
    stability = Stability(energy_L, energy_s)    
    upd_fullDomain = MultiTimeStep(upd_largeDomain, upd_smallDomain, stability)

    # Visualisation Classes
    plot = Plot()
    animate = Animation(plot)
    
    # Solve Loop
    while(upd_fullDomain.large.t <= 0.0016):
        upd_fullDomain.integrate()
        print("Time: ", upd_fullDomain.large.t)
        if (upd_fullDomain.large.n % 500 == 0): # Adjust Number for output plots (Set High for Debugging)
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

        # Export to CSV
        if (vel_csv):
            vHalftoCSV(upd_fullDomain.large.t, upd_fullDomain.large.v, upd_fullDomain.small.v,
                   upd_fullDomain.large.t_prev, upd_fullDomain.large.v_prev, upd_fullDomain.small.v_prev, 
                   upd_fullDomain.large.position, upd_fullDomain.small.position, upd_fullDomain.large.L)

    # Simulation Ended - Post-Processing
    animate.save_MTS_gifs("Proposed_Stability")
    
    if (stability_plots):
        ## Subdomain Stability
        stability.plot_EnergyBalance(True)

        ## Interface Stability
        # Over Large Time Steps
        stability.plot_LMEquiv(csv=False)
        stability.plot_dW_Gamma_dtL(show=True,csv=True) # Forces on Interface * Displacement (Large + Small)
        
        # Over Small Time Steps              
        stability.plot_lm_dts()
        stability.plot_dW_Link(show=True,csv=True)

        # Drifting Conditions
        stability.plot_drift(show=True,csv=True)

    plot.plot_dt_bars(upd_fullDomain.steps_L, upd_fullDomain.steps_S, False)

    print("Minimum Time Step for Large Domain: ", upd_fullDomain.min_dt)
    # Print Total Number of Integration Steps on Large 
    print("Number of Integration Steps: ", upd_fullDomain.el_steps)
    # Print First 10 Time Steps on Large and Small
    print("Time Steps: ", upd_fullDomain.steps_L[:10])
    print("Time Steps: ", upd_fullDomain.steps_S[:10])

if __name__ == "__main__":
    proposedCouplingStability(False, True) # Export CSV, Stability Plots

    