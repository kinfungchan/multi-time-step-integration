import numpy as np
import matplotlib.pyplot as plt

class Stability:

    def __init__(self):

        # Lagrange Multiplier Equivalence at Large Time Steps
        self.lm_L = np.array([0.0])
        self.lm_s = np.array([0.0])
        self.a_diff = np.array([0.0])

        # Energy Methods
        self.W_GammaL = np.array([0.0])
        self.W_GammaS = np.array([0.0])
        self.W_Gamma = np.array([0.0])

        self.KE_L = np.array([0.0])
        self.KE_S = np.array([0.0])

        self.couplingForce_L = np.array([0.0])
        self.couplingForce_S = np.array([0.0])
        
        # Calculate Drifting
        self.a_drift = np.array([0.0])
        self.u_drift = np.array([0.0])
        self.v_drift = np.array([0.0])
        self.t_sync = np.array([0.0])
        self.t_small = np.array([0.0])

    def LagrangeMultiplierEquiv(self, mass_L, mass_S, f_int_L, f_int_s, a_Gamma):
        # Interface Equation of Motion
        iEOM = np.empty((3, 3))
        iEOM[0, 0] = 1 / mass_L; iEOM[0, 1] = 0; iEOM[0, 2] = 1
        iEOM[1, 0] = 0; iEOM[1, 1] = 1 / mass_S; iEOM[1, 2] = 1
        iEOM[2, 0] = 1; iEOM[2, 1] = 1; iEOM[2, 2] = 0

        # Unconstrained Acceleration Vector
        ta = np.empty(3)
        ta[0] = -f_int_L / mass_L 
        ta[1] = -f_int_s / mass_S
        ta[2] = 0

        # Lagrange Multiplier and Frame Acceleration
        x = np.linalg.solve(iEOM, ta) 
        lm_L = x[0]; lm_s = x[1]; a_f = x[2]

        self.couplingForce_L = np.append(self.couplingForce_L, lm_L)
        self.couplingForce_S = np.append(self.couplingForce_S, lm_s)

        return lm_L, lm_s, a_Gamma - a_f
    
    def plot_LMEquiv(self):
        plt.plot(self.t_sync, self.a_diff)
        plt.xlabel("Time (s)")
        plt.ylabel(r"$a_{\Gamma} - a_f$ (m/s$^2$)")
        plt.title("Acceleration Difference between Lagrange Multiplier and Proposed Method")
        plt.show()

        plt.plot(self.t_sync, self.lm_L)
        plt.plot(self.t_sync, self.lm_s)
        plt.legend(["Large", "Small"])
        plt.xlabel("Time (s)")
        plt.ylabel("Lagrange Multiplier (N)")
        plt.title("Lagrange Multiplier for Large and Small Domains")
        plt.show()

        plt.plot(self.t_sync, self.lm_L + self.lm_s)
        plt.xlabel("Time (s)")
        plt.ylabel("Lagrange Multiplier (N)")
        plt.title("Lagrange Multiplier for Large + Small")
        plt.show()

    def calc_KE(self, mass_L, v_L, mass_S, v_S):
        self.KE_S = np.append(self.KE_S, 0.5 * mass_S * v_S**2)
        self.KE_L = np.append(self.KE_L, 0.5 * mass_L * v_L**2)
    
    def calc_IE(self, stress,  E, dx):
        return ((0.5 * stress**2) / E) * dx

    def calc_Work(self, lm_L, lm_s, u_L, u_s):
        self.W_GammaS = np.append(self.W_GammaS, lm_s * u_s)
        self.W_GammaL = np.append(self.W_GammaL, lm_L * u_L)

    def plot_Work(self):
        plt.plot(self.t_sync, self.W_GammaL + self.W_GammaS)
        # plt.plot(self.t_sync, self.W_GammaL)
        # plt.plot(self.t_sync, self.W_GammaS)
        plt.legend(["Large", "Small"])
        plt.xlabel("Time (s)")
        plt.ylabel("Difference in Work (J)")
        plt.title("Over a Large Time Step: Large Work + Small Work")
        plt.show()

    def plot_KE(self):
        # plt.plot(self.t_sync, self.KE_L - self.KE_S)
        plt.plot(self.t_sync, self.KE_L)
        plt.plot(self.t_sync, self.KE_S)
        plt.legend(["Large", "Small"])
        plt.xlabel("Time (s)")
        plt.ylabel("Kinetic Energy (J)")
        plt.title("At Large Time Steps: Large KE and Small KE")
        plt.show()

    def calc_drift(self, a_L, a_s, v_L, v_s, u_L, u_s, t):
        self.a_drift =  np.append(self.a_drift, a_L - a_s)
        self.u_drift =  np.append(self.u_drift, v_L - v_s)
        self.v_drift =  np.append(self.v_drift, u_L - u_s)
        self.t_sync =  np.append(self.t_sync, t)

    def plot_drift(self):
        plt.plot(self.t_sync, self.a_drift)
        plt.xlabel("Time (ms)")
        plt.ylabel("Acceleration Drift (m/s^2)")
        plt.title("Acceleration Drift between Large and Small Domains")
        plt.show()

        plt.plot(self.t_sync, self.u_drift)
        plt.xlabel("Time (ms)")
        plt.ylabel("Displacement Drift (mm)")
        plt.title("Displacement Drift between Large and Small Domains")
        plt.show()

        plt.plot(self.t_sync, self.v_drift)
        plt.xlabel("Time (ms)")
        plt.ylabel("Velocity Drift (mm/ms)")
        plt.title("Velocity Drift between Large and Small Domains")
        plt.show()