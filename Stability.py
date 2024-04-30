import numpy as np
import matplotlib.pyplot as plt

class Stability:

    def __init__(self):

        # Energy Methods
        self.W_GammaL = np.array([0.0])
        self.W_GammaS_k = 0.0
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

    def calc_KE(self, mass_L, v_L, mass_S, v_S):
        self.KE_S = np.append(self.KE_S, 0.5 * mass_S * v_S**2)
        self.KE_L = np.append(self.KE_L, 0.5 * mass_L * v_L**2)
    
    def calc_IE(self, stress,  E, dx):
        return ((0.5 * stress**2) / E) * dx
    
    def calc_Work_s(self, m, a, ma_prev, f_int, f_int_prev, u, u_prev): 
        self.couplingForce_S = np.append(self.couplingForce_S, ((m * a) - f_int))
        self.W_GammaS_k += 0.5 * (((m * a) - ma_prev) - (f_int - f_int_prev)) * (u - u_prev)

    def calc_Work_L(self, m, a, ma_prev, f_int, f_int_prev, u, u_prev):
        self.W_GammaS = np.append(self.W_GammaS, self.W_GammaS_k)
        self.couplingForce_L = np.append(self.couplingForce_L, ((m * a) - f_int))
        self.W_GammaL = np.append(self.W_GammaL, 0.5 * (((m * a) - ma_prev) - (f_int - f_int_prev)) * (u - u_prev))

    def plot_Work(self):
        plt.plot(self.t_sync, self.W_GammaL + self.W_GammaS)
        # plt.plot(self.t_sync, self.W_GammaL)
        # plt.plot(self.t_sync, self.W_GammaS)
        plt.legend(["Large", "Small"])
        plt.xlabel("Time (s)")
        plt.ylabel("Difference in Work (J)")
        plt.title("Over a Large Time Step: Large Work + Small Work")
        plt.show()

    def plot_CouplingForce(self):
        plt.plot(self.t_sync, self.couplingForce_L)
        plt.plot(self.t_small, self.couplingForce_S)
        plt.legend(["Large", "-Small"])
        plt.xlabel("Time (s)")
        plt.ylabel("Coupling Force (N)")
        plt.title("Large Coupling Force and Small Coupling Force")
        plt.show()

    def plot_CouplingForceDiff(self):
        couplingForce_S_3 = self.couplingForce_S[::3]
        plt.plot(self.t_sync, self.couplingForce_L + couplingForce_S_3)
        plt.xlabel("Time (s)")
        plt.ylabel("Difference in Coupling Force (N)")
        plt.title("Difference in Coupling Force between Large and Small Domains at Large Time Steps")
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