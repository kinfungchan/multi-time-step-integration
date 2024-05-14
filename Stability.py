import numpy as np
import matplotlib.pyplot as plt
from Sandbox import exportCSV, writeCSV
from Visualise import Plot

class Stability:

    def __init__(self):

        # Lagrange Multiplier Equivalence at Large Time Steps
        self.lm_L = np.array([0.0])
        self.lm_s = np.array([0.0])
        self.a_diff = np.array([0.0])
        self.a_Gamma = np.array([0.0])
        self.a_f = np.array([0.0])

        # Fint Equivalence at Small Time Steps
        self.f_int_L_rec = np.array([0.0]) # Recovered Large Internal Force from LM
        self.f_int_L_int = np.array([0.0]) # Integrated Large Internal Force
        self.u_s = np.array([0.0]) # Actual small Displacement
        self.u_L_int = np.array([0.0]) # Integrated Large Displacement
        self.a_tilda = np.array([0.0]) # unconstrained acceleration
        self.a_const = np.array([0.0]) # constrained acceleration

        # Kinetic Energy
        self.KE_L = np.array([0.0])
        self.KE_S = np.array([0.0])
        # Coupling Energy
        self.W_Gamma_L_dtL = np.array([0.0])
        self.W_Gamma_S_dtL = np.array([0.0])
        
        # Calculate Drifting
        self.a_drift = np.array([0.0])
        self.u_drift = np.array([0.0])
        self.v_drift = np.array([0.0])
        self.t_sync = np.array([0.0])
        self.t_small = np.array([0.0])


    """
    Stability over a Large Time Step
    
    """

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

        self.lm_L = np.append(self.lm_L, lm_L)
        self.lm_s = np.append(self.lm_s, lm_s)
        self.a_f = np.append(self.a_f, a_f)
        self.a_Gamma = np.append(self.a_Gamma, a_Gamma)

        return lm_L, lm_s, a_f
    
    def plot_LMEquiv(self, csv):
        LM = Plot()
        LM.plot(1, [self.t_sync], [self.a_diff], 
                "Acceleration Difference between Lagrange Multiplier and Proposed Method",
                "Time (s)", r"$\sqrt{(a_f - a_{\Gamma})^2}$ (m/s$^2$)", 
                ["Acceleration Difference"],
                [None, None], [-1e-10, 1e-10],
                True)
        if (csv):
            writeCSV("accel_equiv.csv", self.t_sync, self.a_diff, 't_L', 'a_diff')
            writeCSV("a_f.csv", self.t_sync, self.a_f, 't_L', 'a_f')
            writeCSV("a_Gamma.csv", self.t_sync, self.a_Gamma, 't_L', 'a_Gamma')

        LM.plot(2, [self.t_sync, self.t_sync], [self.lm_L, self.lm_s],
                "Lagrange Multiplier for Large and Small Domains",
                "Time (s)", "Lagrange Multiplier (N)",
                ["Large", "Small"], 
                [None, None], [None, None],
                True)
        if (csv):
            writeCSV("lm_L_equiv.csv", self.t_sync, self.lm_L, 't_L', 'lm_L')
            writeCSV("lm_s_equiv.csv", self.t_sync, self.lm_s, 't_L', 'lm_s')

        LM.plot(1, [self.t_sync], [self.lm_L + self.lm_s],
                "Lagrange Multiplier for Large + Small Domains",
                "Time (s)", "Lagrange Multiplier (N)",
                ["Large + Small"], 
                [None, None], [None, None],
                True)
        if (csv):
            writeCSV("lm_equiv.csv", self.t_sync, self.lm_L + self.lm_s, 't_L', 'lm')


    def calc_W_Gamma_s(self, mass_s, a_Gamma, f_int_s, u_s):
        f_s = self.lm_L[-1] + (mass_s * a_Gamma) + f_int_s
        W_Gamma_s = 0.5 * f_s * u_s

        self.W_Gamma_S_dtL = np.append(self.W_Gamma_S_dtL, W_Gamma_s)

    def calc_W_Gamma_L(self, mass_L, a_Gamma, f_int_L, u_L):
        f_L = self.lm_s[-1] + (mass_L * a_Gamma) + f_int_L
        W_Gamma_L = 0.5 * f_L * u_L

        self.W_Gamma_L_dtL = np.append(self.W_Gamma_L_dtL, W_Gamma_L)

    def plot_W_Gamma_dtL(self):
        W_Gamma = Plot()
        W_Gamma.plot(1, [self.t_sync], [self.W_Gamma_L_dtL + self.W_Gamma_S_dtL],
                    "Coupling Energy for Large and Small Domains over Large Time Steps",
                    "Time (s)", "Coupling Energy (J)",
                    ["Large + Small"], 
                    [None, None], [None, None],
                    True)
        
    def calc_KE(self, mass_L, v_L, mass_S, v_S):
        self.KE_S = np.append(self.KE_S, 0.5 * mass_S * v_S**2)
        self.KE_L = np.append(self.KE_L, 0.5 * mass_L * v_L**2)

    def plot_KE(self):
        plt.plot(self.t_sync, self.KE_L - self.KE_S)
        # plt.plot(self.t_sync, self.KE_L)
        # plt.plot(self.t_sync, self.KE_S)
        plt.legend(["Large", "Small"])
        plt.xlabel("Time (s)")
        plt.ylabel("Kinetic Energy (J)")
        plt.title("At Large Time Steps: Large KE and Small KE")
        plt.show()    

    """
    Stability over a Small Time Step
    
    """

    def fint_Equiv(self, mass_L, mass_s, f_int_s, a_Gamma):
        # Finding an equivalent Large Internal Force over a Small Time Step
        ta1 = -f_int_s / mass_s
        lm_s = (ta1 - a_Gamma) * mass_s
        lm_L = -lm_s
        f_int_L = -lm_L - mass_L * a_Gamma

        return f_int_L

    def plot_fintEquiv(self):
        plt.plot(self.t_small, (self.f_int_L_int - self.f_int_L_rec))
        # plt.plot(self.t_small, self.f_int_L_rec)
        # plt.plot(self.t_small, self.f_int_L_int)
        # plt.legend(["Recovered from LM", "Integrated"])
        plt.xlabel("Time (s)")
        plt.ylabel("Internal Force (N)")
        plt.title("Internal Force Equivalence for Large Domain over Small Time Steps")
        plt.show()

    def plot_u_Equiv(self):
        u_eq = Plot()
        u_eq.plot(1, [self.t_small], [self.u_L_int - self.u_s],
                "Displacement Equivalence for Large Domain over Small Time Steps",
                "Time (s)", "Displacement (m)",
                ["Displacement Difference"], 
                [None, None], [-1e-12, 1e-12],
                True)
    
    def plot_a_small(self):
        # plt.plot(self.t_small, self.a_tilda)
        # plt.plot(self.t_small, self.a_const)
        # plt.legend(["Unconstrained", "Constrained"])
        plt.ylabel("Acceleration (m/s^2)")
        plt.plot(self.t_small, self.a_tilda - self.a_const)
        plt.ylabel(r"$\mathregular{Acceleration\ \tilde{a} - a_{\Gamma}\ (m/s^2)}$")
        plt.xlabel("Time (s)")        
        plt.title("Acceleration for Interface Small Domain over Small Time Steps")
        plt.show()

    """
    Checking of Drifting Conditions
    
    """

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