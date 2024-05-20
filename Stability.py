import numpy as np
import matplotlib.pyplot as plt
from Sandbox import exportCSV, writeCSV
from Energy import SubdomainEnergy
from Visualise import Plot

class Stability:

    def __init__(self, energy_L: SubdomainEnergy, energy_s: SubdomainEnergy):
        self.P = Plot()

        # Energy Balance for Large and Small Domains
        self.energy_L = energy_L
        self.energy_s = energy_s

        # Lagrange Multiplier Equivalence at Large Time Steps
        self.lm_L = np.array([0.0])
        self.lm_s = np.array([0.0])
        self.a_diff = np.array([0.0])
        self.a_Gamma = np.array([0.0])
        self.a_f = np.array([0.0])
        # Coupling Energy
        self.dW_Gamma_L_dtL = np.array([0.0])
        self.dW_Gamma_S_dtL = np.array([0.0])
        self.f_int_s_prev_dtL = 0.0
        self.u_s_prev_dtL = 0.0

        # Fint Equivalence at Small Time Steps
        self.lm_L_dts = np.array([0.0])
        self.lm_s_dts = np.array([0.0])
        self.f_int_L_dts = np.array([0.0])

        self.dW_Link_s = np.array([0.0])
        self.dW_Link_L = np.array([0.0])
        
        # Calculate Drifting
        self.a_drift = np.array([0.0])
        self.u_drift = np.array([0.0])
        self.v_drift = np.array([0.0])
        self.t_sync = np.array([0.0])
        self.t_small = np.array([0.0])

    """
    Energy Balance for Subdomains

    """
    def plot_EnergyBalance(self, show):
        if (show):
            # Energy Balance
            W = Plot()
            W.plot(8, [self.t_small, self.t_sync, self.t_small, self.t_sync, self.t_small, self.t_sync, self.t_small, self.t_sync], 
                    [self.energy_s.KE, self.energy_L.KE, self.energy_s.IE, self.energy_L.IE, self.energy_s.EBAL, self.energy_L.EBAL, self.energy_s.XE, self.energy_L.XE],
                    "Energy Balance for Large and Small Domains",
                    "Time (s)", "Energy (J)",
                    ["Large KE", "Small KE", "Large IE", "Small IE", "Large EBAL", "Small EBAL", "Large ExtE", "Small ExtE"], 
                    [None, None], [None, None],
                    True)

    def plot_externalWork(self):
        plt.plot(self.t_sync, self.energy_L.EE)
        plt.plot(self.t_small, self.energy_s.EE)
        plt.legend(["Large", "Small"])
        plt.xlabel("Time (s)")
        plt.ylabel("Difference in External Work (J)")
        plt.title("External Work Difference between Large and Small Domains")
        plt.show()

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
        # LM = Plot()
        # Plot a_Gamma and a_f
        self.P.plot(2, [self.t_sync, self.t_sync], [self.a_Gamma, self.a_f],
                "Accelerations for Proposed Method and Lagrange Multiplier",
                "Time (s)", "Acceleration (m/s$^2$)",
                ["a_Gamma", "a_f"], 
                [None, None], [None, None],
                True)
        self.P.plot(1, [self.t_sync], [self.a_diff], 
                "Acceleration Difference between Lagrange Multiplier and Proposed Method",
                "Time (s)", r"$\sqrt{(a_f - a_{\Gamma})^2}$ (m/s$^2$)", 
                ["Acceleration Difference"],
                [None, None], [-1e-10, 1e-10],
                True)
        if (csv):
            writeCSV("accel_equiv.csv", self.t_sync, self.a_diff, 't_L', 'a_diff')
            writeCSV("a_f.csv", self.t_sync, self.a_f, 't_L', 'a_f')
            writeCSV("a_Gamma.csv", self.t_sync, self.a_Gamma, 't_L', 'a_Gamma')

        self.P.plot(2, [self.t_sync, self.t_sync], [self.lm_L, self.lm_s],
                "Lagrange Multiplier for Large and Small Domains",
                "Time (s)", "Lagrange Multiplier (N)",
                ["Large", "Small"], 
                [None, None], [None, None],
                True)
        if (csv):
            writeCSV("lm_L_equiv.csv", self.t_sync, self.lm_L, 't_L', 'lm_L')
            writeCSV("lm_s_equiv.csv", self.t_sync, self.lm_s, 't_L', 'lm_s')

        self.P.plot(1, [self.t_sync], [self.lm_L + self.lm_s],
                "Lagrange Multiplier for Large + Small Domains",
                "Time (s)", "Lagrange Multiplier (N)",
                ["Large + Small"], 
                [None, None], [None, None],
                True)
        if (csv):
            writeCSV("lm_equiv.csv", self.t_sync, self.lm_L + self.lm_s, 't_L', 'lm')


    def calc_dW_Gamma_dtL(self, Domain, mass, a_Gamma, a_Gamma_prev, f_int, f_int_prev, u, u_prev):
        if (Domain == "Large"):
            dlm = self.lm_L[-1] + self.lm_L[-2]
        else:
            dlm = self.lm_s[-1] + self.lm_s[-2]

        du = u - u_prev
        dW_Gamma = 0.5 * du * (dlm + (mass * (a_Gamma + a_Gamma_prev)) + (f_int + f_int_prev))

        if (Domain == "Large"):
            self.dW_Gamma_L_dtL = np.append(self.dW_Gamma_L_dtL, dW_Gamma)
        else:
            self.dW_Gamma_S_dtL = np.append(self.dW_Gamma_S_dtL, dW_Gamma)

    def plot_dW_Gamma_dtL(self, show):
        if (show):
            self.P.plot(2, [self.t_sync, self.t_sync], [self.dW_Gamma_L_dtL, self.dW_Gamma_S_dtL],
                        "Interface Energy for Large and Small Domains",
                        "Time (s)", "Interface Energy ($\delta \Gamma$J)",
                        ["$\delta \Gamma_L$", "$\delta \Gamma_s$"], 
                        [None, None], [None, None],
                        True)
            self.P.plot(1, [self.t_sync], [self.dW_Gamma_L_dtL + self.dW_Gamma_S_dtL],
                        "Interface Energy for Large + Small Domains over Large Time Steps",
                        "Time (s)", "Interface Energy ($\delta \Gamma$J)",
                        ["$\delta \Gamma_L + \delta \Gamma_s$"], 
                        [None, None], [None, None],
                        True)  

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
    
    def f_int_L_equiv(self, mass_L, mass_s, f_int_s, a_Gamma):
        # Interface Force Balance Equation
        iFBE = np.empty((3, 3))
        iFBE[0, 0] = 1; iFBE[0, 1] = 0; iFBE[0, 2] = 1
        iFBE[1, 0] = 0; iFBE[1, 1] = 1; iFBE[1, 2] = 0
        iFBE[2, 0] = 1; iFBE[2, 1] = 1; iFBE[2, 2] = 0

        # Vector of Knowns
        b = np.empty(3)
        b[0] = -mass_L * a_Gamma
        b[1] = -mass_s * a_Gamma - f_int_s
        b[2] = 0

        # Solve for Large Internal Force
        x = np.linalg.solve(iFBE, b)
        lm_L = x[0]; lm_s = x[1]; f_int_L = x[2]

        self.lm_L_dts = np.append(self.lm_L_dts, lm_L)
        self.lm_s_dts = np.append(self.lm_s_dts, lm_s)
        self.f_int_L_dts = np.append(self.f_int_L_dts, f_int_L)
        return lm_L, lm_s, f_int_L
    
    def plot_lm_dts(self):
        self.P.plot(2, [self.t_small, self.t_small], [self.lm_L_dts, self.lm_s_dts],
                    "Lagrange Multiplier for Large and Small Domains over Small Time Steps",
                    "Time (s)", "Lagrange Multiplier (N)",
                    ["Large", "Small"], 
                    [None, None], [None, None],
                    True)

        self.P.plot(1, [self.t_small], [self.lm_L_dts + self.lm_s_dts],
                    "Lagrange Multiplier for Large + Small Domains over Small Time Steps",
                    "Time (s)", "Lagrange Multiplier (N)",
                    ["Large + Small"], 
                    [None, None], [-1e-12, 1e-12],
                    True)

    def plot_dW_Link(self):
        self.P.plot(2, [self.t_sync, self.t_sync], [self.dW_Link_L, self.dW_Link_s],
                    "Link Work for Large and Small Domains over Small Time Steps",
                    "Time (s)", "Increments in Coupling Work (J)",
                    ["Large", "Small"], 
                    [None, None], [None, None],
                    True)
        self.P.plot(1, [self.t_sync], [self.dW_Link_L + self.dW_Link_s],
                    "Link Work for Large + Small Domains over Small Time Steps",
                    "Time (s)", "Increments in Coupling Work (J)",
                    ["Large + Small"], 
                    [None, None], [None, None],
                    True)

    """
    Checking of Drifting Conditions
    
    """

    def calc_drift(self, a_L, a_s, v_L, v_s, u_L, u_s, t):
        self.a_drift =  np.append(self.a_drift, a_L - a_s)
        self.v_drift =  np.append(self.v_drift, v_L - v_s)
        self.u_drift =  np.append(self.u_drift, u_L - u_s)
        self.t_sync =  np.append(self.t_sync, t)

    def plot_drift(self, show):
        self.P.plot(1, [self.t_sync], [self.a_drift],
                    "Acceleration Drift between Large and Small Domains",
                    "Time (s)", "Acceleration Drift (m/s^2)",
                    ["Acceleration Drift"], 
                    [None, None], [None, None],
                    show)
        self.P.plot(1, [self.t_sync], [self.v_drift],
                    "Velocity Drift between Large and Small Domains",
                    "Time (s)", "Velocity Drift (m/s)",
                    ["Velocity Drift"], 
                    [None, None], [None, None],
                    show)
        self.P.plot(1, [self.t_sync], [self.u_drift],
                    "Displacement Drift between Large and Small Domains",
                    "Time (s)", "Displacement Drift (m)",
                    ["Displacement Drift"], 
                    [None, None], [None, None],
                    show)
        