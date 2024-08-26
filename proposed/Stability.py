import numpy as np
from literature import writeCSV
from proposed.Energy import SubdomainEnergy
from utils.Visualise import Plot

class Stability:

    def __init__(self, energy_L: SubdomainEnergy, energy_s: SubdomainEnergy):
        self.P = Plot()

        # Energy Balance for Large and Small Domains
        self.energy_L = energy_L
        self.energy_s = energy_s

        # Lagrange Multiplier Equivalence at Large Time Steps
        self.lm_L = np.array([0.0])
        self.lm_s = np.array([0.0])
        self.a_Gamma = np.array([0.0])
        self.a_f = np.array([0.0])
        # Coupling Energy
        self.dW_Gamma_L_dtL = np.array([0.0])
        self.dW_Gamma_S_dtL = np.array([0.0])
        self.dW_Gamma_S_dtS = np.array([0.0]) 
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

        # Check Acceleration Inconsistency
        self.eps = np.array([0.0])

    """
    Stability over a Large Time Step

    Plots
    - Accelerations for Proposed Method and Lagrange Multiplier
    - Acceleration Difference between Lagrange Multiplier and Proposed Method
    - Lagrange Multiplier for Large and Small Domains
    - Difference in Lagrange Multiplier Magnitude for Large and Small Domain
    - Interface Energy for Large and Small Domains
    - Difference in Interface Energy Magnitude for Large and Small Domain    
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
        self.P.plot(2, [self.t_sync, self.t_sync], [self.lm_L, self.lm_s],
                "Fig 6 (L): Lagrange Multiplier for Large and Small Domains",
                "Time (s)", "Lagrange Multiplier (N)",
                ["Large $\lambda_L$ ", "Small $\lambda_s$"], 
                [None, None], [None, None],
                True)
        if (csv):
            writeCSV("lm_L_equiv.csv", self.t_sync, self.lm_L, 't_L', 'lm_L')
            writeCSV("lm_s_equiv.csv", self.t_sync, self.lm_s, 't_L', 'lm_s')
        self.P.plot(2, [self.t_sync, self.t_sync], [self.a_Gamma, self.a_f],
                "Fig 6 (R): Accelerations for Proposed Method and Lagrange Multiplier",
                "Time (s)", "Acceleration (m/s$^2$)",
                ["a_Gamma", "a_f"], 
                [None, None], [None, None],
                True)

    def calc_dW_Gamma_dtL(self, Domain, mass, a_Gamma, a_Gamma_prev, f_int, f_int_prev, u, u_prev):
        if (Domain == "Large"):
            dlm = self.lm_L[-1] + self.lm_L[-2]
        else: # Small
            dlm = self.lm_s[-1] + self.lm_s[-2]

        du = u - u_prev
        dW_Gamma = 0.5 * du * (dlm + (mass * (a_Gamma + a_Gamma_prev)) + (f_int + f_int_prev))

        if (Domain == "Large"):
            self.dW_Gamma_L_dtL = np.append(self.dW_Gamma_L_dtL, dW_Gamma)
        else: # Small
            self.dW_Gamma_S_dtL = np.append(self.dW_Gamma_S_dtL, dW_Gamma)

    def plot_dW_Gamma_dtL(self, show, csv):
        if (csv):
            writeCSV("dW_Gamma_L.csv", self.t_sync, self.dW_Gamma_L_dtL, 't_L', 'dW_Gamma_L')
            writeCSV("dW_Gamma_s.csv", self.t_sync, self.dW_Gamma_S_dtL, 't_L', 'dW_Gamma_s')
        self.P.plot(1, [self.t_sync], [self.dW_Gamma_L_dtL + self.dW_Gamma_S_dtL],
                    "Fig 7 (L): Increment in Work Done across the Interface",
                    "Time (s)", "Interface Energy ($\delta \Gamma$J)",
                    ["$\delta W_{\Gamma L} + \delta W_{\Gamma s}$"], 
                    [None, None], [None, None],
                    show)  
        if (csv):
            writeCSV("dW_Gamma.csv", self.t_sync, self.dW_Gamma_L_dtL + self.dW_Gamma_S_dtL, 't_L', 'dW_Gamma')

    """
    Stability over a Small Time Step

    Plots
    - Lagrange Multiplier for Large and Small Domains over Small Time Steps
    - Difference in Lagrange Multiplier Magnitude for Large and Small Domain
    - Increment in Link Work for Large and Small Domains over Small Time Steps
    - Difference in Increment of Link Work Magnitude for Large and Small Domains over Small Time Steps    
    """
    def fint_Equiv(self, mass_L, mass_s, f_int_s, a_Gamma):
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

    def plot_dW_Link(self, show, csv):
        self.P.plot(2, [self.t_sync, self.t_sync], [self.dW_Link_L, self.dW_Link_s],
                    "Fig 7 (R): Increment in Work associated to coupling of subdomains recovered each $\Delta t_s$",
                    "Time (s)", "Increments in Coupling Work (J)",
                    ["Large $\delta W_{Link, L}}$", "Small $\delta W_{Link, s}}$"], 
                    [None, None], [None, None],
                    show)
        if (csv):
            writeCSV("dW_Link_L.csv", self.t_sync, self.dW_Link_L, 't_L', 'dW_Link_L')
            writeCSV("dW_Link_s.csv", self.t_sync, self.dW_Link_s, 't_L', 'dW_Link_s')

    def plot_eps(self, show, csv):
        self.P.plot(1, [self.t_small], [self.eps],
                    "APPDX: Acceleration Inconsistency",
                    "Time (s)", "Acceleration Inconsistency",
                    ["$\epsilon$"], 
                    [None, None], [None, None],
                    show)
        if (csv):
            writeCSV("eps.csv", self.t_small, self.eps, 't_s', 'eps')

    def calc_dW_Gamma_dtS(self, Domain, mass, a_Gamma, a_Gamma_prev, f_int, f_int_prev, u, u_prev, lm, lm_prev):
        if (Domain == "Small"):
            dlm = lm + lm_prev
        else: # Raise error
            assert False, "Domain must be Small"

        du = u - u_prev
        dW_Gamma = 0.5 * du * (dlm + (mass * (a_Gamma + a_Gamma_prev)) + (f_int + f_int_prev))

        return dW_Gamma

    def plot_dW_Gamma_dtS(self, show, csv):
        self.P.plot(2, [self.t_sync, self.t_sync], [self.dW_Gamma_L_dtL + self.dW_Gamma_S_dtL, self.dW_Gamma_L_dtL + self.dW_Gamma_S_dtS],
                    "Fig APPDX (L): Increment in Work Done across the Interface over a Large and Small Time Step",
                    "Time (s)", "Interface Energy ($\delta \Gamma$J)",
                    ["$\delta W_{\Gamma} \quad each \quad \Delta t_L$", "$\delta W_{\Gamma} \quad each \quad \Delta t_s$"], 
                    [None, None], [None, None],
                    show)
        if (csv):
            writeCSV("dW_Gamma_S.csv", self.t_sync, self.dW_Gamma_S_dtL, 't_L', 'dW_Gamma_s')

    """
    Checking of Drifting Conditions
    
    """
    def calc_drift(self, a_L, a_s, v_L, v_s, u_L, u_s, t):
        if (a_L - a_s == 0):
            self.a_drift = np.append(self.a_drift, 0)
        else:
            self.a_drift =  np.append(self.a_drift, (abs(a_L - a_s) / a_L)**2)
        if (v_L - v_s == 0):
            self.v_drift = np.append(self.v_drift, 0)
        else:
            self.v_drift =  np.append(self.v_drift, (abs(v_L - v_s)/ v_L)**2)
        if (u_L - u_s == 0):
            self.u_drift = np.append(self.u_drift, 0)
        else:
            self.u_drift =  np.append(self.u_drift, (abs(u_L - u_s) / u_L)**2)
        self.t_sync =  np.append(self.t_sync, t)

    def plot_drift(self, show, csv):
        self.P.plot(1, [self.t_sync], [self.a_drift],
                    "Acceleration Drift between Large and Small Domains",
                    "Time (s)", "Acceleration RSE (%)",
                    ["Acceleration Drift"], 
                    [None, None], [-10, 100],
                    show)
        if csv:
            writeCSV("a_drift.csv", self.t_sync, self.a_drift, 't_L', 'a_drift')
        
        self.P.plot(1, [self.t_sync], [self.v_drift],
                    "Velocity Drift between Large and Small Domains",
                    "Time (s)", "Velocity RSE (%)",
                    ["Velocity Drift"], 
                    [None, None], [-10, 100],
                    show)
        if csv:
            writeCSV("v_drift.csv", self.t_sync, self.v_drift, 't_L', 'v_drift')
        self.P.plot(1, [self.t_sync], [self.u_drift],
                    "Displacement Drift between Large and Small Domains",
                    "Time (s)", "Displacement RSE (%)",
                    ["Displacement Drift"], 
                    [None, None], [-10, 100],
                    show)
        if csv:
            writeCSV("u_drift.csv", self.t_sync, self.u_drift, 't_L', 'u_drift')
        