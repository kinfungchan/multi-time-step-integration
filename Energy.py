import numpy as np
from Visualise import Plot

class SubdomainEnergy:

    def __init__(self):

        # Subdomain Energy Balance
        self.KE = np.array([0.0]) # Kinetic Energy
        self.IE = np.array([0.0]) # Internal Energy
        self.XE = np.array([0.0]) # External Energy
        self.BV = np.array([0.0]) # Bulk Viscosity Energy
        self.EBAL = np.array([0.0]) # Energy Balance
        # Note to Account for External Energy

    def calc_KE_subdomain(self, n_nodes, mass, velocity):
        KE = 0.0
        for i in range(n_nodes):
            KE += 0.5 * mass[i] * velocity[i]**2
        self.KE = np.append(self.KE, KE)

    def calc_IE_subdomain(self, n_elem, stress, E, dx):
        IE = 0.0
        for j in range(n_elem):
            IE += ((0.5 * stress[j]**2) / E) * dx
        self.IE = np.append(self.IE, IE)

    def calc_BV_subdomain(self, n_elem, stress, E, dx):
        BV = 0.0
        for j in range(n_elem):
            BV += ((0.5 * stress[j]**2) / E) * dx
        self.BV = np.append(self.BV, BV)

    def calc_total_energy(self):
        XE = self.KE[-1] + self.IE[-1] - self.BV[-1]
        self.XE = np.append(self.XE, XE)
        self.EBAL = np.append(self.EBAL, self.KE[-1] + self.IE[-1] - self.BV[-1] - self.XE[-1]) # Account for External Energy for EBAL
        # self.EBAL = np.append(self.EBAL, self.KE[-1] + self.IE[-1] - self.XE[-1])

    def calc_energy_balance_subdomain(self, n_nodes, n_elem, mass, velocity, stress, bv, E, dx):
        self.calc_KE_subdomain(n_nodes, mass, velocity)
        self.calc_IE_subdomain(n_elem, stress, E, dx)
        self.calc_BV_subdomain(n_elem, bv, E, dx)
        self.calc_total_energy()

