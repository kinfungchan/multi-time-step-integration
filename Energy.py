import numpy as np
from Visualise import Plot

class SubdomainEnergy:

    def __init__(self):

        # Subdomain Energy Balance
        self.KE = np.array([0.0]) # Kinetic Energy
        self.IE = np.array([0.0]) # Internal Energy
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

    def calc_total_energy(self):
        EBAL = self.KE[-1] + self.IE[-1]
        self.EBAL = np.append(self.EBAL, EBAL)

    def calc_energy_balance_subdomain(self, n_nodes, n_elem, mass, velocity, stress, E, dx):
        self.calc_KE_subdomain(n_nodes, mass, velocity)
        self.calc_IE_subdomain(n_elem, stress, E, dx)
        self.calc_total_energy()

