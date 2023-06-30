import numpy as np
import matplotlib.pyplot as plt

""""
Look to Evaluate the stability of Our Multi-Time Step Integration Method

"""

class Domain:

    def __init__(self, n_nodes, n_elements, dx, E, rho, time):
        self.n_elem = n_elements
        self.n_node = n_nodes
        self.dx = dx
        self.v_prev = np.zeros(self.n_node)
        self.v_curr = np.zeros(self.n_node)
        self.u = np.zeros(self.n_node)
        self.t = 0
        self.dt_prev = 0
        self.dt_curr = 0
        self.E = E
        self.rho = rho
        self.t = time
        self.step = 0

    def prescribe_bcs(self):
        for i in range(self.n_node):
            pass

    def get_f_int(self):
        pass

    def solveq(self):
        # Leap frog time integration for a free node at time n
        a = -self.f_int / self.mass
        self.v_curr = self.v_prev + (a * ((self.dt_prev + self.dt_curr) / 2))
        self.u_curr = self.u_prev + self.v_curr * self.dt_curr
        self.step += 1


    def analyser(self):
        print(f" Let the analysing begin for step {self.step} ")
        #get_f_int(self)
        #prescribe_bcs(self)
        #solveq(self)
        #compute_energy(self)
       





if __name__=='__main__':
    print(" Let's try and prove our method is stable ")

    """ Monolithic Simulation """
    n_nodes = 376
    n_elements = 375
    E = 1e+10
    rho = 1e+03
    dx = 1
    time = 0

    infinite_bar = Domain(n_nodes, n_elements, dx, E, rho, time)
    Domain.analyser(infinite_bar)