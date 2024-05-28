from utils.Visualise import Plot

"""
In this module we reproduce the graphs from the paper:

K.F. Chan, N. Bombace, D. Sap, D. Wason, and N. Petrinic (2024),
A Multi-Time Stepping Algorithm for the Modelling of Heterogeneous
Structures with Explicit Time Integration, I.J. Num. Meth. in Eng.

"""

class Outputs:
    """
    Output returned from each method.
    """
    def __init__(self, domains, steps):
        # dt bars
        self.domains = domains
        self.steps = steps

class Paper: 
    """
    Takes the Output from each method and plots the graphs.
    """

    def __init__(self, Proposed: Outputs, Cho: Outputs, Dvorak: Outputs):
        self.Prop = Proposed
        self.Cho = Cho
        self.Dvo = Dvorak

        self.plot = Plot()  

    # Figure 9 - Comparison of Time Steps for Dvorak, Cho and Proposed Methods
    def dt_bars(self):
        combined_domains = self.Dvo.domains + self.Cho.domains + self.Prop.domains
        combined_steps = self.Dvo.steps + self.Cho.steps + self.Prop.steps
        self.plot.plot_dt_bars(combined_domains, combined_steps, True)

    # Figure 10 - Square Wave t=1.0ms
    def sq_wave_0_00100():
        pass

    # Figure 11 - Square Wave t=1.25ms
    def sq_wave_0_000125():
        pass

    # Figure 12 - Square Wave t=1.5ms
    def sq_wave_0_000150():
        pass

    def all_plots(self):
        self.dt_bars()
        # self.sq_wave_0_00100()
        # self.sq_wave_0_000125()
        # self.sq_wave_0_000150()