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
    def __init__(self, domains, steps, v_L, v_s, pos_L, pos_s):
        # dt bars
        self.domains = domains
        self.steps = steps

        # square waves
        self.sq_L_0_00100 = v_L[0]
        self.sq_S_0_00100 = v_s[0]

        self.sq_L_0_00125 = v_L[1]
        self.sq_S_0_00125 = v_s[1]

        self.sq_L_0_00150 = v_L[2]
        self.sq_S_0_00150 = v_s[2]

        self.sq_L_pos = pos_L
        self.sq_S_pos = pos_s

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
    def sq_wave_0_00100(self):
        self.plot.plot(6, [self.Dvo.sq_L_pos, self.Dvo.sq_S_pos, self.Cho.sq_L_pos, self.Cho.sq_S_pos, self.Prop.sq_L_pos, self.Prop.sq_S_pos], 
                    [self.Dvo.sq_L_0_00100, self.Dvo.sq_S_0_00100, self.Cho.sq_L_0_00100, self.Cho.sq_S_0_00100, self.Prop.sq_L_0_00100, self.Prop.sq_S_0_00100],
                    "Fig 10: Square wave propagation through a 1D heterogeneous bar",
                    "Position (m)", "Velocity (m/s)",
                    ["Dvorak L", "Dvorak S", "Cho L", "Cho S", "Prop L", "Prop S"], 
                    [None, None], [None, None],
                    True)  

    # Figure 11 - Square Wave t=1.25ms
    def sq_wave_0_000125(self):
        self.plot.plot(6, [self.Dvo.sq_L_pos, self.Dvo.sq_S_pos, self.Cho.sq_L_pos, self.Cho.sq_S_pos, self.Prop.sq_L_pos, self.Prop.sq_S_pos], 
                    [self.Dvo.sq_L_0_00125, self.Dvo.sq_S_0_00125, self.Cho.sq_L_0_00125, self.Cho.sq_S_0_00125, self.Prop.sq_L_0_00125, self.Prop.sq_S_0_00125],
                    "Fig 11: Square wave propagation through a 1D heterogeneous bar",
                    "Position (m)", "Velocity (m/s)",
                    ["Dvorak L", "Dvorak S", "Cho L", "Cho S", "Prop L", "Prop S"], 
                    [None, None], [None, None],
                    True) 
        

    # Figure 12 - Square Wave t=1.5ms
    def sq_wave_0_000150(self):
        self.plot.plot(6, [self.Dvo.sq_L_pos, self.Dvo.sq_S_pos, self.Cho.sq_L_pos, self.Cho.sq_S_pos, self.Prop.sq_L_pos, self.Prop.sq_S_pos], 
                    [self.Dvo.sq_L_0_00150, self.Dvo.sq_S_0_00150, self.Cho.sq_L_0_00150, self.Cho.sq_S_0_00150, self.Prop.sq_L_0_00150, self.Prop.sq_S_0_00150],
                    "Fig 12: Square wave propagation through a 1D heterogeneous bar",
                    "Position (m)", "Velocity (m/s)",
                    ["Dvorak L", "Dvorak S", "Cho L", "Cho S", "Prop L", "Prop S"], 
                    [None, None], [None, None],
                    True) 

    def all_plots(self):
        self.dt_bars()
        self.sq_wave_0_00100()
        self.sq_wave_0_000125()
        self.sq_wave_0_000150()