import numpy as np
import matplotlib.pyplot as plt
import imageio 
import os

class Plot:
        
    def __init__(self):
        pass

    def plot(self, n_plots, x, y, title, xlabel, ylabel, legend, xlim, ylim, show):
        plt.style.use('ggplot')
        for i in range(n_plots):
                plt.plot(x[i], y[i])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.legend(legend)
        if show:
            plt.show()

    def plot_dt_bars(self, steps_L, steps_S, show):
        x = ['L', 'S']
        n_steps = 10 # Number of Steps to Plot
        data = np.empty((n_steps, 2))
        for i in range(n_steps):
            data[i] = np.array([steps_L[i], 
                                steps_S[i]])
            
        plt.figure(figsize=(10, 6))
        for i in range(1, n_steps):        
            plt.bar(x, data[i], bottom=np.sum(data[:i], axis=0), color=plt.cm.tab10(i), label=f'Local Step {i}')

        plt.ylabel('Time (s)')
        plt.title('Time Steps taken for New Multi-step')
        plt.legend()
        if show:
            plt.show()

class Animation:
     
    def __init__(self, Plot: Plot):
        self.P = Plot
        self.filenames_accel = []
        self.filenames_vel = []
        self.filenames_disp = []
        self.filenames_stress = []
     
    def save_single_plot(self, n_plots, x, y, title, xlabel, ylabel, filenames, n, t):
        filenames.append(f'FEM1D_{title}{n}.png')
        self.P.plot(n_plots, x, y, title, xlabel, ylabel,
                    t, [None, None], [None, None], False)
        plt.savefig(f'FEM1D_{title}{n}.png')
        plt.close()
            

    def create_gif(self, gif_name, filenames):
        with imageio.get_writer(gif_name, mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        for filename in set(filenames):
            os.remove(filename)

    def save_gifs(self):
        self.create_gif('Proposed_Multi-time-step_accel.gif', self.filenames_accel)
        self.create_gif('Proposed_Multi-time-step_vel.gif', self.filenames_vel)
        self.create_gif('Proposed_Multi-time-step_disp.gif', self.filenames_disp)
        self.create_gif('Proposed_Multi-time-step_stress.gif', self.filenames_stress)
        