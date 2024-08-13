import numpy as np

class History:
    '''
    History class to store results for all time steps
    '''
    def __init__(self, coordinates, n_nodes, n_elem):
        self.t = np.array([0.0])
        self.coordinates = np.zeros((1, n_nodes), dtype=float)
        self.coordinates[0] = coordinates # Append first time step coordinates

        # Initialize 3D arrays for acceleration, velocity, and displacement
        self.accel = np.zeros((1, n_nodes), dtype=float)
        self.vel = np.zeros((1, n_nodes), dtype=float)
        self.displ = np.zeros((1, n_nodes), dtype=float)

        # Initialize 2D arrays for stress components
        self.stress = np.zeros((1, n_elem), dtype=float)
        self.strain = np.zeros((1, n_elem), dtype=float)

    def append_timestep(self, t, coordinates, accel, vel, displ, stress, strain):
        self.t = np.append(self.t, t)
        self.coordinates = np.vstack((self.coordinates, [coordinates]))
        
        # Append new time step data to 3D arrays
        self.accel = np.vstack((self.accel, [accel]))
        self.vel = np.vstack((self.vel, [vel]))
        self.displ = np.vstack((self.displ, [displ]))

        # Append new time step data to 2D arrays
        self.stress = np.vstack((self.stress, [stress]))
        self.strain = np.vstack((self.strain, [strain]))

    def write_to_csv(self, filename):
        '''
        Write history data to a csv file
        '''
        np.savetxt(filename + "_t.csv", self.t, delimiter=",")
        np.savetxt(filename + "_coordinates.csv", self.coordinates, delimiter=",")
        np.savetxt(filename + "_accel.csv", self.accel, delimiter=",")
        np.savetxt(filename + "_vel.csv", self.vel, delimiter=",")
        np.savetxt(filename + "_displ.csv", self.displ, delimiter=",")
        np.savetxt(filename + "_stress.csv", self.stress, delimiter=",")
        np.savetxt(filename + "_strain.csv", self.strain, delimiter=",")
