import numpy as np
import random as r



class Extracellular:
    """ Initialization called once. Class holds information about each gradient for each simulation
    """
    def __init__(self, size, dx, dy, dz, diffuse_const, avg_initial, parallel):
        """ size: dimensions of the environment space
            dx: x direction step size
            dy: y direction step size
            dz: z direction step size
            diffuse_const: diffusion constant
            avg_initial: the average initial value for each concentration point
            parallel: whether gpu processing is being used
        """
        self.size = size
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.diffuse_const = diffuse_const
        self.avg_initial = avg_initial
        self.parallel = parallel

        # squaring the approximation of the differential
        self.dx2 = dx ** 2
        self.dy2 = dy ** 2
        self.dz2 = dz ** 2

        # calculate the max time step size
        self.dt = (self.dx2 * self.dy2 * self.dz2) / (2 * diffuse_const * (self.dx2 + self.dy2 + self.dz2))

        # the points at which the diffusion values are calculated
        x_size = int(self.size[0] / dx)
        y_size = int(self.size[1] / dy)
        z_size = int(self.size[2] / dz)
        self.diffuse_values = np.zeros((x_size, y_size, z_size))


    def initialize(self):
        """ Set up the environment space with a series
            of concentration values
        """
        # if the simulation is in 2D

        if self.size[2] == 0:
            for i in range(len(self.diffuse_values.size[0])):
                for j in range(len(self.diffuse_values.size[1])):
                    self.diffuse_values[i][j][0] = r.random() * self.avg_initial

        # if the simulation is in 3D
        else:
            for i in range(len(self.diffuse_values.size[0])):
                for j in range(len(self.diffuse_values.size[1])):
                    for k in range(len(self.diffuse_values.size[2])):
                        self.diffuse_values[i][j][k] = r.random() * self.avg_initial


    def update(self, simulation):
        """ Updates the environment space by "smoothing"
            the concentrations of the space
        """
        time_steps = simulation.time_step // self.dt

        a = self.diffuse_values

        for i in range(time_steps):
            x = (a[2:][1:-1][1:-1] - 2 * a[1:-1][1:-1][1:-1] + a[:-2][1:-1][1:-1]) / self.dx2
            y = (a[1:-1][2:][1:-1] - 2 * a[1:-1][1:-1][1:-1] + a[1:-1][:-2][1:-1]) / self.dy2
            z = (a[1:-1][1:-1][2:] - 2 * a[1:-1][1:-1][1:-1] + a[1:-1][1:-1][:-2]) / self.dz2

            self.diffuse_values[1:-1][1:-1][1:-1] = a[1:-1][1:-1][1:-1] + self.diffuse_const * self.dt * (x + y + z)



