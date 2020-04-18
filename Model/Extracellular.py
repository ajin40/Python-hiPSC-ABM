import numpy as np
import random as r



class Extracellular:
    """ Initialization called once. Class holds information about each gradient for each simulation
    """
    def __init__(self, size, dx, dy, dz, diffuse_const, avg_initial, maximum, parallel):
        """ size: dimensions of the environment space
            dx: x direction step size
            dy: y direction step size
            dz: z direction step size
            diffuse_const: diffusion constant
            avg_initial: the average initial value for each concentration point
            maximum: the max concentration at a step
            parallel: whether gpu processing is being used
        """
        self.size = size
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.diffuse_const = diffuse_const
        self.avg_initial = avg_initial
        self.maximum = maximum
        self.parallel = parallel

        # squaring the approximation of the differential
        self.dx2 = dx ** 2
        self.dy2 = dy ** 2
        self.dz2 = dz ** 2

        # calculate the max time step size
        self.dt = (self.dx2 * self.dy2 * self.dz2) / (2 * diffuse_const * (self.dx2 + self.dy2 + self.dz2))

        # the points at which the diffusion values are calculated
        if dx == 0:
            x_steps = 1
        else:
            x_steps = int(self.size[0] / dx) + 1

        if dy == 0:
            y_steps = 1
        else:
            y_steps = int(self.size[1] / dy) + 1

        if dz == 0:
            z_steps = 1
        else:
            z_steps = int(self.size[2] / dz) + 1

        self.diffuse_values = np.zeros((x_steps, y_steps, z_steps))

    def initialize(self):
        """ Set up the environment space with a series
            of concentration values
        """
        # if the simulation is in 2D
        if self.size[2] == 0:
            for i in range(self.diffuse_values.shape[0]):
                for j in range(self.diffuse_values.shape[1]):
                    self.diffuse_values[i][j][0] = r.random() * self.avg_initial

        # if the simulation is in 3D
        else:
            for i in range(self.diffuse_values.shape[0]):
                for j in range(self.diffuse_values.shape[1]):
                    for k in range(self.diffuse_values.shape[2]):
                        self.diffuse_values[i][j][k] = r.random() * self.avg_initial


    def update(self, simulation):
        """ Updates the environment space by "smoothing"
            the concentrations of the space
        """
        time_steps = int(simulation.time_step / self.dt)

        a = self.diffuse_values

        for i in range(time_steps):
            x = (a[2:][1:-1][1:-1] - 2 * a[1:-1][1:-1][1:-1] + a[:-2][1:-1][1:-1]) / self.dx2
            y = (a[1:-1][2:][1:-1] - 2 * a[1:-1][1:-1][1:-1] + a[1:-1][:-2][1:-1]) / self.dy2
            z = (a[1:-1][1:-1][2:] - 2 * a[1:-1][1:-1][1:-1] + a[1:-1][1:-1][:-2]) / self.dz2

            self.diffuse_values[1:-1][1:-1][1:-1] = a[1:-1][1:-1][1:-1] + self.diffuse_const * self.dt * (x + y + z)