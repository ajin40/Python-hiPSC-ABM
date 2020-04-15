import numpy as np
import random as r



class Gradient:
    """ Initialization called once. Class holds information about each gradient for each simulation
    """
    def __init__(self, size, dx, dy, dz, diffuse_const, avg_initial):
        """ size: dimensions of the environment space
            dx: x direction step size
            dy: y direction step size
            dz: z direction step size
            diffuse_const: diffusion constant
            avg_initial: the average initial value for each concentration point
        """
        self.size = size
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.diffuse_const = diffuse_const
        self.avg_initial = avg_initial

        # squaring the approximation of the differential
        self.dx2 = dx ** 2
        self.dy2 = dy ** 2
        self.dz2 = dz ** 2

        # calculate the max time step size
        self.dt = (self.dx2 * self.dy2 * self.dz2) / (2 * diffuse_const * (self.dx2 + self.dy2 + self.dz2))

        # the points at which the diffusion values are calculated
        self.diffuse_values = np.zeros((size[0] // dx, size[1] // dy, size[2] // dz))


    def initialize_diffusion(self):
        """ Set up the environment space with a series
            of concentration values
        """
        for i in range(len(self.diffuse_values[0])):
            for j in range(len(self.diffuse_values[1])):
                for k in range(len(self.diffuse_values[2])):
                    self.diffuse_values = r.random() * self.avg_initial


    def update_diffusion(self, simulation):
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



