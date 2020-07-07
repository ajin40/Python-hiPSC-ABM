import numpy as np


class Extracellular:
    """ Initialization called once. Class holds information about each gradient for each simulation
    """
    def __init__(self, size, resolution, diffuse_const, parallel):
        """ size: dimensions of the environment space
            dx: x direction step size
            dy: y direction step size
            dz: z direction step size
            diffuse_const: diffusion constant
            parallel: whether gpu processing is being used
        """
        self.size = size
        self.dx = resolution[0]
        self.dy = resolution[1]
        self.dz = resolution[2]
        self.diffuse_const = diffuse_const
        self.parallel = parallel

        # squaring the approximation of the differential
        self.dx2 = self.dx ** 2
        self.dy2 = self.dy ** 2
        self.dz2 = self.dz ** 2

        # get the time step value for diffusion updates depending on whether 2D or 3D
        if self.size[2] == 0:
            self.dt = (self.dx2 * self.dy2) / (2 * diffuse_const * (self.dx2 + self.dy2))
        else:
            self.dt = (self.dx2 * self.dy2 * self.dz2) / (2 * diffuse_const * (self.dx2 + self.dy2 + self.dz2))

        # the points at which the diffusion values are calculated
        x_steps = int(self.size[0] / self.dx) + 1
        y_steps = int(self.size[1] / self.dy) + 1
        z_steps = int(self.size[2] / self.dz) + 1
        self.diffuse_values = np.zeros((x_steps, y_steps, z_steps))

    def update_gradient(self, simulation):
        """ Updates the environment space by "smoothing"
            the concentrations of the space
        """
        # get the number of times this will be run
        time_steps = int(simulation.time_step_value / self.dt)

        # perform the following operations on the diffusion points at each time step, depending on 2D or 3D
        # 2D
        if simulation.size[2] == 0:
            for i in range(time_steps):
                # make the variable name smaller for easier writing
                a = self.diffuse_values

                x = (a[2:, 1:-1] - 2 * a[1:-1, 1:-1] + a[:-2, 1:-1]) / self.dx2
                y = (a[1:-1, 2:] - 2 * a[1:-1, 1:-1] + a[1:-1, :-2]) / self.dy2

                # update the array
                self.diffuse_values[1:-1, 1:-1] = a[1:-1, 1:-1] + self.diffuse_const * self.dt * (x + y)
        # 3D
        else:
            for i in range(time_steps):
                # make the variable name smaller for easier writing
                a = self.diffuse_values

                x = (a[2:][1:-1][1:-1] - 2 * a[1:-1][1:-1][1:-1] + a[:-2][1:-1][1:-1]) / self.dx2
                y = (a[1:-1][2:][1:-1] - 2 * a[1:-1][1:-1][1:-1] + a[1:-1][:-2][1:-1]) / self.dy2
                z = (a[1:-1][1:-1][2:] - 2 * a[1:-1][1:-1][1:-1] + a[1:-1][1:-1][:-2]) / self.dz2

                # update the array
                self.diffuse_values[1:-1][1:-1][1:-1] = a[1:-1][1:-1][1:-1] + self.diffuse_const * self.dt * (x + y + z)
