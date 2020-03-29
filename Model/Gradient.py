import numpy as np
import random as r
from Model import Parallel


class Gradient:
    """ called once holds important information about the
        simulation
    """
    def __init__(self, name, size, max, parallel):
        """ name: where the cell is located on the grid "[x,y]"
            size: the size of the grid
            max_concentration: the maximum amount of molecule on a patch of the grid
            parallel: if parts of the simulation will be run in parallel
        """
        self.name = name
        self.size = size
        self.max = max
        self.parallel = parallel

        # create a grid of zeros with defined size
        self.grid = np.zeros(self.size)


    def initialize_grid(self):
        """ sets up the grid with initial concentrations of molecule
        """
        # currently this is setting the grid up with the same concentration everywhere until I implement the random
        # function for Numba/Cuda

        # if this will run parallel
        if self.parallel:
            Parallel.initialize_grid_gpu(self)

        # otherwise it'll loop over the entire volume of the grid
        else:
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    for k in range(self.size[2]):
                        self.grid[i][j][k] = r.randint(0, self.max)

    def update_grid(self):
        """ degrades every patch in the grid a uniform amount
        """

        # if this will run parallel
        if self.parallel:
            Parallel.update_grid_gpu(self)

        # otherwise it'll loop over the entire volume of the grid
        else:
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    for k in range(self.size[2]):
                        if self.grid[i][j][k] >= 1:
                            self.grid[i][j][k] += -1
