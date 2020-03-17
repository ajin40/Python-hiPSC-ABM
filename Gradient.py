#########################################################
# Name:    Gradient                                     #
# Author:  Jack Toppen                                  #
# Date:    3/17/20                                      #
#########################################################
import numpy as np
import random as r
import Parallel

"""
The Gradient class.
"""

class Gradient:
    """ called once holds important information about the
            simulation
        """
    def __init__(self, name, size, max_concentration, parallel):
        self.name = name
        self.size = size
        self.max_concentration = max_concentration
        self.parallel = parallel

        self.grid = np.zeros(self.size)


    def initialize_grid(self):

        if self.parallel:
            Parallel.initialize_grid_gpu(self)
        else:
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    for k in range(self.size[2]):
                        self.grid[i][j][k] = r.randint(0, self.max_concentration)


    def update_grid(self):

        if self.parallel:
            Parallel.update_grid_gpu(self)
        else:
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    for k in range(self.size[2]):
                        if self.grid[i][j][k] >= 1:
                            self.grid[i][j][k] += -1
