#########################################################
# Name:    Gradient                                     #
# Author:  Jack Toppen                                  #
# Date:    3/17/20                                      #
#########################################################
import numpy as np
import random as r
import Parallel

class Gradient:
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
                    self.grid[i][j] = r.randint(0, self.max_concentration)


    def update_grid(self):

        if self.parallel:
            Parallel.update_grid_gpu(self)
        else:
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    if self.grid[i][j] >= 1:
                        self.grid[i][j] += -1
