import numpy as np
from numba import cuda
import random as r
import math

class Gradient:
    def __init__(self, size, max_concentration, parallel):
        self.size = size
        self.max_concentration = max_concentration
        self.parallel = parallel

        self.grid = np.zeros(self.size)



    def initialize_grid(self):

        if self.parallel:
            self.initialize_grid_gpu()
        else:
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    self.grid[i][j] = r.randint(0, self.max_concentration)



    def update_grid(self):

        if self.parallel:
            self.update_grid_gpu()
        else:
            for i in range(self.size[1]):
                for j in range(self.size[2]):
                    if self.grid[i][j] >= 1:
                        self.grid[i][j] += -1


    def initialize_grid_gpu(self):
        array = cuda.to_device(self.grid)

        threads_per_block = (32, 32)
        blocks_per_grid_x = math.ceil(self.grid.shape[0] / threads_per_block[0])
        blocks_per_grid_y = math.ceil(self.grid.shape[1] / threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        initialize_grid_cuda[blocks_per_grid, threads_per_block](array)

        self.grid = array.copy_to_host()


    def update_grid_gpu(self):
        array = cuda.to_device(self.grid)

        threads_per_block = (32, 32)
        blocks_per_grid_x = math.ceil(self.grid.shape[0] / threads_per_block[0])
        blocks_per_grid_y = math.ceil(self.grid.shape[1] / threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        initialize_grid_cuda[blocks_per_grid, threads_per_block](array)

        self.grid = array.copy_to_host()




@cuda.jit
def initialize_grid_cuda(grid_array):
    a, b = cuda.grid(2)
    if a < grid_array.shape[0] and b < grid_array.shape[1]:
        grid_array[a, b] += 10

@cuda.jit
def update_grid_cuda(grid_array):
    a, b = cuda.grid(2)
    if a < grid_array.shape[0] and b < grid_array.shape[1] and grid_array[a, b] >= 1:
        grid_array[a, b] -= 1