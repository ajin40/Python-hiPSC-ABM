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
                    if self.grid[np.array([0]), np.array([i]), np.array([j])] >= 1:
                        self.grid[np.array([0]), np.array([i]), np.array([j])] += -1


    def initialize_grid_gpu(self):
        an_array = self.grid
        an_array_gpu = cuda.to_device(an_array)
        threads_per_block = (32, 32)
        blocks_per_grid_x = math.ceil(an_array.shape[0] / threads_per_block[0])
        blocks_per_grid_y = math.ceil(an_array.shape[1] / threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        initialize_grid_cuda[blocks_per_grid, threads_per_block](an_array_gpu)

        self.grid = an_array_gpu.copy_to_host()


    def update_grid_gpu(self):
        an_array = self.grid
        an_array_gpu = cuda.to_device(an_array)
        threads_per_block = (32, 32)
        blocks_per_grid_x = math.ceil(an_array.shape[0] / threads_per_block[0])
        blocks_per_grid_y = math.ceil(an_array.shape[1] / threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        update_grid_cuda[blocks_per_grid, threads_per_block](an_array_gpu)

        self.grid = an_array_gpu.copy_to_host()




@cuda.jit
def initialize_grid_cuda(grid_array):
    x, y = cuda.grid(2)
    if x < grid_array.shape[1] and y < grid_array.shape[2]:
        grid_array[0][x, y] += 10

@cuda.jit
def update_grid_cuda(grid_array):
    x, y = cuda.grid(2)
    if x < grid_array.shape[1] and y < grid_array.shape[2] and grid_array[0][x, y] >= 1:
        grid_array[0][x, y] -= 1