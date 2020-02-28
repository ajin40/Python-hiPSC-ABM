#########################################################
# Name:    Model_CUDA                                   #
# Author:  Jack Toppen                                  #
# Date:    2/15/20                                      #
#########################################################
from numba import cuda
import math
import numpy as np


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


def check_edge_gpu(self):
    self.network.clear()


    rows = len(self.objects)
    columns = len(self.objects)
    edges_array = np.zeros((rows, columns))

    location_array = np.empty((0, 2), int)

    for i in range(len(self.objects)):
        location_array = np.append(location_array, np.array([self.objects[i].location]), axis=0)


    location_array_device_in = cuda.to_device(location_array)
    edges_array_device_in = cuda.to_device(edges_array)

    threads_per_block = (32, 32)
    blocks_per_grid_x = math.ceil(edges_array.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(edges_array.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    check_edge_cuda[blocks_per_grid, threads_per_block](location_array_device_in, edges_array_device_in)

    output = edges_array_device_in.copy_to_host()

    output = np.triu(output)


    edges = np.argwhere(output == 1)


    for i in range(len(self.objects)):
        self.network.add_node(self.objects[i])

    for i in range(len(edges)):
        self.network.add_edge(self.objects[edges[i][0]], self.objects[edges[i][1]])



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

@cuda.jit
def check_edge_cuda(locations, edges_array):
    a, b = cuda.grid(2)
    if a < edges_array.shape[0] and b < edges_array.shape[1]:
        location_x1 = locations[a][0]
        location_y1 = locations[a][1]
        location_x2 = locations[b][0]
        location_y2 = locations[b][1]
        mag = ((location_x1 - location_x2)**2 + (location_y1 - location_y2)**2) ** 0.5
        if a != b and mag <= 24:
            edges_array[a, b] = 1
