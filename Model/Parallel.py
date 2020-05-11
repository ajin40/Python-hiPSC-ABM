from numba import cuda
import math
import numpy as np

def initialize_grid_gpu(simulation):
    """ the parallel form of "initialize_grid"
    """
    # sets up the correct allocation of threads and blocks
    threads_per_block = (10, 10, 10)
    blocks_per_grid_x = math.ceil(simulation.grid.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(simulation.grid.shape[1] / threads_per_block[1])
    blocks_per_grid_z = math.ceil(simulation.grid.shape[2] / threads_per_block[2])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

    # turns the grid into a form able to send to the gpu
    cuda_grid = cuda.to_device(simulation.grid)

    # calls the cuda function, gives the correct allocation and sends the grid
    initialize_grid_cuda[blocks_per_grid, threads_per_block](cuda_grid)

    # returns the grid from the cuda function and updates the simulation instance variable
    simulation.grid = cuda_grid.copy_to_host()


@cuda.jit
def initialize_grid_cuda(grid_array):
    """ this is the function being run in parallel
    """
    # a, b, and c provide the location on the array as this parallel function runs
    # the "3" tells cuda to return all three positions
    a, b, c = cuda.grid(3)

    # checks that the location of the current thread is within the array and updates the value
    if a < grid_array.shape[0] and b < grid_array.shape[1] and c < grid_array.shape[2]:
        grid_array[a][b][c] += 5.0


def update_grid_gpu(simulation):
    """ the parallel form of "update_grid"
    """
    # sets up the correct allocation of threads and blocks
    threads_per_block = (10, 10, 10)
    blocks_per_grid_x = math.ceil(simulation.grid.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(simulation.grid.shape[1] / threads_per_block[1])
    blocks_per_grid_z = math.ceil(simulation.grid.shape[2] / threads_per_block[2])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

    # turns the grid into a form able to send to the gpu
    cuda_grid = cuda.to_device(simulation.grid)

    # calls the cuda function, gives the correct allocation and sends the grid
    initialize_grid_cuda[blocks_per_grid, threads_per_block](cuda_grid)

    # returns the grid from the cuda function and updates the simulation instance variable
    simulation.grid = cuda_grid.copy_to_host()


@cuda.jit
def update_grid_cuda(grid_array):
    """ this is the function being run in parallel
    """
    # a, b, and c provide the location on the array as this parallel function runs
    # the "3" tells cuda to return all three positions
    a, b, c = cuda.grid(3)

    # checks that the location of the current thread is within the array and updates the value
    if a < grid_array.shape[0] and b < grid_array.shape[1] and c < grid_array.shape[2] and grid_array[a][b][c] > 0:
        grid_array[a][b][c] -= 1.0


def check_neighbors_gpu(simulation):
    """ checks all of the distances between cells if it
        is less than a fixed value create a connection
        between two cells.
    """

    # distance threshold between two cells to designate a neighbor and turns it into a gpu array
    distance = simulation.neighbor_distance
    distance_cuda = cuda.to_device(np.array([simulation.neighbor_distance]))

    # divides the space into blocks and gives a holder of fixed size for each block
    x = int(simulation.size[0] / distance + 3)
    y = int(simulation.size[1] / distance + 3)
    z = int(simulation.size[2] / distance + 3)
    size = 100

    # assigns values of -1 to denote emptiness
    blocks = np.ones((x, y, z, size)) * -1
    blocks_help = np.zeros((x, y, z))

    # assigns cells to blocks as a general location
    for i in range(len(simulation.cells)):
        # offset blocks by 1 to avoid missing cells
        location_x = int(simulation.cells[i].location[0] / distance) + 1
        location_y = int(simulation.cells[i].location[1] / distance) + 1
        location_z = int(simulation.cells[i].location[2] / distance) + 1

        # tries to place the cell in the holder for the block. if the holder's value is other than -1 it will move
        # to the next spot to see if it's empty
        place = 0
        while blocks[location_x][location_y][location_z][place] != -1:
            place += 1
        # gives the cell's array location
        blocks[location_x][location_y][location_z][place] = i
        blocks_help[location_x][location_y][location_z] = place + 1

    # turn the blocks array into a format to be sent to the gpu
    blocks_cuda = cuda.to_device(blocks)
    blocks_help_cuda = cuda.to_device(blocks_help)

    # loops over all over the cells and puts their locations into a holder array and turns it into a gpu array
    location_array = np.empty((0, 3), float)
    for i in range(len(simulation.cells)):
        location_array = np.append(location_array, [simulation.cells[i].location], axis=0)
    location_array_cuda = cuda.to_device(location_array)

    # an array used to hold neighbors sent back from the gpu to be read on the cpu
    output_array = np.ones((len(simulation.cells), 100)) * -1
    output_cuda = cuda.to_device(output_array)

    # sets up the correct allocation of threads and blocks
    threads_per_block = 72
    blocks_per_grid = math.ceil(location_array.size / threads_per_block)


    # calls the cuda function with the given inputs
    check_neighbors_cuda[blocks_per_grid, threads_per_block](location_array_cuda, blocks_cuda, blocks_help_cuda, distance_cuda, output_cuda)

    # returns the array back from the gpu
    output = output_cuda.copy_to_host()

    edges = np.argwhere(output != -1)

    # forms an edge between cells based on the results from edges
    for i in range(len(edges)):
        x = int(edges[i][0])
        y = int(output[edges[i][0]][edges[i][1]])
        simulation.neighbor_graph.add_edge(simulation.cells[x], simulation.cells[y])



@cuda.jit
def check_neighbors_cuda(location_array, blocks, blocks_help, distance, output):
    # a provides the location on the array as it runs
    a = cuda.grid(1)

    # checks to see that position is in the array
    if a < location_array.shape[0]:

        # gets the block location based on how they were inputted
        location_x = int(location_array[a][0] / distance[0]) + 1
        location_y = int(location_array[a][1] / distance[0]) + 1
        location_z = int(location_array[a][2] / distance[0]) + 1

        place = 0
        # looks at the blocks surrounding the current block as these are the ones containing the neighbors

        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # the current block in question
                    num = int(blocks_help[location_x + i][location_y + j][location_z + k])

                    # loops over the values in the current block
                    for l in range(num):

                        # makes sure the value is an integer had some problems before
                        value = int(blocks[location_x + i][location_y + j][location_z + k][l])

                        # get the magnitude and make sure no the same cell
                        if magnitude(location_array[a], location_array[value]) <= distance[0] and a != value:
                            # assign the array location showing that this cell is a neighbor
                            output[a][place] = value
                            place += 1



@cuda.jit(device=True)
def magnitude(location_one, location_two):
    total = 0
    for i in range(0, 3):
        total += (location_one[i] - location_two[i]) ** 2

    return total ** 0.5