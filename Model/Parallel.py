from numba import cuda
import math
import numpy as np
import cupy as cp

def update_gradient_gpu(extracellular, simulation):
    """ This is a near identical function to the
        non-parallel one; however, this uses
        cupy which is identical to numpy, but
        it is run on a cuda gpu instead
    """
    # get the number of times this will be run
    time_steps = int(simulation.time_step_value / extracellular.dt)

    # make the variable name smaller for easier writing
    a = cp.asarray(extracellular.diffuse_values)

    # perform the following operations on the diffusion points at each time step
    for i in range(time_steps):

        x = (a[2:][1:-1][1:-1] - 2 * a[1:-1][1:-1][1:-1] + a[:-2][1:-1][1:-1]) / extracellular.dx2
        y = (a[1:-1][2:][1:-1] - 2 * a[1:-1][1:-1][1:-1] + a[1:-1][:-2][1:-1]) / extracellular.dy2
        z = (a[1:-1][1:-1][2:] - 2 * a[1:-1][1:-1][1:-1] + a[1:-1][1:-1][:-2]) / extracellular.dz2

        # update the array, assign a variable for ease of writing
        new_value = a[1:-1][1:-1][1:-1] + extracellular.diffuse_const * extracellular.dt * (x + y + z)
        a[1:-1][1:-1][1:-1] = new_value

    # turn it back into a numpy array
    extracellular.diffuse_values = cp.asnumpy(a)


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
    holder_size = 100

    # assigns values of -1 to denote a lack of cells
    blocks = np.ones((x, y, z, holder_size)) * -1

    # an array used to accelerate the cuda function by telling the function how many cells are in a given block
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

        # updates the total amount cells in a block
        blocks_help[location_x][location_y][location_z] = place + 1

    # turn the blocks array and the blocks_help array into a format to be sent to the gpu
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
    check_neighbors_cuda[blocks_per_grid, threads_per_block](location_array_cuda, blocks_cuda, blocks_help_cuda,
                                                             distance_cuda, output_cuda)
    # returns the array back from the gpu
    output = output_cuda.copy_to_host()

    # get the places where there are neighbors indicated by their index
    edges = np.argwhere(output != -1)

    # forms an edge between cells based on the results from edges
    for i in range(len(edges)):
        # sometimes values end up as floats of an int, this prevents that
        cell_1_index = int(edges[i][0])
        cell_2_index = int(output[edges[i][0]][edges[i][1]])

        # adds an edge between these two cells
        simulation.neighbor_graph.add_edge(simulation.cells[cell_1_index], simulation.cells[cell_2_index])


@cuda.jit
def check_neighbors_cuda(location_array, blocks, blocks_help, distance, output):
    """ This is the parallelized function for checking
        neighbors that is run numerous times.
    """
    # a provides the location on the array as it runs, essentially loops over the cells
    index_1 = cuda.grid(1)

    # checks to see that position is in the array, double-check as GPUs can be weird sometimes
    if index_1 < location_array.shape[0]:
        # gets the block location based on how they were inputted
        location_x = int(location_array[index_1][0] / distance[0]) + 1
        location_y = int(location_array[index_1][1] / distance[0]) + 1
        location_z = int(location_array[index_1][2] / distance[0]) + 1

        # holds the index of the neighbor as it's added to the output array, faster than another for loop
        place = 0

        # looks at the blocks surrounding the current block as these are the ones containing the neighbors
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # gets the number of cells in each block thanks to the helper array, int to prevent problems
                    number_cells = int(blocks_help[location_x + i][location_y + j][location_z + k])

                    # loops over the cell indices in the current block
                    for l in range(number_cells):
                        # gets the index of the potential neighbor
                        index_2 = int(blocks[location_x + i][location_y + j][location_z + k][l])

                        # get the magnitude via the device function and make sure not the same cell
                        if magnitude(location_array[index_1], location_array[index_2]) <= distance[0] and \
                                index_1 != index_2:
                            # assign the array location showing that this cell is a neighbor
                            output[index_1][place] = index_2
                            place += 1


@cuda.jit(device=True)
def magnitude(location_one, location_two):
    """ This is a cuda device function for
        finding magnitude give two vectors
    """
    total = 0
    for i in range(0, 3):
        total += (location_one[i] - location_two[i]) ** 2

    return total ** 0.5