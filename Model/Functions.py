import numpy as np
import time
from numba import cuda
import math

import Backend


def check_neighbors(simulation):
    """ checks all of the distances between cells if it
        is less than a fixed value create a connection
        between two cells.
    """
    # start time of the function
    simulation.check_neighbors_time = -1 * time.time()

    # if a static variable has not been created to hold the maximum number of neighbors, create one
    if not hasattr(check_neighbors, "max_neighbors"):
        # begin with a low number of neighbors that can be revalued if the number of  neighbors exceed this value
        check_neighbors.max_neighbors = 2

    # clear all of the edges in the neighbor graph
    simulation.neighbor_graph.delete_edges(None)

    # create a 3D array used to hold edges for each of the cells
    edge_holder = np.zeros((simulation.number_cells, check_neighbors.max_neighbors, 2), dtype=int)

    # divides the space into bins and gives a holder of fixed size for each bin, the addition of 5 offsets
    # the space to prevent any errors, and 100 is the max cells for a bin which can be changed given errors
    bins_size = simulation.size // simulation.neighbor_distance + np.array([5, 5, 5])
    bins_size_help = tuple(bins_size.astype(int))
    bins_size = np.append(bins_size, 100)
    bins_size = tuple(bins_size.astype(int))

    # an empty array used to represent the bins the cells are put into
    bins = np.empty(bins_size, dtype=int)

    # an array used to accelerate the function by eliminating the lookup for number of cells in a bin
    bins_help = np.zeros(bins_size_help, dtype=int)

    # call the gpu version
    if simulation.parallel:
        #
        error_array = np.zeros(simulation.number_cells, dtype=int)

        # assign the cells to bins this is done all together compared to the cpu version
        bins, bins_help = Backend.put_cells_in_bins(simulation.number_cells, simulation.neighbor_distance, bins,
                                                    bins_help, simulation.cell_locations)

        # turn the following into arrays that can be interpreted by the gpu
        bins_cuda = cuda.to_device(bins)
        bins_help_cuda = cuda.to_device(bins_help)
        distance_cuda = cuda.to_device(simulation.neighbor_distance)
        edge_holder_cuda = cuda.to_device(edge_holder)
        locations_cuda = cuda.to_device(simulation.cell_locations)
        error_array_cuda = cuda.to_device(error_array)

        # sets up the correct allocation of threads and blocks
        threads_per_block = 72
        blocks_per_grid = math.ceil(simulation.number_cells / threads_per_block)

        # calls the cuda function with the given inputs
        Backend.check_neighbors_gpu[blocks_per_grid, threads_per_block](locations_cuda, bins_cuda, bins_help_cuda,
                                                                        distance_cuda, edge_holder_cuda,
                                                                        error_array_cuda)
        # returns the array back from the gpu
        edge_holder = edge_holder_cuda.copy_to_host()
        error_array = error_array_cuda.copy_to_host()

        while np.amax(error_array) > max_neighbors:
            check_neighbors.max_neighbors = np.amax(error_array)
            max_neighbors = check_neighbors.max_neighbors
            edge_holder = np.zeros((simulation.number_cells, max_neighbors, 2), dtype=int)
            error_array = np.zeros(simulation.number_cells, dtype=int)
            edge_holder_cuda = cuda.to_device(edge_holder)
            error_array_cuda = cuda.to_device(error_array)
            Backend.check_neighbors_gpu[blocks_per_grid, threads_per_block](locations_cuda, bins_cuda, bins_help_cuda,
                                                                            distance_cuda, edge_holder_cuda,
                                                                            error_array_cuda)
            edge_holder = edge_holder_cuda.copy_to_host()
            error_array = error_array_cuda.copy_to_host()
            print(check_neighbors.max_neighbors)

    # call the cpu version
    else:
        edge_holder = Backend.check_neighbors_cpu(simulation.number_cells, simulation.neighbor_distance, edge_holder,
                                                  bins, bins_help, simulation.cell_locations)

    # reshape the array for the igraph library, add the new edges, and remove any duplicate edges or loops
    edge_holder = edge_holder.reshape((-1, 2))
    edge_holder = edge_holder[~np.all(edge_holder == 0, axis=1)]

    if simulation.parallel:
        edge_holder = np.sort(edge_holder, axis=0)
        edge_holder = edge_holder[::2]

    print(len(edge_holder))

    simulation.neighbor_graph.add_edges(edge_holder)
    simulation.neighbor_graph.simplify()

    # calculate the total time elapsed for the function
    simulation.check_neighbors_time += time.time()
