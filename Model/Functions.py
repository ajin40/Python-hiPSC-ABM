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
        # begin with a low number of neighbors that can be revalued if the number of neighbors exceeds this value
        check_neighbors.max_neighbors = 2

    # clear all of the edges in the neighbor graph
    simulation.neighbor_graph.delete_edges(None)

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

    # assign the cells to bins so that the searches may be parallel
    bins, bins_help = Backend.put_cells_in_bins(simulation.number_cells, simulation.neighbor_distance, bins,
                                                bins_help, simulation.cell_locations)

    # this will run once and if all edges are included in edge_holder, the loop will break. if not this will
    # run a second time with an updated value for number of predicted neighbors such that all edges are included
    # initial condition for starting the while-loop
    max_neighbors = check_neighbors.max_neighbors + 1
    while max_neighbors > check_neighbors.max_neighbors:
        # revalue the static variable to the maximum number of neighbors
        check_neighbors.max_neighbors = max_neighbors

        # create a 3D array used to hold edges for each of the cells
        edge_holder = np.zeros((simulation.number_cells, check_neighbors.max_neighbors, 2), dtype=int)
        max_array = np.zeros(simulation.number_cells, dtype=int)

        # call the gpu version
        if simulation.parallel:
            # turn the following into arrays that can be interpreted by the gpu
            bins_cuda = cuda.to_device(bins)
            bins_help_cuda = cuda.to_device(bins_help)
            distance_cuda = cuda.to_device(simulation.neighbor_distance)
            edge_holder_cuda = cuda.to_device(edge_holder)
            locations_cuda = cuda.to_device(simulation.cell_locations)
            max_array_cuda = cuda.to_device(max_array)

            # sets up the correct allocation of threads and blocks
            threads_per_block = 72
            blocks_per_grid = math.ceil(simulation.number_cells / threads_per_block)

            # calls the cuda function with the given inputs
            Backend.check_neighbors_gpu[blocks_per_grid, threads_per_block](locations_cuda, bins_cuda, bins_help_cuda,
                                                                            distance_cuda, edge_holder_cuda,
                                                                            max_array_cuda)
            # returns the array back from the gpu
            edge_holder = edge_holder_cuda.copy_to_host()
            max_array = max_array_cuda.copy_to_host()

        # call the cpu version
        else:
            # call the jit function with the given inputs
            edge_holder, max_array = Backend.check_neighbors_cpu(simulation.number_cells, simulation.cell_locations,
                                                                 bins, bins_help, simulation.neighbor_distance,
                                                                 edge_holder, max_neighbors, max_array)
        # revalue the max_neighbors based on the output array
        max_neighbors = np.amax(max_array)

    # reshape the array so that the output is compatible with the igraph library and remove leftover zero columns
    edge_holder = edge_holder.reshape((-1, 2))
    edge_holder = edge_holder[~np.all(edge_holder == 0, axis=1)]

    # sort the array to remove duplicate edges produced by the parallel search method
    edge_holder = np.sort(edge_holder, axis=0)
    edge_holder = edge_holder[::2]

    # add the edges to the neighbor graph and simplify the graph to remove any extraneous loops or repreated edges
    simulation.neighbor_graph.add_edges(edge_holder)
    simulation.neighbor_graph.simplify()

    # calculate the total time elapsed for the function
    simulation.check_neighbors_time += time.time()
