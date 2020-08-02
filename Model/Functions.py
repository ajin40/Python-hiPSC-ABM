import numpy as np
import time
from numba import cuda
import math

import Backend


def assign_bins(simulation, distance):
    """ generalizes cell locations to a bin within a multi-
        dimensional array, used for a parallel fixed-radius
        neighbor search
    """
    # if a static variable has not been created to hold the maximum number of cells in a bin, create one
    if not hasattr(assign_bins, "max_cells"):
        # begin with a low number of cells that can be revalued if the max number of cells exceeds this value
        assign_bins.max_cells = 1

    # if there is enough space for all cells that should be in a bin, break out of the loop. if there isn't
    # enough space update the amount of needed space and re-put the cells in bins
    while True:
        # calculate the size of the array used to represent the bins and the bins helper array, include extra bins
        # for cells that may fall outside of the space
        bins_size = simulation.size // distance + np.array([5, 5, 5])
        bins_size_help = tuple(bins_size.astype(int))
        bins_size = np.append(bins_size, assign_bins.max_cells)
        bins_size = tuple(bins_size.astype(int))

        # an empty array used to represent the bins the cells are put into
        bins = np.empty(bins_size, dtype=int)

        # an array used to accelerate the search method by eliminating the lookup for number of cells in a bin
        bins_help = np.zeros(bins_size_help, dtype=int)

        # assign the cells to bins so that the searches may be parallel
        bins, bins_help = Backend.assign_bins_cpu(simulation.number_cells, simulation.neighbor_distance, bins,
                                                  bins_help, simulation.cell_locations)

        # either break the loop if all cells were accounted for or revalue the maximum number of cells based on
        # the output of the function call
        max_cells = np.amax(bins_help)
        if assign_bins.max_cells >= max_cells:
            break
        else:
            assign_bins.max_cells = max_cells

    # return the two arrays
    return bins, bins_help


def check_neighbors(simulation):
    """ checks all of the distances between cells if it
        is less than a fixed value create a connection
        between two cells.
    """
    # start time of the function
    simulation.check_neighbors_time = -1 * time.time()

    # if a static variable has not been created to hold the maximum number of neighbors, create one
    if not hasattr(check_neighbors, "max_neighbors"):
        # begin with a low number of neighbors that can be revalued if the max number of neighbors exceeds this value
        check_neighbors.max_neighbors = 1

    # clear all of the edges in the neighbor graph
    simulation.neighbor_graph.delete_edges(None)

    # calls the function that generates an array of bins that generalize the cell locations in addition to a
    # helper array that assists the search method in counting cells in a particular bin
    bins, bins_help = assign_bins(simulation, simulation.neighbor_distance)

    # this will run once and if all edges are included in edge_holder, the loop will break. if not this will
    # run a second time with an updated value for number of predicted neighbors such that all edges are included
    while True:
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

        # either break the loop if all neighbors were accounted for or revalue the maximum number of neighbors
        # based on the output of the function call
        max_neighbors = np.amax(max_array)
        if check_neighbors.max_neighbors >= max_neighbors:
            break
        else:
            check_neighbors.max_neighbors = max_neighbors

    # reshape the array so that the output is compatible with the igraph library and remove leftover zero columns
    edge_holder = edge_holder.reshape((-1, 2))
    edge_holder = edge_holder[~np.all(edge_holder == 0, axis=1)]

    # sort the array to remove duplicate edges produced by the parallel search method
    edge_holder = np.sort(edge_holder)
    edge_holder = np.sort(edge_holder, axis=0)
    edge_holder = edge_holder[::2]

    # add the edges to the neighbor graph and simplify the graph to remove any extraneous loops or repeated edges
    simulation.neighbor_graph.add_edges(edge_holder)

    # calculate the total time elapsed for the function
    simulation.check_neighbors_time += time.time()

