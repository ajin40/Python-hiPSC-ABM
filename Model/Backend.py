import numpy as np
from numba import jit, cuda, prange


@cuda.jit(device=True)
def magnitude(location_one, location_two):
    """ This is a cuda device function for
        finding magnitude given two vectors
    """
    total = 0
    for i in range(0, 3):
        total += (location_one[i] - location_two[i]) ** 2

    return total ** 0.5


@jit(nopython=True)
def assign_bins_cpu(number_cells, distance, bins, bins_help, cell_locations):
    """ Helps speed up the process of assigning
        cells to bins
    """
    for i in range(number_cells):
        # offset bins by 2 to avoid missing cells
        block_location = cell_locations[i] // distance + np.array([2, 2, 2])
        x, y, z = int(block_location[0]), int(block_location[1]), int(block_location[2])

        # use the help array to place the cells in corresponding bins
        place = bins_help[x][y][z]

        # gives the cell's array location
        bins[x][y][z][place] = i

        # updates the total amount cells in a bin
        bins_help[x][y][z] += 1

    return bins, bins_help


@jit(nopython=True)
def check_neighbors_cpu(number_cells, cell_locations, bins, bins_help, distance, edge_holder, max_neighbors, max_array):
    """ This is the Numba optimized version of
        the check_neighbors function that runs
        solely on the cpu.
    """
    # loops over all cells, with the current cell being the focus of the search method
    for focus in prange(number_cells):
        # holds the total amount of edges for a given cell
        edge_counter = 0

        # offset bins by 2 to avoid missing cells
        block_location = cell_locations[focus] // distance + np.array([2, 2, 2])
        x, y, z = int(block_location[0]), int(block_location[1]), int(block_location[2])

        # loop over the bins that surround the current bin
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of cells for the current bin
                    bin_count = bins_help[x + i][y + j][z + k]

                    # go through that bin determining if a cell is a neighbor
                    for l in range(bin_count):
                        # get the index of the current cell in question
                        current = int(bins[x + i][y + j][z + k][l])

                        # check to see if that cell is within the search radius
                        vector = cell_locations[current] - cell_locations[focus]
                        if np.linalg.norm(vector) <= distance and focus != current:
                            # if within the bounds of the array add the edge
                            if edge_counter < max_neighbors:
                                # update the edge array and increase the index for the next addition
                                edge_holder[focus][edge_counter][0] = focus
                                edge_holder[focus][edge_counter][1] = current
                            edge_counter += 1

        # update the error array with the total number of edges per cell, if there are more neighbors than expected
        # the function will run again with the correct number of neighbors
        max_array[focus] = edge_counter

    # return the updated edges and the array with the counts of neighbors per cell
    return edge_holder, max_array


@cuda.jit
def check_neighbors_gpu(locations, bins, bins_help, distance, edge_holder, error_array):
    """ This is the parallelized function for checking
        neighbors that is run numerous times.
    """
    # get the index in the array
    focus = cuda.grid(1)

    # checks to see that position is in the array, double-check as GPUs can be weird sometimes
    if focus < locations.shape[0]:
        # holds the total amount of edges for a given cell
        edge_counter = 0

        # gets the block location based on how they were inputted
        x = int(locations[focus][0] / distance[0]) + 2
        y = int(locations[focus][1] / distance[0]) + 2
        z = int(locations[focus][2] / distance[0]) + 2

        # looks local bins as these are the ones containing the neighbors
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # gets the number of cells in each bin, int to prevent problems
                    bin_count = int(bins_help[x + i][y + j][z + k])

                    # loops over the cell indices in the current block
                    for l in range(bin_count):
                        # gets the index of the potential neighbor
                        current = int(bins[x + i][y + j][z + k][l])

                        # get the magnitude via the device function and make sure not the same cell
                        if magnitude(locations[focus], locations[current]) <= distance[0] and focus != current:
                            # assign the array location showing that this cell is a neighbor
                            edge_holder[focus][edge_counter][0] = focus
                            edge_holder[focus][edge_counter][1] = current
                            edge_counter += 1

        # update the error array with the total number of edges per cell, if there are more neighbors than expected
        # the function will run again with the correct number of neighbors
        error_array[focus] = edge_counter


@jit(nopython=True)
def jkr_neighbors_cpu(number_cells, distance, edge_holder, bins, bins_help, cell_locations, cell_radii, max_neighbors,
                      max_array):
    """ This is the Numba optimized version of
        the jkr_neighbors function that runs
        solely on the cpu.
    """
    # loops over all cells, with the current cell being the pivot of the search method
    for focus in range(number_cells):
        # holds the total amount of edges for a given cell
        edge_counter = 0

        # offset bins by 2 to avoid missing cells
        block_location = cell_locations[focus] // distance + np.array([2, 2, 2])
        x, y, z = int(block_location[0]), int(block_location[1]), int(block_location[2])

        # loop over the bins that surround the current bin
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of cells in a bin
                    bin_count = bins_help[x + i][y + j][z + k]

                    # go through the bin determining if a cell is a neighbor
                    for l in range(bin_count):
                        # get the index of the current cell in question
                        current = int(bins[x + i][y + j][z + k][l])

                        # get the magnitude of the distance between the cells
                        mag = np.linalg.norm(cell_locations[current] - cell_locations[focus])

                        # calculate the cell overlap
                        overlap = cell_radii[current] + cell_radii[focus] - mag

                        # if there is overlap and not the same cell add the edge
                        if overlap >= 0 and focus != current:
                            # if within the bounds of the array add the edge
                            if edge_counter < max_neighbors:
                                # update the edge array and increase the index for the next addition
                                edge_holder[focus][edge_counter][0] = focus
                                edge_holder[focus][edge_counter][1] = current
                            edge_counter += 1

        # update the error array with the total number of edges per cell, if there are more neighbors than expected
        # the function will run again with the correct number of neighbors
        max_array[focus] = edge_counter

    # return the updated edges and the array with the counts of neighbors per cell
    return edge_holder, max_array


@cuda.jit
def jkr_neighbors_gpu(locations, radii, bins, bins_help, distance, edge_holder, max_array):
    """ This CUDA kernel for finding jkr neighbors
        with the help of an NVIDIA gpu.
    """
    # get the index in the array
    focus = cuda.grid(1)

    # checks to see that position is in the array, double-check as GPUs can be weird sometimes
    if focus < locations.shape[0]:
        # holds the total amount of edges for a given cell
        edge_counter = 0

        # gets the block location based on how they were inputted
        x = int(locations[focus][0] / distance[0]) + 2
        y = int(locations[focus][1] / distance[0]) + 2
        z = int(locations[focus][2] / distance[0]) + 2

        # looks at the blocks surrounding the current block as these are the ones containing the neighbors
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # gets the number of cells in each bin, int to prevent problems
                    bin_count = int(bins_help[x + i][y + j][z + k])

                    # loops over the cell indices in the current block
                    for l in range(bin_count):
                        # gets the index of the potential neighbor
                        current = int(bins[x + i][y + j][z + k][l])

                        # get the magnitude via the device function
                        mag = magnitude(locations[focus], locations[current])

                        # calculate the cell overlap
                        overlap = radii[focus] + radii[current] - mag

                        # if overlap and not the same cell add it to the edges
                        if overlap >= 0 and focus != current:
                            # assign the array location showing that this cell is a neighbor
                            edge_holder[focus][edge_counter][0] = focus
                            edge_holder[focus][edge_counter][1] = current
                            edge_counter += 1

        # update the error array with the total number of edges per cell, if there are more neighbors than expected
        # the function will run again with the correct number of neighbors
        max_array[focus] = edge_counter
