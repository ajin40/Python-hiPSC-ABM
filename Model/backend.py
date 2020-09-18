import numpy as np
from numba import jit, cuda, prange
import math
import random as r
from functools import wraps
import time


def remove_cell(simulation, index):
    """ given the index of a cell to remove, this will remove
        that from each array, graphs, and reduce the total
        number of cells
    """
    # go through all instance variable names
    for name in simulation.cell_array_names:
        # if the instance variable is 1-dimensional
        if simulation.__dict__[name].ndim == 1:
            simulation.__dict__[name] = np.delete(simulation.__dict__[name], index)
        # if the instance variable is 2-dimensional
        else:
            simulation.__dict__[name] = np.delete(simulation.__dict__[name], index, axis=0)

    # remove the particular index from the following graphs
    simulation.neighbor_graph.delete_vertices(index)
    simulation.jkr_graph.delete_vertices(index)

    # reduce the number of cells by 1
    simulation.number_cells -= 1


def divide_cell(simulation, index):
    """ Takes a cell or rather an index in the holder arrays
        and adds a new cell (index). This also updates factors
        such as size and counters.
    """
    # go through all instance variable names and copy the values to a newly appended index
    for name in simulation.cell_array_names:
        # get the instance variable from the class attribute dictionary
        value = simulation.__dict__[name][index]

        # if the instance variable is 1-dimensional
        if simulation.__dict__[name].ndim == 1:
            simulation.__dict__[name] = np.append(simulation.__dict__[name], value)
        # if the instance variable is 2-dimensional
        else:
            simulation.__dict__[name] = np.append(simulation.__dict__[name], [value], axis=0)

    # move the cells to positions that are representative of the new locations of daughter cells
    division_position = random_vector(simulation) * (simulation.max_radius - simulation.min_radius)
    simulation.cell_locations[index] += division_position
    simulation.cell_locations[-1] -= division_position

    # reduce both radii to minimum size and set the division counters to zero
    simulation.cell_radii[index] = simulation.cell_radii[-1] = simulation.min_radius
    simulation.cell_div_counter[index] = simulation.cell_div_counter[-1] = 0

    # add the new cell to the following graphs
    simulation.neighbor_graph.add_vertex()
    simulation.jkr_graph.add_vertex()

    # increase the number of cells by 1
    simulation.number_cells += 1


def assign_bins(simulation, distance, max_cells):
    """ generalizes cell locations to a bin within a multi-
        dimensional array, used for a parallel fixed-radius
        neighbor search
    """
    # if there is enough space for all cells that should be in a bin, break out of the loop. if there isn't
    # enough space update the amount of needed space and re-put the cells in bins. this will run once if the prediction
    # of max neighbors is correct, twice if it isn't the first time
    while True:
        # calculate the size of the array used to represent the bins and the bins helper array, include extra bins
        # for cells that may fall outside of the space
        bins_help_size = np.ceil(simulation.size / distance).astype(int) + np.array([5, 5, 5], dtype=int)
        bins_size = np.append(bins_help_size, max_cells)

        # create the arrays for "bins" and "bins_help"
        bins_help = np.zeros(bins_help_size, dtype=int)
        bins = np.empty(bins_size, dtype=int)

        # use jit function to speed up assignment
        bins, bins_help = assign_bins_cpu(simulation.number_cells, simulation.cell_locations, distance, bins, bins_help)

        # either break the loop if all cells were accounted for or revalue the maximum number of cells based on
        # the output of the function call
        new_max_cells = np.amax(bins_help)
        if max_cells >= new_max_cells:
            break
        else:
            max_cells = new_max_cells * 2

    # return the three arrays
    return bins, bins_help, max_cells


@jit(nopython=True)
def assign_bins_cpu(number_cells, cell_locations, distance, bins, bins_help):
    """ this is the just-in-time compiled version of assign_bins
        that runs solely on the cpu
    """
    # go through all cells
    for i in range(number_cells):
        # generalize location and offset it by 2 to avoid missing cells
        block_location = cell_locations[i] // distance + np.array([2, 2, 2])
        x, y, z = int(block_location[0]), int(block_location[1]), int(block_location[2])

        # use the help array to get the new index for the cell in the bin
        place = bins_help[x][y][z]

        # adds the index in the cell array to the bin
        bins[x][y][z][place] = i

        # update the number of cells in a bin
        bins_help[x][y][z] += 1

    # return the arrays now filled with cell indices
    return bins, bins_help


@jit(nopython=True, parallel=True)
def check_neighbors_cpu(number_cells, cell_locations, bins, bins_help, distance, edge_holder, if_nonzero, edge_count,
                        max_neighbors):
    """ this is the just-in-time compiled version of check_neighbors
        that runs in parallel on the cpu
    """
    # loops over all cells, with the current cell index being the focus
    for focus in prange(number_cells):
        # get the starting index for writing to the edge holder array
        start = focus * max_neighbors

        # holds the total amount of edges for a given cell
        edge_counter = 0

        # offset bins by 2 to avoid missing cells that fall outside the space
        block_location = cell_locations[focus] // distance + np.array([2, 2, 2])
        x, y, z = int(block_location[0]), int(block_location[1]), int(block_location[2])

        # loop over the bin the cell is in and the surrounding bins
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of cells for the current bin
                    bin_count = bins_help[x + i][y + j][z + k]

                    # go through that bin determining if a cell is a neighbor
                    for l in range(bin_count):
                        # get the index of the current cell in question
                        current = int(bins[x + i][y + j][z + k][l])

                        # check to see if that cell is within the search radius and not the same cell
                        vector = cell_locations[current] - cell_locations[focus]
                        if np.linalg.norm(vector) <= distance and focus < current:
                            # if within the bounds of the array, add the edge
                            if edge_counter < max_neighbors:
                                # get the index to place the edge
                                index = start + edge_counter

                                # update the edge array and identify that this edge is nonzero
                                edge_holder[index][0] = focus
                                edge_holder[index][1] = current
                                if_nonzero[index] = 1

                            # increase the count of edges for a cell and the index for the next edge
                            edge_counter += 1

        # update the array with number of edges for the cell
        edge_count[focus] = edge_counter

    # return the updated edges and the array with the counts of neighbors per cell
    return edge_holder, if_nonzero, edge_count


@cuda.jit
def check_neighbors_gpu(cell_locations, bins, bins_help, distance, edge_holder, if_nonzero, edge_count, max_neighbors):
    """ this is the cuda kernel for the check_neighbors function
        that runs on a NVIDIA gpu
    """
    # get the index in the array
    focus = cuda.grid(1)

    # get the starting index for writing to the edge holder array
    start = focus * max_neighbors[0]

    # checks to see that position is in the array
    if focus < cell_locations.shape[0]:
        # holds the total amount of edges for a given cell
        edge_counter = 0

        # offset bins by 2 to avoid missing cells that fall outside the space
        x = int(cell_locations[focus][0] / distance[0]) + 2
        y = int(cell_locations[focus][1] / distance[0]) + 2
        z = int(cell_locations[focus][2] / distance[0]) + 2

        # loop over the bin the cell is in and the surrounding bins
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of cells for the current bin
                    bin_count = int(bins_help[x + i][y + j][z + k])

                    # go through that bin determining if a cell is a neighbor
                    for l in range(bin_count):
                        # get the index of the current cell in question
                        current = int(bins[x + i][y + j][z + k][l])

                        # check to see if that cell is within the search radius and not the same cell
                        if magnitude(cell_locations[focus], cell_locations[current]) <= distance[0] and focus < current:
                            # if within the bounds of the array, add the edge
                            if edge_counter < max_neighbors[0]:
                                # get the index to place the edge
                                index = start + edge_counter

                                # update the edge array and identify that this edge is nonzero
                                edge_holder[index][0] = focus
                                edge_holder[index][1] = current
                                if_nonzero[index] = 1

                                # increase the count of edges for a cell and the index for the next edge
                            edge_counter += 1

        # update the array with number of edges for the cell
        edge_count[focus] = edge_counter


@jit(nopython=True, parallel=True)
def jkr_neighbors_cpu(number_cells, cell_locations, cell_radii, bins, bins_help, distance, edge_holder, if_nonzero,
                      edge_count, max_neighbors):
    """ this is the just-in-time compiled version of jkr_neighbors
        that runs in parallel on the cpu
    """
    # loops over all cells, with the current cell index being the focus
    for focus in prange(number_cells):
        # get the starting index for writing to the edge holder array
        start = focus * max_neighbors

        # holds the total amount of edges for a given cell
        edge_counter = 0

        # offset bins by 2 to avoid missing cells that fall outside the space
        block_location = cell_locations[focus] // distance + np.array([2, 2, 2])
        x, y, z = int(block_location[0]), int(block_location[1]), int(block_location[2])

        # loop over the bin the cell is in and the surrounding bins
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of cells for the current bin
                    bin_count = bins_help[x + i][y + j][z + k]

                    # go through that bin determining if a cell is a neighbor
                    for l in range(bin_count):
                        # get the index of the current cell in question
                        current = int(bins[x + i][y + j][z + k][l])

                        # get the magnitude of the distance vector between the cells
                        mag = np.linalg.norm(cell_locations[current] - cell_locations[focus])

                        # calculate the overlap of the cells
                        overlap = cell_radii[current] + cell_radii[focus] - mag

                        # if there is 0 or more overlap and not the same cell add the edge
                        if overlap >= 0 and focus < current:
                            # if within the bounds of the array, add the edge
                            if edge_counter < max_neighbors:
                                # get the index to place the edge
                                index = start + edge_counter

                                # update the edge array and identify that this edge is nonzero
                                edge_holder[index][0] = focus
                                edge_holder[index][1] = current
                                if_nonzero[index] = 1

                                # increase the count of edges for a cell and the index for the next edge
                            edge_counter += 1

        # update the array with number of edges for the cell
        edge_count[focus] = edge_counter

    # return the updated edges and the array with the counts of neighbors per cell
    return edge_holder, if_nonzero, edge_count


@cuda.jit
def jkr_neighbors_gpu(cell_locations, radii, bins, bins_help, distance, edge_holder, if_nonzero, edge_count,
                      max_neighbors):
    """ this is the cuda kernel for the jkr_neighbors function
        that runs on a NVIDIA gpu
    """
    # get the index in the array
    focus = cuda.grid(1)

    # get the starting index for writing to the edge holder array
    start = focus * max_neighbors[0]

    # checks to see that position is in the array
    if focus < cell_locations.shape[0]:
        # holds the total amount of edges for a given cell
        edge_counter = 0

        # offset bins by 2 to avoid missing cells that fall outside the space
        x = int(cell_locations[focus][0] / distance[0]) + 2
        y = int(cell_locations[focus][1] / distance[0]) + 2
        z = int(cell_locations[focus][2] / distance[0]) + 2

        # loop over the bin the cell is in and the surrounding bins
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of cells for the current bin
                    bin_count = int(bins_help[x + i][y + j][z + k])

                    # go through that bin determining if a cell is a neighbor
                    for l in range(bin_count):
                        # get the index of the current cell in question
                        current = int(bins[x + i][y + j][z + k][l])

                        # get the magnitude of the distance vector between the cells
                        mag = magnitude(cell_locations[focus], cell_locations[current])

                        # calculate the overlap of the cells
                        overlap = radii[focus] + radii[current] - mag

                        # if there is 0 or more overlap and not the same cell add the edge
                        if overlap >= 0 and focus < current:
                            # if within the bounds of the array, add the edge
                            if edge_counter < max_neighbors[0]:
                                # get the index to place the edge
                                index = start + edge_counter

                                # update the edge array and identify that this edge is nonzero
                                edge_holder[index][0] = focus
                                edge_holder[index][1] = current
                                if_nonzero[index] = 1

                                # increase the count of edges for a cell and the index for the next edge
                            edge_counter += 1

        # update the array with number of edges for the cell
        edge_count[focus] = edge_counter


@jit(nopython=True, parallel=True)
def get_forces_cpu(jkr_edges, delete_edges, cell_locations, cell_radii, jkr_forces, poisson, youngs,
                   adhesion_const):
    """ this is the just-in-time compiled version of get_forces
        that runs in parallel on the cpu
    """
    # loops over the jkr edges
    for edge_index in prange(len(jkr_edges)):
        # get the cell indices of the edge
        cell_1 = jkr_edges[edge_index][0]
        cell_2 = jkr_edges[edge_index][1]

        # get the vector between the centers of the cells and the magnitude of this vector
        vector = cell_locations[cell_1] - cell_locations[cell_2]
        mag = np.linalg.norm(vector)

        # get the total overlap of the cells
        overlap = cell_radii[cell_1] + cell_radii[cell_2] - mag

        # gets two values used for JKR
        e_hat = (((1 - poisson ** 2) / youngs) + ((1 - poisson ** 2) / youngs)) ** -1
        r_hat = ((1 / cell_radii[cell_1]) + (1 / cell_radii[cell_2])) ** -1

        # used to calculate the max adhesive distance after bond has been already formed
        overlap_ = (((math.pi * adhesion_const) / e_hat) ** (2 / 3)) * (r_hat ** (1 / 3))

        # get the nondimensionalized overlap
        d = overlap / overlap_

        # check to see if the cells will have a force interaction
        if d > -0.360562:
            # plug the value of d into the nondimensionalized equation for the JKR force
            f = (-0.0204 * d ** 3) + (0.4942 * d ** 2) + (1.0801 * d) - 1.324

            # convert from the nondimensionalization to find the adhesive force
            jkr_force = f * math.pi * adhesion_const * r_hat

            # calculate the normalized vector via a reduction as the parallel jit prefers this
            normal = np.array([0.0, 0.0, 0.0])
            if mag != 0:
                normal += vector / mag

            # adds the adhesive force as a vector in opposite directions to each cell's force holder
            jkr_forces[cell_1] += jkr_force * normal
            jkr_forces[cell_2] -= jkr_force * normal

        # remove the edge if the it fails to meet the criteria for distance, JKR simulating that the bond is broken
        else:
            delete_edges[edge_index] = 1

    # return the updated jkr forces and the edges to be deleted
    return jkr_forces, delete_edges


@cuda.jit
def get_forces_gpu(jkr_edges, delete_edges, cell_locations, cell_radii, jkr_forces, poisson, youngs, adhesion_const):
    """ this is the cuda kernel for the get_forces function
        that runs on a NVIDIA gpu
    """
    # get the index in the array
    edge_index = cuda.grid(1)

    # checks to see that position is in the array
    if edge_index < jkr_edges.shape[0]:
        # get the cell indices of the edge
        cell_1 = jkr_edges[edge_index][0]
        cell_2 = jkr_edges[edge_index][1]

        # get the locations of the two cells
        location_1 = cell_locations[cell_1]
        location_2 = cell_locations[cell_2]

        # get the magnitude of the vector
        mag = magnitude(location_1, location_2)

        # get the total overlap of the cells
        overlap = cell_radii[cell_1] + cell_radii[cell_2] - mag

        # gets two values used for JKR
        e_hat = (((1 - poisson[0] ** 2) / youngs[0]) + ((1 - poisson[0] ** 2) / youngs[0])) ** -1
        r_hat = ((1 / cell_radii[cell_1]) + (1 / cell_radii[cell_2])) ** -1

        # used to calculate the max adhesive distance after bond has been already formed
        overlap_ = (((math.pi * adhesion_const[0]) / e_hat) ** (2 / 3)) * (r_hat ** (1 / 3))

        # get the nondimensionalized overlap
        d = overlap / overlap_

        # check to see if the cells will have a force interaction
        if d > -0.360562:
            # plug the value of d into the nondimensionalized equation for the JKR force
            f = (-0.0204 * d ** 3) + (0.4942 * d ** 2) + (1.0801 * d) - 1.324

            # convert from the nondimensionalization to find the adhesive force
            jkr_force = f * math.pi * adhesion_const[0] * r_hat

            for i in range(3):
                # get the vector by axis between the two cells
                vector = location_1[i] - location_2[i]

                # if the magnitude is 0 use the zero vector, otherwise find the normalized vector
                if mag != 0:
                    normal = vector / mag
                else:
                    normal = 0

                # adds the adhesive force as a vector in opposite directions to each cell's force holder
                jkr_forces[cell_1][i] += jkr_force * normal
                jkr_forces[cell_2][i] -= jkr_force * normal

        # remove the edge if the it fails to meet the criteria for distance, JKR simulating that the bond is broken
        else:
            delete_edges[edge_index] = 1


@jit(nopython=True, parallel=True)
def apply_forces_cpu(number_cells, cell_jkr_force, cell_motility_force, cell_locations, cell_radii, viscosity, size,
                     move_time_step):
    """ this is the just-in-time compiled version of apply_forces
        that runs in parallel on the cpu
    """
    # loops over all cells using the explicit parallel loop from Numba
    for i in prange(number_cells):
        # stokes law for velocity based on force and fluid viscosity
        stokes_friction = 6 * math.pi * viscosity * cell_radii[i]

        # update the velocity of the cell based on the solution
        velocity = (cell_motility_force[i] + cell_jkr_force[i]) / stokes_friction

        # set the possible new location
        new_location = cell_locations[i] + velocity * move_time_step

        # loops over all directions of space
        for j in range(0, 3):
            # check if new location is in the space, if not return it to the space limits
            if new_location[j] > size[j]:
                cell_locations[i][j] = size[j]
            elif new_location[j] < 0:
                cell_locations[i][j] = 0.0
            else:
                cell_locations[i][j] = new_location[j]

    # return the updated cell locations
    return cell_locations


@cuda.jit
def apply_forces_gpu(cell_jkr_force, cell_motility_force, cell_locations, cell_radii, viscosity, size, move_time_step):
    """ This is the parallelized function for applying
        forces that is run numerous times.
    """
    # get the index in the array
    index = cuda.grid(1)

    # checks to see that position is in the array
    if index < cell_locations.shape[0]:
        # stokes law for velocity based on force and fluid viscosity
        stokes_friction = 6 * math.pi * viscosity[0] * cell_radii[index]

        # loops over all directions of space
        for i in range(3):
            # update the velocity of the cell based on the solution
            velocity = (cell_jkr_force[index][i] + cell_motility_force[index][i]) / stokes_friction

            # set the possible new location
            new_location = cell_locations[index][i] + velocity * move_time_step[0]

            # check if new location is in the space, if not return it to the space limits
            if new_location > size[i]:
                cell_locations[index][i] = size[i]
            elif new_location < 0:
                cell_locations[index][i] = 0.0
            else:
                cell_locations[index][i] = new_location


@jit(nopython=True, parallel=True)
def nearest_cpu(number_cells, cell_locations, bins, bins_help, distance, if_diff, gata6_high, nanog_high, nearest_gata6,
                nearest_nanog, nearest_diff):
    """ this is the just-in-time compiled version of nearest
        that runs in parallel on the cpu
    """
    # loops over all cells, with the current cell index being the focus
    for focus in prange(number_cells):
        # offset bins by 2 to avoid missing cells that fall outside the space
        block_location = cell_locations[focus] // distance + np.array([2, 2, 2])
        x, y, z = int(block_location[0]), int(block_location[1]), int(block_location[2])

        # initialize these variables with essentially nothing values and the distance as an initial comparison
        nearest_gata6_index, nearest_nanog_index, nearest_diff_index = np.nan, np.nan, np.nan
        nearest_gata6_dist, nearest_nanog_dist, nearest_diff_dist = distance * 2, distance * 2, distance * 2

        # loop over the bin the cell is in and the surrounding bin
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of cells for the current bin
                    bin_count = bins_help[x + i][y + j][z + k]

                    # go through that bin
                    for l in range(bin_count):
                        # get the index of the current cell in question
                        current = int(bins[x + i][y + j][z + k][l])

                        # check to see if that cell is within the search radius and not the same cell
                        mag = np.linalg.norm(cell_locations[current] - cell_locations[focus])
                        if mag <= distance and focus != current:
                            # update the nearest differentiated cell first
                            if if_diff[current]:
                                # if it's closer than the last cell, update the nearest magnitude and index
                                if mag < nearest_diff_dist:
                                    nearest_diff_index = current
                                    nearest_diff_dist = mag

                            # update the nearest gata6 high cell making sure not nanog high
                            elif gata6_high[current]:
                                if not nanog_high[current]:
                                    # if it's closer than the last cell, update the nearest magnitude and index
                                    if mag < nearest_gata6_dist:
                                        nearest_gata6_index = current
                                        nearest_gata6_dist = mag

                            # update the nearest nanog high cell
                            elif nanog_high[current]:
                                # if it's closer than the last cell, update the nearest magnitude and index
                                if mag < nearest_nanog_dist:
                                    nearest_nanog_index = current
                                    nearest_nanog_dist = mag

        # update the nearest cell of desired type
        nearest_gata6[focus] = nearest_gata6_index
        nearest_nanog[focus] = nearest_nanog_index
        nearest_diff[focus] = nearest_diff_index

    # return the updated edges
    return nearest_gata6, nearest_nanog, nearest_diff


@cuda.jit
def nearest_gpu(cell_locations, bins, bins_help, distance, if_diff, gata6_high, nanog_high, nearest_gata6,
                nearest_nanog, nearest_diff):
    """ this is the cuda kernel for the nearest function
        that runs on a NVIDIA gpu
    """
    # get the index in the array
    focus = cuda.grid(1)

    # checks to see that position is in the array
    if focus < cell_locations.shape[0]:
        # offset bins by 2 to avoid missing cells that fall outside the space
        x = int(cell_locations[focus][0] / distance[0]) + 2
        y = int(cell_locations[focus][1] / distance[0]) + 2
        z = int(cell_locations[focus][2] / distance[0]) + 2

        # initialize these variables with essentially nothing values and the distance as an initial comparison
        nearest_gata6_index, nearest_nanog_index, nearest_diff_index = np.nan, np.nan, np.nan
        nearest_gata6_dist, nearest_nanog_dist, nearest_diff_dist = distance[0] * 2, distance[0] * 2, distance[0] * 2

        # loop over the bin the cell is in and the surrounding bins
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of cells for the current bin
                    bin_count = int(bins_help[x + i][y + j][z + k])

                    # go through that bin determining if a cell is a neighbor
                    for l in range(bin_count):
                        # get the index of the current cell in question
                        current = int(bins[x + i][y + j][z + k][l])

                        # check to see if that cell is within the search radius and not the same cell
                        mag = magnitude(cell_locations[focus], cell_locations[current])
                        if mag <= distance[0] and focus != current:
                            # update the nearest differentiated cell first
                            if if_diff[current]:
                                # if it's closer than the last cell, update the nearest magnitude and index
                                if mag < nearest_diff_dist:
                                    nearest_diff_index = current
                                    nearest_diff_dist = mag

                            # update the nearest gata6 high cell making sure not nanog high
                            elif gata6_high[current]:
                                if not nanog_high[current]:
                                    # if it's closer than the last cell, update the nearest magnitude and index
                                    if mag < nearest_gata6_dist:
                                        nearest_gata6_index = current
                                        nearest_gata6_dist = mag

                            # update the nearest nanog high cell
                            elif nanog_high[current]:
                                # if it's closer than the last cell, update the nearest magnitude and index
                                if mag < nearest_nanog_dist:
                                    nearest_nanog_index = current
                                    nearest_nanog_dist = mag

        # update the nearest cell of certain types
        nearest_gata6[focus] = nearest_gata6_index
        nearest_nanog[focus] = nearest_nanog_index
        nearest_diff[focus] = nearest_diff_index


# @jit(nopython=True)
def setup_diffuse_bins_cpu(diffuse_locations, diffuse_radius, diffuse_bins, diffuse_bins_help):
    """ this is the just-in-time compiled version of
        setup_diffusion_bins that runs solely on the cpu
    """
    # loop over all diffusion points
    for i in range(diffuse_locations.shape[0]):
        for j in range(diffuse_locations.shape[1]):
            for k in range(diffuse_locations.shape[2]):
                # get the location in the bin array
                bin_location = diffuse_locations[i][j][k] // diffuse_radius + np.array([2, 2, 2])
                x, y, z = int(bin_location[0]), int(bin_location[1]), int(bin_location[2])

                # get the index of the where the point will be added
                place = diffuse_bins_help[x][y][z]

                # add the diffusion point to a corresponding bin and increase the place index
                diffuse_bins[x][y][z][place][0] = i
                diffuse_bins[x][y][z][place][1] = j
                diffuse_bins[x][y][z][place][2] = k
                diffuse_bins_help[x][y][z] += 1

    # return the arrays now filled with points
    return diffuse_bins, diffuse_bins_help


# @jit(nopython=True)
def update_diffusion_cpu(base, temp_base, time_steps, dt, dx2, dy2, dz2, diffuse, size, max_fgf4):
    """ this is the just-in-time compiled version of
        update_diffusion that runs solely on the cpu
    """
    # finite difference to solve laplacian diffusion equation
    # 2D
    if size[2] == 0:
        for i in range(time_steps):
            # add the temporary gradient to the main gradient to slowly increment concentrations
            base += temp_base

            # set max and min concentration values
            base[base > max_fgf4] = max_fgf4
            base[base < 0] = 0

            # mirror the edges of the diffusion space to create initial conditions
            base[:, 0, 1:-1] = base[:, 1, 1:-1]
            base[:, -1, 1:-1] = base[:, -2, 1:-1]
            base[0, :, 1:-1] = base[1, :, 1:-1]
            base[-1, :, 1:-1] = base[-2, :, 1:-1]

            # perform the first part of the calculation
            x = (base[2:, 1:-1, 1:-1] - 2 * base[1:-1, 1:-1, 1:-1] + base[:-2, 1:-1, 1:-1]) / dx2
            y = (base[1:-1, 2:, 1:-1] - 2 * base[1:-1, 1:-1, 1:-1] + base[1:-1, :-2, 1:-1]) / dy2

            # update the gradient array
            base[1:-1, 1:-1, 1:-1] = base[1:-1, 1:-1, 1:-1] + diffuse * dt * (x + y)

    # 3D
    else:
        for i in range(time_steps):
            # add the temporary gradient to the main gradient to slowly increment concentrations
            base += temp_base

            # mirror the edges of the diffusion space to create initial conditions
            # x and y direction
            base[:, 0, 1:-1] = base[:, 1, 1:-1]
            base[:, -1, 1:-1] = base[:, -2, 1:-1]
            base[0, :, 1:-1] = base[1, :, 1:-1]
            base[-1, :, 1:-1] = base[-2, :, 1:-1]

            # x and z direction
            base[:, 1:-1, 0] = base[:, 1:-1, 1]
            base[:, 1:-1, -1] = base[:, 1:-1, -2]
            base[0, 1:-1, :] = base[1, 1:-1, :]
            base[-1, 1:-1, :] = base[-2, 1:-1, :]

            # y and z direction
            base[1:-1, :, 0] = base[1:-1, :, 1]
            base[1:-1, :, -1] = base[1:-1, :, -2]
            base[1:-1, 0, :] = base[1:-1, 1, :]
            base[1:-1, -1, :] = base[1:-1, -2, :]

            # perform the first part of the calculation
            x = (base[2:, 1:-1, 1:-1] - 2 * base[1:-1, 1:-1, 1:-1] + base[:-2, 1:-1, 1:-1]) / dx2
            y = (base[1:-1, 2:, 1:-1] - 2 * base[1:-1, 1:-1, 1:-1] + base[1:-1, :-2, 1:-1]) / dy2
            z = (base[1:-1, 1:-1, 2:] - 2 * base[1:-1, 1:-1, 1:-1] + base[1:-1, 1:-1, :-2]) / dz2

            # update the gradient array
            base[1:-1, 1:-1, 1:-1] = base[1:-1, 1:-1, 1:-1] + diffuse * dt * (x + y + z)

    # return the gradient back to the simulation
    return base


@jit(nopython=True, parallel=True)
def highest_fgf4_cpu(number_cells, cell_locations, diffuse_bins, diffuse_bins_help, diffuse_locations, diffuse_radius,
                     cell_highest_fgf4, fgf4_values):
    """ This is the Numba optimized version of
        the highest_fgf4 function.
    """
    for focus in prange(number_cells):
        # offset bins by 2 to avoid missing points
        block_location = cell_locations[focus] // diffuse_radius + np.array([2, 2, 2])
        x, y, z = int(block_location[0]), int(block_location[1]), int(block_location[2])

        # create an initial value to check for the highest fgf4 point in a radius
        highest_index_x = np.nan
        highest_index_y = np.nan
        highest_index_z = np.nan
        highest_value = 0

        # loop over the bin the cell is in and the surrounding bins
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of cells for the current bin
                    bin_count = int(diffuse_bins_help[x + i][y + j][z + k])

                    # go through the bin determining if a cell is a neighbor
                    for l in range(bin_count):
                        # get the index of the current cell in question
                        x_ = int(diffuse_bins[x + i][y + j][z + k][l][0])
                        y_ = int(diffuse_bins[x + i][y + j][z + k][l][1])
                        z_ = int(diffuse_bins[x + i][y + j][z + k][l][2])

                        # check to see if that cell is within the search radius and not the same cell
                        mag = np.linalg.norm(diffuse_locations[x_][y_][z_] - cell_locations[focus])
                        if mag < diffuse_radius:
                            if fgf4_values[x_][y_][z_] > highest_value:
                                highest_index_x = x_
                                highest_index_y = y_
                                highest_index_z = z_
                                highest_value = fgf4_values[x_][y_][z_]

        # update the highest fgf4 diffusion point
        cell_highest_fgf4[focus][0] = highest_index_x
        cell_highest_fgf4[focus][1] = highest_index_y
        cell_highest_fgf4[focus][2] = highest_index_z

        # # create a holder for nearby diffusion points, a counter for the number, and values
        # holder = np.zeros((8, 3))
        # count = 0
        # values = np.zeros(8)
        #
        # # loop over the bins that surround the current bin
        # for i in range(-1, 2):
        #     for j in range(-1, 2):
        #         for k in range(-1, 2):
        #             # get the count of points in a bin
        #             bin_count = diffuse_bins_help[x + i][y + j][z + k]
        #
        #             # go through the bin determining if a bin is within the search radius
        #             for l in range(bin_count):
        #                 # get the indices of the current point in question
        #                 x_ = int(diffuse_bins[x + i][y + j][z + k][l][0])
        #                 y_ = int(diffuse_bins[x + i][y + j][z + k][l][1])
        #                 z_ = int(diffuse_bins[x + i][y + j][z + k][l][2])
        #
        #                 # check to see if that point is within the search radius
        #                 m = np.linalg.norm(diffuse_locations[x_][y_][z_] - cell_locations[pivot_index])
        #                 if m < diffuse_radius:
        #                     # if it is, add it to the holder and its value to values
        #                     holder[count][0] = x_
        #                     holder[count][1] = y_
        #                     holder[count][2] = z_
        #                     values[count] = fgf4_values[x_][y_][z_]
        #                     count += 1
        #
        # # get the sum of the array
        # sum_ = np.sum(values)
        #
        # # calculate probability of moving toward each point
        # if sum_ == 0:
        #     probs = np.zeros(8)
        # else:
        #     probs = values / sum_
        #
        # # randomly choose based on a custom distribution the diffusion point to move to
        # thing = np.random.choice(np.arange(8), p=probs)
        # print(probs)
        #
        # # get the index
        # index = holder[thing]
        #
        # # update the highest fgf4 diffusion point
        # cell_highest_fgf4[pivot_index][0] = index[0]
        # cell_highest_fgf4[pivot_index][1] = index[1]
        # cell_highest_fgf4[pivot_index][2] = index[2]

    # return the array back
    return cell_highest_fgf4


@cuda.jit
def highest_fgf4_gpu(cell_locations, diffuse_bins, diffuse_bins_help, diffuse_locations, diffuse_radius,
                     cell_highest_fgf4, fgf4_values):
    """ this is the cuda kernel for the highest_fgf4
        function that runs on a NVIDIA gpu
    """
    # get the index in the array
    focus = cuda.grid(1)

    # checks to see that position is in the array
    if focus < cell_locations.shape[0]:
        # offset bins by 2 to avoid missing cells that fall outside the space
        x = int(cell_locations[focus][0] / diffuse_radius[0]) + 2
        y = int(cell_locations[focus][1] / diffuse_radius[0]) + 2
        z = int(cell_locations[focus][2] / diffuse_radius[0]) + 2

        # create an initial value to check for the highest fgf4 point in a radius
        highest_index_x = np.nan
        highest_index_y = np.nan
        highest_index_z = np.nan
        highest_value = 0

        # loop over the bin the cell is in and the surrounding bins
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of cells for the current bin
                    bin_count = int(diffuse_bins_help[x + i][y + j][z + k])

                    # go through the bin determining if a cell is a neighbor
                    for l in range(bin_count):
                        # get the index of the current cell in question
                        x_ = int(diffuse_bins[x + i][y + j][z + k][l][0])
                        y_ = int(diffuse_bins[x + i][y + j][z + k][l][1])
                        z_ = int(diffuse_bins[x + i][y + j][z + k][l][2])

                        # check to see if that cell is within the search radius and not the same cell
                        mag = magnitude(diffuse_locations[x_][y_][z_], cell_locations[focus])
                        if mag < diffuse_radius[0]:
                            if fgf4_values[x_][y_][z_] > highest_value:
                                highest_index_x = x_
                                highest_index_y = y_
                                highest_index_z = z_
                                highest_value = fgf4_values[x_][y_][z_]

        # update the highest fgf4 diffusion point
        cell_highest_fgf4[focus][0] = highest_index_x
        cell_highest_fgf4[focus][1] = highest_index_y
        cell_highest_fgf4[focus][2] = highest_index_z


@jit(nopython=True, parallel=True)
def remove_gata6_edges(length, fds, edges, delete, delete_help):
    """ used by the outside cluster function to
        remove differentiated/gata6 high edges.
    """
    # go through edges
    for i in prange(length):
        # add to the delete array if either node is differentiated/gata6 high
        if fds[edges[i][0]][2] or fds[edges[i][1]][2]:
            delete[i] = i
            delete_help[i] = 1

    return delete, delete_help


@jit(nopython=True, parallel=True)
def nearest_cluster_cpu(number_cells, cell_locations, bins, bins_help, distance, if_nanog, cell_cluster_nearest,
                        members):
    """ This is the Numba optimized version of
        the nearest_cluster function.
    """
    # loops over all cells, with the current cell index being the focus
    for focus in prange(number_cells):
        if if_nanog[focus]:
            # offset bins by 2 to avoid missing cells that fall outside the space
            block_location = cell_locations[focus] // distance + np.array([2, 2, 2])
            x, y, z = int(block_location[0]), int(block_location[1]), int(block_location[2])

            # initialize these variables with essentially nothing values and the distance as an initial comparison
            nearest_cell_index = np.nan
            nearest_cell_dist = distance * 2

            # loop over the bin the cell is in and the surrounding bin
            for i in range(-1, 2):
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        # get the count of cells for the current bin
                        bin_count = bins_help[x + i][y + j][z + k]

                        # go through that bin
                        for l in range(bin_count):
                            # get the index of the current cell in question
                            current = int(bins[x + i][y + j][z + k][l])

                            # make sure not differentiated, not same cell, and not in the same cluster
                            if if_nanog[current] and focus != current and members[focus] != members[current]:
                                # check to see if that cell is within the search radius and not the same cell
                                mag = np.linalg.norm(cell_locations[current] - cell_locations[focus])
                                if mag <= distance:
                                    # if it's closer than the last cell, update the nearest magnitude and index
                                    if mag < nearest_cell_dist:
                                        nearest_cell_index = current
                                        nearest_cell_dist = mag

            # update the nearest cell of desired type
            cell_cluster_nearest[focus] = nearest_cell_index

    # return the updated edges
    return cell_cluster_nearest


@cuda.jit
def nearest_cluster_gpu(cell_locations, bins, bins_help, distance, if_nanog, cell_nearest_cluster, members):
    """ this is the cuda kernel for the nearest function
        that runs on a NVIDIA gpu
    """
    # get the index in the array
    focus = cuda.grid(1)

    # checks to see that position is in the array
    if focus < cell_locations.shape[0]:
        if if_nanog[focus]:
            # offset bins by 2 to avoid missing cells that fall outside the space
            x = int(cell_locations[focus][0] / distance[0]) + 2
            y = int(cell_locations[focus][1] / distance[0]) + 2
            z = int(cell_locations[focus][2] / distance[0]) + 2

            # initialize these variables with essentially nothing values and the distance as an initial comparison
            nearest_cell_index = np.nan
            nearest_cell_dist = distance[0] * 2

            # loop over the bin the cell is in and the surrounding bin
            for i in range(-1, 2):
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        # get the count of cells for the current bin
                        bin_count = bins_help[x + i][y + j][z + k]

                        # go through that bin
                        for l in range(bin_count):
                            # get the index of the current cell in question
                            current = int(bins[x + i][y + j][z + k][l])

                            # make sure not differentiated, not same cell, and not in the same cluster
                            if if_nanog[current] and focus != current and members[focus] != members[current]:
                                # check to see if that cell is within the search radius and not the same cell
                                mag = magnitude(cell_locations[current], cell_locations[focus])
                                if mag <= distance[0]:
                                    # if it's closer than the last cell, update the nearest magnitude and index
                                    if mag < nearest_cell_dist:
                                        nearest_cell_index = current
                                        nearest_cell_dist = mag

            # update the nearest cell of desired type
            cell_nearest_cluster[focus] = nearest_cell_index


@cuda.jit(device=True)
def magnitude(location_one, location_two):
    """ this is the cuda kernel device function that is used
        to calculate the magnitude between two points
    """
    # hold the value as the function runs
    count = 0

    # go through x, y, and z coordinates
    for i in range(0, 3):
        count += (location_one[i] - location_two[i]) ** 2

    # return the magnitude
    return count ** 0.5


def normal_vector(vector):
    """ returns the normalized vector
    """
    mag = np.linalg.norm(vector)
    if mag == 0:
        return np.array([0, 0, 0])
    else:
        return vector / mag


def random_vector(simulation):
    """ computes a random point on a unit sphere centered at the origin
        Returns - point [x,y,z]
    """
    # a random angle on the cell
    theta = r.random() * 2 * math.pi

    # determine if simulation is 2D or 3D
    if simulation.size[2] == 0:
        # 2D vector
        x, y, z = math.cos(theta), math.sin(theta), 0.0

    else:
        # 3D vector
        phi = r.random() * 2 * math.pi
        radius = math.cos(phi)
        x, y, z = radius * math.cos(theta), radius * math.sin(theta), math.sin(phi)

    # return random vector
    return np.array([x, y, z])


def record_time(function):
    """ a decorator used to time individual functions
    """
    @wraps(function)
    def wrap(simulation):
        # start time of the function
        simulation.function_times[function.__name__] = -1 * time.perf_counter()

        # call the actual function
        function(simulation)

        # end time of the function
        simulation.function_times[function.__name__] += time.perf_counter()

    return wrap
