import numpy as np
from numba import jit, cuda, prange
import math


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
def check_neighbors_cpu(number_cells, cell_locations, bins, bins_help, distance, edge_holder, edge_count,
                        max_neighbors):
    """ this is the just-in-time compiled version of check_neighbors
        that runs in parallel on the cpu
    """
    # loops over all cells, with the current cell index being the focus
    for focus in prange(number_cells):
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
                        if np.linalg.norm(vector) <= distance and focus != current:
                            # if within the bounds of the array, add the edge
                            if edge_counter < max_neighbors:
                                # update the edge array
                                edge_holder[focus][edge_counter][0] = focus
                                edge_holder[focus][edge_counter][1] = current

                            # increase the count of edges for placement of next edge and ensuring all edges are
                            # counted
                            edge_counter += 1

        # update the array with number of edges for the cell
        edge_count[focus] = edge_counter

    # return the updated edges and the array with the counts of neighbors per cell
    return edge_holder, edge_count


@cuda.jit
def check_neighbors_gpu(locations, bins, bins_help, distance, edge_holder, edge_count):
    """ this is the cuda kernel for the check_neighbors function
        that runs on a NVIDIA gpu
    """
    # get the index in the array
    focus = cuda.grid(1)

    # checks to see that position is in the array
    if focus < locations.shape[0]:
        # holds the total amount of edges for a given cell
        edge_counter = 0

        # offset bins by 2 to avoid missing cells that fall outside the space
        x = int(locations[focus][0] / distance[0]) + 2
        y = int(locations[focus][1] / distance[0]) + 2
        z = int(locations[focus][2] / distance[0]) + 2

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
                        if magnitude(locations[focus], locations[current]) <= distance[0] and focus != current:
                            # update the edge array
                            edge_holder[focus][edge_counter][0] = focus
                            edge_holder[focus][edge_counter][1] = current

                            # increase the count of edges for a cell
                            edge_counter += 1

        # update the array with number of edges for the cell
        edge_count[focus] = edge_counter


@jit(nopython=True, parallel=True)
def jkr_neighbors_cpu(number_cells, cell_locations, cell_radii, bins, bins_help, distance, edge_holder, edge_count,
                      max_neighbors):
    """ this is the just-in-time compiled version of jkr_neighbors
        that runs in parallel on the cpu
    """
    # loops over all cells, with the current cell index being the focus
    for focus in prange(number_cells):
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
                        if overlap >= 0 and focus != current:
                            # if within the bounds of the array add the edge
                            if edge_counter < max_neighbors:
                                # update the edge array
                                edge_holder[focus][edge_counter][0] = focus
                                edge_holder[focus][edge_counter][1] = current
                            # increase the count of edges for placement of next edge and ensuring all edges are
                            # counted
                            edge_counter += 1

        # update the array with number of edges for the cell
        edge_count[focus] = edge_counter

    # return the updated edges and the array with the counts of neighbors per cell
    return edge_holder, edge_count


@cuda.jit
def jkr_neighbors_gpu(locations, radii, bins, bins_help, distance, edge_holder, edge_count):
    """ this is the cuda kernel for the jkr_neighbors function
        that runs on a NVIDIA gpu
    """
    # get the index in the array
    focus = cuda.grid(1)

    # checks to see that position is in the array
    if focus < locations.shape[0]:
        # holds the total amount of edges for a given cell
        edge_counter = 0

        # offset bins by 2 to avoid missing cells that fall outside the space
        x = int(locations[focus][0] / distance[0]) + 2
        y = int(locations[focus][1] / distance[0]) + 2
        z = int(locations[focus][2] / distance[0]) + 2

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
                        mag = magnitude(locations[focus], locations[current])

                        # calculate the overlap of the cells
                        overlap = radii[focus] + radii[current] - mag

                        # if there is 0 or more overlap and not the same cell add the edge
                        if overlap >= 0 and focus != current:
                            # update the edge array
                            edge_holder[focus][edge_counter][0] = focus
                            edge_holder[focus][edge_counter][1] = current

                            # increase the count of edges for a cell
                            edge_counter += 1

        # update the array with number of edges for the cell
        edge_count[focus] = edge_counter


@jit(nopython=True, parallel=True)
def get_forces_cpu(jkr_edges, delete_jkr_edges, poisson, youngs_mod, adhesion_const, cell_locations,
                   cell_radii, cell_jkr_force):
    """ This is the Numba optimized version of
        the get_forces function that runs
        solely on the cpu.
    """
    # loops over the jkr edges in parallel
    for i in prange(len(jkr_edges)):
        # get the indices of the nodes in the edge
        index_1 = jkr_edges[i][0]
        index_2 = jkr_edges[i][1]

        # hold the vector between the centers of the cells and the magnitude of this vector
        disp_vector = cell_locations[index_1] - cell_locations[index_2]
        mag = np.linalg.norm(disp_vector)
        normal = np.array([0.0, 0.0, 0.0])

        # the parallel jit prefers reductions so it's better to initialize the value of the normal and revalue it
        if mag != 0:
            normal += disp_vector / mag

        # get the total overlap of the cells used later in calculations
        overlap = cell_radii[index_1] + cell_radii[index_2] - mag

        # gets two values used for JKR
        e_hat = (((1 - poisson ** 2) / youngs_mod) + ((1 - poisson ** 2) / youngs_mod)) ** -1
        r_hat = ((1 / cell_radii[index_1]) + (1 / cell_radii[index_2])) ** -1

        # used to calculate the max adhesive distance after bond has been already formed
        overlap_ = (((math.pi * adhesion_const) / e_hat) ** (2 / 3)) * (r_hat ** (1 / 3))

        # get the nondimensionalized overlap, used for later calculations and checks
        # also for the use of a polynomial approximation of the force
        d = overlap / overlap_

        # check to see if the cells will have a force interaction
        if d > -0.360562:
            # plug the value of d into the nondimensionalized equation for the JKR force
            f = (-0.0204 * d ** 3) + (0.4942 * d ** 2) + (1.0801 * d) - 1.324

            # convert from the nondimensionalization to find the adhesive force
            jkr_force = f * math.pi * adhesion_const * r_hat

            # adds the adhesive force as a vector in opposite directions to each cell's force holder
            cell_jkr_force[index_1] += jkr_force * normal
            cell_jkr_force[index_2] -= jkr_force * normal

        # remove the edge if the it fails to meet the criteria for distance, JKR simulating that
        # the bond is broken
        else:
            delete_jkr_edges[i] = i

    # return the updated jkr forces and the edges to be deleted
    return cell_jkr_force, delete_jkr_edges


@cuda.jit
def get_forces_gpu(jkr_edges, delete_jkr_edges, locations, radii, forces, poisson, youngs, adhesion_const):
    """ The parallelized function that
        is run numerous times
    """
    # a provides the location on the array as it runs, essentially loops over the cells
    edge_index = cuda.grid(1)

    # checks to see that position is in the array, double-check as GPUs can be weird sometimes
    if edge_index < jkr_edges.shape[0]:

        # get the indices of the cells in the edge
        index_1 = jkr_edges[edge_index][0]
        index_2 = jkr_edges[edge_index][1]

        # get the locations of the two cells
        location_1 = locations[index_1]
        location_2 = locations[index_2]

        # get the displacement vector between the two cells
        vector_x = location_1[0] - location_2[0]
        vector_y = location_1[1] - location_2[1]
        vector_z = location_1[2] - location_2[2]

        # get the magnitude of the displacement
        mag = magnitude(location_1, location_2)

        # make sure the magnitude is not 0
        if mag != 0:
            normal_x = vector_x / mag
            normal_y = vector_y / mag
            normal_z = vector_z / mag

        # if so return the zero vector
        else:
            normal_x, normal_y, normal_z = 0, 0, 0

        # get the total overlap of the cells used later in calculations
        overlap = radii[index_1] + radii[index_2] - mag

        # gets two values used for JKR
        e_hat = (((1 - poisson[0] ** 2) / youngs[0]) + ((1 - poisson[0] ** 2) / youngs[0])) ** -1
        r_hat = ((1 / radii[index_1]) + (1 / radii[index_2])) ** -1

        # used to calculate the max adhesive distance after bond has been already formed
        overlap_ = (((math.pi * adhesion_const[0]) / e_hat) ** (2 / 3)) * (r_hat ** (1 / 3))

        # get the nondimensionalized overlap, used for later calculations and checks
        # also for the use of a polynomial approximation of the force
        d = overlap / overlap_

        # check to see if the cells will have a force interaction
        if d > -0.360562:
            # plug the value of d into the nondimensionalized equation for the JKR force
            f = (-0.0204 * d ** 3) + (0.4942 * d ** 2) + (1.0801 * d) - 1.324

            # convert from the nondimensionalization to find the adhesive force
            jkr_force = f * math.pi * adhesion_const[0] * r_hat

            # adds the adhesive force as a vector in opposite directions to each cell's force holder
            forces[index_1][0] += jkr_force * normal_x
            forces[index_1][1] += jkr_force * normal_y
            forces[index_1][2] += jkr_force * normal_z

            forces[index_2][0] -= jkr_force * normal_x
            forces[index_2][1] -= jkr_force * normal_y
            forces[index_2][2] -= jkr_force * normal_z

        # remove the edge if the it fails to meet the criteria for distance, JKR simulating that
        # the bond is broken
        else:
            delete_jkr_edges[edge_index] = edge_index


@jit(nopython=True, parallel=True)
def apply_forces_cpu(number_cells, cell_jkr_force, cell_motility_force, cell_locations, cell_radii, viscosity, size,
                     move_time_step):
    """ This is the Numba optimized version of
        the apply_forces function that runs
        solely on the cpu.
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
            # check if new location is in environment space if not simulation a collision with the bounds
            if new_location[j] > size[j]:
                cell_locations[i][j] = size[j]
            elif new_location[j] < 0:
                cell_locations[i][j] = 0.0
            else:
                cell_locations[i][j] = new_location[j]

    # return the updated cell locations
    return cell_locations


@cuda.jit
def apply_forces_gpu(inactive_forces, active_forces, locations, radii, viscosity, size, move_time_step):
    """ This is the parallelized function for applying
        forces that is run numerous times.
    """
    # get the index in the array
    index = cuda.grid(1)

    # double check that this in still in the array
    if index < locations.shape[0]:
        # stokes law for velocity based on force and fluid viscosity
        stokes_friction = 6 * math.pi * viscosity[0] * radii[index]

        # update the velocity of the cell based on the solution
        velocity_x = (active_forces[index][0] + inactive_forces[index][0]) / stokes_friction
        velocity_y = (active_forces[index][1] + inactive_forces[index][1]) / stokes_friction
        velocity_z = (active_forces[index][2] + inactive_forces[index][2]) / stokes_friction

        # set the possible new location
        new_location_x = locations[index][0] + velocity_x * move_time_step[0]
        new_location_y = locations[index][1] + velocity_y * move_time_step[0]
        new_location_z = locations[index][2] + velocity_z * move_time_step[0]

        # the following check that the cell's new location is within the simulation space
        # for the x direction
        if new_location_x > size[0]:
            locations[index][0] = size[0]
        elif new_location_x < 0:
            locations[index][0] = 0.0
        else:
            locations[index][0] = new_location_x

        # for the y direction
        if new_location_y > size[1]:
            locations[index][1] = size[1]
        elif new_location_y < 0:
            locations[index][1] = 0.0
        else:
            locations[index][1] = new_location_y

        # for the z direction
        if new_location_z > size[2]:
            locations[index][2] = size[2]
        elif new_location_z < 0:
            locations[index][2] = 0.0
        else:
            locations[index][2] = new_location_z
