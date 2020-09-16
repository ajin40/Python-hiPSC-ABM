import numpy as np
import time
from numba import cuda
import math
import random as r
import copy

import backend


def info(simulation):
    """ gives an idea of how the simulation is running
        and records the beginning of the step in real time
    """
    # records when the step begins, used for measuring efficiency
    simulation.step_start = time.perf_counter()

    # prints the current step number and the number of cells
    print("Step: " + str(simulation.current_step))
    print("Number of cells: " + str(simulation.number_cells))


@backend.record_time
def cell_update(simulation):
    """ loops over all indices of cells and updates
        them accordingly
    """
    # loop over the cells
    for i in range(simulation.number_cells):
        # Cell death
        cell_death(simulation, i)

        # Differentiated surround
        cell_diff_surround(simulation, i)

        # Growth
        cell_growth(simulation, i)

        # Division
        cell_division(simulation, i)

        # Extracellular interaction and GATA6 pathway
        cell_pathway(simulation, i)


def cell_death(simulation, index):
    """ marks the cell for removal if it meets
        the criteria for cell death
    """
    # checks to see if cell is pluripotent
    if simulation.cell_states[index] == "Pluripotent":
        # looks at the neighbors and counts them, increasing the death counter if not enough neighbors
        neighbors = simulation.neighbor_graph.neighbors(index)
        if len(neighbors) < simulation.lonely_cell:
            simulation.cell_death_counter[index] += 1
        # if not reset the death counter back to zero
        else:
            simulation.cell_death_counter[index] = 0

        # removes cell if it meets the parameters
        if simulation.cell_death_counter[index] >= simulation.death_thresh:
            simulation.cells_to_remove = np.append(simulation.cells_to_remove, index)


def cell_diff_surround(simulation, index):
    """ simulates the phenomenon that differentiated cells
        induce the differentiation of a pluripotent cell
    """
    # checks to see if cell is pluripotent and GATA6 low
    if simulation.cell_states[index] == "Pluripotent" and simulation.cell_fds[index][2] == 0:
        # finds neighbors of a cell
        neighbors = simulation.neighbor_graph.neighbors(index)

        # holds the current number differentiated neighbors
        num_diff_neighbors = 0

        # loops over the neighbors of a cell
        for j in range(len(neighbors)):
            # checks to see if current neighbor is differentiated if so add it to the counter
            if simulation.cell_states[neighbors[j]] == "Differentiated":
                num_diff_neighbors += 1

            # if the number of differentiated meets the threshold, increase the counter and break the loop
            if num_diff_neighbors >= simulation.diff_surround:
                simulation.cell_diff_counter[index] += r.randint(0, 2)
                break


def cell_growth(simulation, index):
    """ simulates the growth of a cell leading up to its
        division, currently linear, radius growth
    """
    # increase the cell radius based on the state and whether or not it has reached the max size
    if simulation.cell_radii[index] < simulation.max_radius:
        # pluripotent growth
        if simulation.cell_states[index] == "Pluripotent":
            simulation.cell_radii[index] += simulation.pluri_growth

        # differentiated growth
        else:
            simulation.cell_radii[index] += simulation.diff_growth


def cell_division(simulation, index):
    """ either marks the cell for division or increases the
        counter to division given certain criteria
    """
    # # checks to see if the non-moving cell should divide or increase its division counter
    # if not simulation.cell_motion[index]:
    # if it's a differentiated cell
    if simulation.cell_states[index] == "Differentiated":
        # check the threshold
        if simulation.cell_div_counter[index] >= simulation.diff_div_thresh:
            # if under the threshold
            neighbors = simulation.neighbor_graph.neighbors(index)
            if len(neighbors) < simulation.contact_inhibit:
                simulation.cells_to_divide = np.append(simulation.cells_to_divide, index)

        else:
            # stochastically increase the division counter by either 0, 1, or 2
            simulation.cell_div_counter[index] += r.randint(0, 2)

    # no contact inhibition for pluripotent cells
    else:
        # check the threshold
        if simulation.cell_div_counter[index] >= simulation.pluri_div_thresh:
            simulation.cells_to_divide = np.append(simulation.cells_to_divide, index)

        else:
            # stochastically increase the division counter by either 0, 1, or 2
            simulation.cell_div_counter[index] += r.randint(0, 2)


def cell_pathway(simulation, index):
    """ simulates the gata6 pathway and extracellular
        interaction of the cell
    """
    # take the location of a cell and determine the nearest diffusion point by creating a zone around a
    # diffusion point an any cells in the zone will base their value off of that
    half_index_x = simulation.cell_locations[index][1] // (simulation.dx / 2)
    half_index_y = simulation.cell_locations[index][0] // (simulation.dy / 2)
    half_index_z = simulation.cell_locations[index][2] // (simulation.dz / 2)
    index_x = math.ceil(half_index_x / 2)
    index_y = math.ceil(half_index_y / 2)
    index_z = math.ceil(half_index_z / 2)

    # if the diffusion point value is less than the max FGF4 it can hold and the cell is NANOG high
    # increase the FGF4 value by 1
    if simulation.cell_fds[index][3] == 1 and simulation.fgf4_values[index_x][index_y][index_z] < simulation.max_fgf4:
        simulation.fgf4_values_temp[index_x][index_y][index_z] += 1

    # activate the following pathway based on if dox (after 24 hours) has been induced yet
    if simulation.current_step >= 49 and simulation.dox_value > simulation.cell_dox_value[index]:
        # if the FGF4 amount for the location is greater than 0, set the fgf4_bool value to be 1 for the
        # functions
        if simulation.fgf4_values[index_x][index_y][index_z] > 0:
            fgf4_bool = 1
        else:
            fgf4_bool = 0

        # Finite dynamical system and state change
        # temporarily hold the FGFR value
        temp_fgfr = simulation.cell_fds[index][0]

        # only update the booleans when the counter matches the boolean update rate
        if simulation.cell_fds_counter[index] % simulation.fds_thresh == 0:
            # number of states for the finite dynamical system
            num_states = 2

            # xn is equal to the value corresponding to its function
            x1 = fgf4_bool
            x2 = simulation.cell_fds[index][0]
            x3 = simulation.cell_fds[index][1]
            x4 = simulation.cell_fds[index][2]
            x5 = simulation.cell_fds[index][3]

            # evaluate the functions by turning them from strings to equations
            new_fgfr = (x1 * x4) % num_states
            new_erk = x2 % num_states
            new_gata6 = (1 + x5 + x5 * x4) % num_states
            new_nanog = ((x3 + 1) * (x4 + 1)) % num_states

            # updates self.booleans with the new boolean values
            simulation.cell_fds[index] = np.array([new_fgfr, new_erk, new_gata6, new_nanog])

            # if the temporary FGFR value is 0 and the new FGFR value is 1 decrease the amount of FGF4 by 1
            # this simulates FGFR using FGF4
            if temp_fgfr == 0 and new_fgfr == 1:
                if simulation.fgf4_values[index_x][index_y][index_z] > 1:
                    simulation.fgf4_values_temp[index_x][index_y][index_z] -= 1

        # increase the finite dynamical system counter
        simulation.cell_fds_counter[index] += 1

        # if the cell is GATA6 high and pluripotent increase the differentiation counter by 1
        if simulation.cell_fds[index][2] == 1 and simulation.cell_states[index] == "Pluripotent":
            simulation.cell_diff_counter[index] += r.randint(0, 2)

            # if the differentiation counter is greater than the threshold, differentiate
            if simulation.cell_diff_counter[index] >= simulation.pluri_to_diff:
                # change the state to differentiated
                simulation.cell_states[index] = "Differentiated"

                # make sure NANOG is low or rather 0
                simulation.cell_fds[index][3] = 0

                # allow the cell to actively move again
                simulation.cell_motion[index] = True


@backend.record_time
def cell_motility(simulation):
    """ gives the cells a motive force depending on
        set rules for the cell types.
    """
    # this is the motility force of the cells
    motility_force = 0.000000002

    # loop over all of the cells
    for i in range(simulation.number_cells):
        # get the neighbors of the cell
        neighbors = simulation.neighbor_graph.neighbors(i)

        # if the cell state is differentiated and moving
        if simulation.cell_states[i] == "Differentiated" and simulation.cell_motion[i]:
            # if not surrounded by more than 8 cells, move away from surrounding nanog high cells
            if len(neighbors) < 6:
                # create a vector to hold the sum of normal vectors between a cell and its neighbors
                vector_holder = np.array([0.0, 0.0, 0.0])

                # loop over the neighbors
                count = 0
                for j in range(len(neighbors)):
                    # if neighbor is nanog high, add vector to the cell to the holder
                    if simulation.cell_fds[neighbors[j]][3]:
                        count += 1
                        vector = simulation.cell_locations[neighbors[j]] - simulation.cell_locations[i]
                        vector_holder += vector

                # if there is at least one nanog high cell move away from it
                if count > 0:
                    # get the normal vector
                    normal = backend.normal_vector(vector_holder)

                    # move in direction opposite to nanog high cells
                    simulation.cell_motility_force[i] += motility_force * normal * -1

                # if no nanog high cells around, move randomly
                else:
                    simulation.cell_motility_force[i] += backend.random_vector(simulation) * motility_force

            # set the motion to be false if it is surrounded by more than 8 cells
            else:
                simulation.cell_motion[i] = False

        # if the cell is gata6 high and nanog low
        elif simulation.cell_fds[i][2] == 1 and not simulation.cell_fds[i][3] == 1:
            # if not surrounded by more than 8 cells, move to nearest differentiated cell
            if len(neighbors) < 6:
                # continue if using Guye et al. movement and if there exists differentiated cells
                if simulation.guye_move and not np.isnan(simulation.cell_nearest_diff[i]):
                    # get the differentiated neighbors
                    guye_neighbor = int(simulation.cell_nearest_diff[i])

                    # get the normal vector
                    vector = simulation.cell_locations[guye_neighbor] - simulation.cell_locations[i]
                    normal = backend.normal_vector(vector)

                    # calculate the motility force
                    simulation.cell_motility_force[i] += normal * motility_force

                # if no Guye movement or no differentiated cells nearby, move randomly
                else:
                    simulation.cell_motility_force[i] += backend.random_vector(simulation) * motility_force

            # set the motion to be false if it is surrounded by more than 8 cells
            else:
                simulation.cell_motion[i] = False

        # if the cell is nanog high and gata6 low
        elif simulation.cell_fds[i][3] == 1 and not simulation.cell_fds[i][2] == 1:
            # set the motion to be false if there are enough nanog high neighbors
            # if len(neighbors) < simulation.move_thresh:
            if len(neighbors) < 6:
                # move based on fgf4 concentrations
                if simulation.fgf4_move:
                    # makes sure not the numpy nan type, proceed if actual value
                    if (np.isnan(simulation.cell_highest_fgf4[i]) == np.zeros(3, dtype=bool)).all():
                        # get the location of the diffusion point and move toward it
                        x = int(simulation.cell_highest_fgf4[i][0])
                        y = int(simulation.cell_highest_fgf4[i][1])
                        z = int(simulation.cell_highest_fgf4[i][2])
                        vector = simulation.cell_locations[i] - simulation.diffuse_locations[x][y][z]
                        normal = backend.normal_vector(vector)
                        simulation.cell_motility_force[i] += normal * motility_force

                # move based on Eunbi's model
                elif simulation.eunbi_move:
                    # if there is a gata6 high cell nearby, move away from it
                    if not np.isnan(simulation.cell_nearest_gata6[i]):
                        nearest_index = int(simulation.cell_nearest_gata6[i])
                        vector = simulation.cell_locations[nearest_index] - simulation.cell_locations[i]
                        normal = backend.normal_vector(vector)
                        simulation.cell_motility_force[i] += normal * motility_force

                    # if there is a nanog high cell nearby, move to it
                    elif not np.isnan(simulation.cell_nearest_nanog[i]):
                        nearest_index = int(simulation.cell_nearest_nanog[i])
                        vector = simulation.cell_locations[nearest_index] - simulation.cell_locations[i]
                        normal = backend.normal_vector(vector)
                        simulation.cell_motility_force[i] += normal * motility_force

                    # if nothing else, move randomly
                    else:
                        simulation.cell_motility_force[i] += backend.random_vector(simulation) * motility_force

                # if no specific movement type, move randomly
                else:
                    simulation.cell_motility_force[i] += backend.random_vector(simulation) * motility_force

            else:
                simulation.cell_motion[i] = False

                # if not np.isnan(simulation.cell_cluster_nearest[i]):
                #     pluri_index = int(simulation.cell_cluster_nearest[i])
                #     vector = simulation.cell_locations[pluri_index] - simulation.cell_locations[i]
                #     normal = backend.normal_vector(vector)
                #     simulation.cell_motility_force[i] += normal * motility_force * 0.05

        # if both gata6/nanog high or both low
        else:
            # get general neighbors for inhibiting movement
            if len(neighbors) >= 6:
                simulation.cell_motion[i] = False

            # if actively moving, move randomly
            if simulation.cell_motion[i]:
                simulation.cell_motility_force[i] += backend.random_vector(simulation) * motility_force


@backend.record_time
def alt_cell_motility(simulation):
    """ gives the cells a motive force depending on
        set rules for the cell types expect these rules
        are more similar to NetLogo
    """
    # this is the motility force of the cells
    motility_force = 0.0000000001

    # loop over all of the cells
    for i in range(simulation.number_cells):
        # see if the cell is moving or not
        if simulation.cell_motion[i]:
            # get the neighbors of the cell if the cell is actively moving
            neighbors = simulation.neighbor_graph.neighbors(i)

            # if cell is surrounded by other cells, inhibit the motion
            if len(neighbors) >= 6:
                simulation.cell_motion[i] = False

            # if not, calculate the active movement for the step
            else:
                if simulation.cell_states[i] == "Differentiated":
                    # if there is a nanog high cell nearby, move away from it
                    if not np.isnan(simulation.cell_nearest_nanog[i]):
                        nearest_index = int(simulation.cell_nearest_nanog[i])
                        vector = simulation.cell_locations[nearest_index] - simulation.cell_locations[i]
                        normal = backend.normal_vector(vector)
                        simulation.cell_motility_force[i] += normal * motility_force * -1

                    # if no nearby nanog high cells, move randomly
                    else:
                        simulation.cell_motility_force[i] += backend.random_vector(simulation) * motility_force

                # if the cell is gata6 high and nanog low
                elif simulation.cell_fds[i][2] == 1 and not simulation.cell_fds[i][3] == 1:
                    # if there is a differentiated cell nearby, move toward it
                    if not np.isnan(simulation.cell_nearest_diff[i]):
                        nearest_index = int(simulation.cell_nearest_diff[i])
                        vector = simulation.cell_locations[nearest_index] - simulation.cell_locations[i]
                        normal = backend.normal_vector(vector)
                        simulation.cell_motility_force[i] += normal * motility_force

                    # if no nearby differentiated cells, move randomly
                    else:
                        simulation.cell_motility_force[i] += backend.random_vector(simulation) * motility_force
                    # simulation.cell_motility_force[i] += backend.random_vector(simulation) * motility_force

                # if the cell is nanog high and gata6 low
                elif simulation.cell_fds[i][3] == 1 and not simulation.cell_fds[i][2] == 1:
                    # if there is a nanog high cell nearby, move toward it
                    if not np.isnan(simulation.cell_nearest_nanog[i]):
                        nearest_index = int(simulation.cell_nearest_nanog[i])
                        vector = simulation.cell_locations[nearest_index] - simulation.cell_locations[i]
                        normal = backend.normal_vector(vector)
                        simulation.cell_motility_force[i] += normal * motility_force * 0.8
                        simulation.cell_motility_force[i] += backend.random_vector(simulation) * motility_force * 0.2

                    # if there is a gata6 high cell nearby, move away from it
                    elif not np.isnan(simulation.cell_nearest_gata6[i]):
                        nearest_index = int(simulation.cell_nearest_gata6[i])
                        vector = simulation.cell_locations[nearest_index] - simulation.cell_locations[i]
                        normal = backend.normal_vector(vector)
                        simulation.cell_motility_force[i] += normal * motility_force * -1

                    else:
                        simulation.cell_motility_force[i] += backend.random_vector(simulation) * motility_force

                # if both gata6/nanog high or both low, move randomly
                else:
                    simulation.cell_motility_force[i] += backend.random_vector(simulation) * motility_force


@backend.record_time
def update_queue(simulation):
    """ add and removes cells to and from the simulation
        either all at once or in "groups"
    """
    # give how many cells are being added/removed during a given step
    print("Adding " + str(len(simulation.cells_to_divide)) + " cells...")
    print("Removing " + str(len(simulation.cells_to_remove)) + " cells...")

    # loops over all indices that are set to divide
    for i in range(len(simulation.cells_to_divide)):
        # get the index and divide that cell
        index = simulation.cells_to_divide[i]
        backend.divide_cell(simulation, index)

        # Cannot add all of the new cells, otherwise several cells are likely to be added in
        #   close proximity to each other at later time steps. Such addition, coupled with
        #   handling collisions, make give rise to sudden changes in overall positions of
        #   cells within the simulation. Instead, collisions are handled after 'group' number
        #   of cells are added.

        # if group is equal to 0, all will be added in at once
        if simulation.group != 0:
            if (i + 1) % simulation.group == 0:
                # call the handle movement function, which should reduce the problems described above
                handle_movement(simulation)

    # loops over all indices that are set to be removed
    for i in range(len(simulation.cells_to_remove)):
        # get the index and remove it
        index = simulation.cells_to_remove[i]
        backend.remove_cell(simulation, index)

        # adjusts the indices as deleting part of the array may alter the other indices to remove
        for j in range(i + 1, len(simulation.cells_to_remove)):
            # if the current cell being deleted falls before the other cell, shift the indices by 1
            if index < simulation.cells_to_remove[j]:
                simulation.cells_to_remove[j] -= 1

        # if group is equal to 0, all will be removed at once
        if simulation.group != 0:
            if (i + 1) % simulation.group == 0:
                # call the handle movement function, which should reduce the problems described above
                handle_movement(simulation)

    # clear the arrays for the next step
    simulation.cells_to_divide = np.array([], dtype=int)
    simulation.cells_to_remove = np.array([], dtype=int)


@backend.record_time
def check_neighbors(simulation):
    """ for all cells, determines which cells fall within a fixed
        radius to denote a neighbor then stores this information
        in a graph (uses a bin/bucket sorting method)
    """
    # radius of search (meters) in which all cells within are classified as neighbors
    neighbor_distance = 0.00001

    # if a static variable has not been created to hold the maximum number of neighbors, create one
    if not hasattr(check_neighbors, "max_neighbors"):
        # begin with a low number of neighbors that can be revalued if the max number of neighbors exceeds this value
        check_neighbors.max_neighbors = 5

    # if a static variable has not been created to hold the maximum number of cells in a bin, create one
    if not hasattr(check_neighbors, "max_cells"):
        # begin with a low number of cells that can be revalued if the max number of cells exceeds this value
        check_neighbors.max_cells = 5

    # clear all of the edges in the neighbor graph
    simulation.neighbor_graph.delete_edges(None)

    # calls the function that generates an array of bins that generalize the cell locations in addition to a
    # creating a helper array that assists the search method in counting cells in a particular bin
    bins, bins_help, max_cells = backend.assign_bins(simulation, neighbor_distance, check_neighbors.max_cells)

    # update the value of the max number of cells in a bin and double it
    check_neighbors.max_cells = max_cells

    # this will run once and if all edges are included in edge_holder, the loop will break. if not, this will
    # run a second time with an updated value for number of predicted neighbors such that all edges are included
    while True:
        # create an array used to hold edges, an array to say if nonzero, and an array to count the edges per cell
        length = simulation.number_cells * check_neighbors.max_neighbors
        edge_holder = np.zeros((length, 2), dtype=int)
        if_nonzero = np.zeros(length, dtype=bool)
        edge_count = np.zeros(simulation.number_cells, dtype=int)

        # call the nvidia gpu version
        if simulation.parallel:
            # turn the following into arrays that can be interpreted by the gpu
            locations_cuda = cuda.to_device(simulation.cell_locations)
            bins_cuda = cuda.to_device(bins)
            bins_help_cuda = cuda.to_device(bins_help)
            distance_cuda = cuda.to_device(neighbor_distance)
            edge_holder_cuda = cuda.to_device(edge_holder)
            if_nonzero_cuda = cuda.to_device(if_nonzero)
            edge_count_cuda = cuda.to_device(edge_count)
            max_neighbors_cuda = cuda.to_device(check_neighbors.max_neighbors)

            # allocate threads and blocks for gpu memory
            threads_per_block = 72
            blocks_per_grid = math.ceil(simulation.number_cells / threads_per_block)

            # call the cuda kernel with given parameters
            backend.check_neighbors_gpu[blocks_per_grid, threads_per_block](locations_cuda, bins_cuda, bins_help_cuda,
                                                                            distance_cuda, edge_holder_cuda,
                                                                            if_nonzero_cuda, edge_count_cuda,
                                                                            max_neighbors_cuda)
            # return the arrays back from the gpu
            edge_holder = edge_holder_cuda.copy_to_host()
            if_nonzero = if_nonzero_cuda.copy_to_host()
            edge_count = edge_count_cuda.copy_to_host()

        # call the cpu version
        else:
            edge_holder, if_nonzero, edge_count = backend.check_neighbors_cpu(simulation.number_cells,
                                                                              simulation.cell_locations, bins,
                                                                              bins_help, neighbor_distance, edge_holder,
                                                                              if_nonzero, edge_count,
                                                                              check_neighbors.max_neighbors)

        # either break the loop if all neighbors were accounted for or revalue the maximum number of neighbors
        # based on the output of the function call and double it
        max_neighbors = np.amax(edge_count)
        if check_neighbors.max_neighbors >= max_neighbors:
            break
        else:
            check_neighbors.max_neighbors = max_neighbors * 2

    # reduce the edges to only nonzero edges
    edge_holder = edge_holder[if_nonzero]

    # add the edges to the neighbor graph
    simulation.neighbor_graph.add_edges(edge_holder)


@backend.record_time
def handle_movement(simulation):
    """ runs the following functions together for the time period
        of the step. resets the motility force array to zero after
        movement is done
    """
    # if a static variable for holding step time hasn't been created, create one
    if not hasattr(handle_movement, "steps"):
        # get the total amount of times the cells will be incrementally moved during the step
        handle_movement.steps = math.ceil(simulation.step_dt / simulation.move_dt)

    # run the following movement functions consecutively
    for i in range(handle_movement.steps):
        # determines which cells will have physical interactions and save this to a graph
        jkr_neighbors(simulation)

        # go through the edges found in the above function and calculate resulting JKR forces
        get_forces(simulation)

        # apply all forces such as motility and JKR to the cells
        apply_forces(simulation)

    # reset motility forces back to zero vectors
    simulation.cell_motility_force = np.zeros((simulation.number_cells, 3), dtype=float)


@backend.record_time
def jkr_neighbors(simulation):
    """ for all cells, determines which cells will have physical
        interactions with other cells returns this information
        as an array of edges
    """
    # radius of search (meters) in which neighbors will have physical interactions, double the max cell radius
    jkr_distance = 2 * simulation.max_radius

    # if a static variable has not been created to hold the maximum number of neighbors, create one
    if not hasattr(jkr_neighbors, "max_neighbors"):
        # begin with a low number of neighbors that can be revalued if the max number of neighbors exceeds this value
        jkr_neighbors.max_neighbors = 5

    # if a static variable has not been created to hold the maximum number of cells in a bin, create one
    if not hasattr(jkr_neighbors, "max_cells"):
        # begin with a low number of cells that can be revalued if the max number of cells exceeds this value
        jkr_neighbors.max_cells = 5

    # calls the function that generates an array of bins that generalize the cell locations in addition to a
    # helper array that assists the search method in counting cells in a particular bin
    bins, bins_help, max_cells = backend.assign_bins(simulation, jkr_distance, jkr_neighbors.max_cells)

    # update the value of the max number of cells in a bin
    jkr_neighbors.max_cells = max_cells

    # this will run once and if all edges are included in edge_holder, the loop will break. if not this will
    # run a second time with an updated value for number of predicted neighbors such that all edges are included
    while True:
        # create an array used to hold edges, an array to say where edges are, and an array to count the edges per cell
        length = simulation.number_cells * jkr_neighbors.max_neighbors
        edge_holder = np.zeros((length, 2), dtype=int)
        if_nonzero = np.zeros(length, dtype=bool)
        edge_count = np.zeros(simulation.number_cells, dtype=int)

        # call the nvidia gpu version
        if simulation.parallel:
            # turn the following into arrays that can be interpreted by the gpu
            locations_cuda = cuda.to_device(simulation.cell_locations)
            radii_cuda = cuda.to_device(simulation.cell_radii)
            bins_cuda = cuda.to_device(bins)
            bins_help_cuda = cuda.to_device(bins_help)
            jkr_distance_cuda = cuda.to_device(jkr_distance)
            edge_holder_cuda = cuda.to_device(edge_holder)
            if_nonzero_cuda = cuda.to_device(if_nonzero)
            edge_count_cuda = cuda.to_device(edge_count)
            max_neighbors_cuda = cuda.to_device(jkr_neighbors.max_neighbors)

            # allocate threads and blocks for gpu memory
            threads_per_block = 72
            blocks_per_grid = math.ceil(simulation.number_cells / threads_per_block)

            # call the cuda kernel with given parameters
            backend.jkr_neighbors_gpu[blocks_per_grid, threads_per_block](locations_cuda, radii_cuda, bins_cuda,
                                                                          bins_help_cuda, jkr_distance_cuda,
                                                                          edge_holder_cuda, if_nonzero_cuda,
                                                                          edge_count_cuda, max_neighbors_cuda)
            # return the arrays back from the gpu
            edge_holder = edge_holder_cuda.copy_to_host()
            if_nonzero = if_nonzero_cuda.copy_to_host()
            edge_count = edge_count_cuda.copy_to_host()

        # call the cpu version
        else:
            edge_holder, if_nonzero, edge_count = backend.jkr_neighbors_cpu(simulation.number_cells,
                                                                            simulation.cell_locations,
                                                                            simulation.cell_radii, bins, bins_help,
                                                                            jkr_distance, edge_holder, if_nonzero,
                                                                            edge_count, jkr_neighbors.max_neighbors)

        # either break the loop if all neighbors were accounted for or revalue the maximum number of neighbors
        # based on the output of the function call and double it
        max_neighbors = np.amax(edge_count)
        if jkr_neighbors.max_neighbors >= max_neighbors:
            break
        else:
            jkr_neighbors.max_neighbors = max_neighbors * 2

    # reduce the edges to only nonzero edges
    edge_holder = edge_holder[if_nonzero]

    # add the edges and simplify the graph as this is a running graph that is never cleared due to its use
    # for holding adhesive JKR bonds from step to step
    simulation.jkr_graph.add_edges(edge_holder)
    simulation.jkr_graph.simplify()


@backend.record_time
def get_forces(simulation):
    """ goes through all of "JKR" edges and quantifies any
        resulting adhesive or repulsion forces between
        pairs of cells
    """
    # parameters that rarely change
    adhesion_const = 0.000107    # the adhesion constant in kg/s from P Pathmanathan et al.
    poisson = 0.5    # Poisson's ratio for the cells, 0.5 means incompressible
    youngs = 1000    # Young's modulus for the cells in kPa

    # get the edges as a numpy array, count them, and create an array used to delete edges
    jkr_edges = np.array(simulation.jkr_graph.get_edgelist())
    number_edges = len(jkr_edges)
    delete_edges = np.zeros(number_edges, dtype=bool)

    # only continue if edges exist, if no edges compiled functions will raise errors
    if number_edges > 0:
        # call the nvidia gpu version
        if simulation.parallel:
            # turn the following into arrays that can be interpreted by the gpu
            jkr_edges_cuda = cuda.to_device(jkr_edges)
            delete_edges_cuda = cuda.to_device(delete_edges)
            locations_cuda = cuda.to_device(simulation.cell_locations)
            radii_cuda = cuda.to_device(simulation.cell_radii)
            forces_cuda = cuda.to_device(simulation.cell_jkr_force)
            poisson_cuda = cuda.to_device(poisson)
            youngs_cuda = cuda.to_device(youngs)
            adhesion_const_cuda = cuda.to_device(adhesion_const)

            # allocate threads and blocks for gpu memory
            threads_per_block = 72
            blocks_per_grid = math.ceil(number_edges / threads_per_block)

            # call the cuda kernel with given parameters
            backend.get_forces_gpu[blocks_per_grid, threads_per_block](jkr_edges_cuda, delete_edges_cuda,
                                                                       locations_cuda, radii_cuda, forces_cuda,
                                                                       poisson_cuda, youngs_cuda, adhesion_const_cuda)
            # return the new forces and the edges to be deleted
            forces = forces_cuda.copy_to_host()
            delete_edges = delete_edges_cuda.copy_to_host()

        # call the cpu version
        else:
            forces, delete_edges = backend.get_forces_cpu(jkr_edges, delete_edges, simulation.cell_locations,
                                                          simulation.cell_radii, simulation.cell_jkr_force, poisson,
                                                          youngs, adhesion_const)

        # update the jkr edges to remove any edges that have be broken and update the cell jkr forces
        delete_edges_indices = np.arange(number_edges)[delete_edges]
        simulation.jkr_graph.delete_edges(delete_edges_indices)
        simulation.cell_jkr_force = forces


@backend.record_time
def apply_forces(simulation):
    """ Turns the active motility/division forces and
        inactive JKR forces into movement
    """
    # parameters that rarely change
    viscosity = 10000    # the viscosity of the medium in Ns/m used for stokes friction

    # call the nvidia gpu version
    if simulation.parallel:
        # turn the following into arrays that can be interpreted by the gpu
        jkr_forces_cuda = cuda.to_device(simulation.cell_jkr_force)
        motility_forces_cuda = cuda.to_device(simulation.cell_motility_force)
        locations_cuda = cuda.to_device(simulation.cell_locations)
        radii_cuda = cuda.to_device(simulation.cell_radii)
        viscosity_cuda = cuda.to_device(viscosity)
        size_cuda = cuda.to_device(simulation.size)
        move_dt_cuda = cuda.to_device(simulation.move_dt)

        # allocate threads and blocks for gpu memory
        threads_per_block = 72
        blocks_per_grid = math.ceil(simulation.number_cells / threads_per_block)

        # call the cuda kernel with given parameters
        backend.apply_forces_gpu[blocks_per_grid, threads_per_block](jkr_forces_cuda, motility_forces_cuda,
                                                                     locations_cuda, radii_cuda, viscosity_cuda,
                                                                     size_cuda, move_dt_cuda)
        # return the new cell locations from the gpu
        new_locations = locations_cuda.copy_to_host()

    # call the cpu version
    else:
        new_locations = backend.apply_forces_cpu(simulation.number_cells, simulation.cell_jkr_force,
                                                 simulation.cell_motility_force, simulation.cell_locations,
                                                 simulation.cell_radii, viscosity, simulation.size,
                                                 simulation.move_dt)

    # update the locations and reset the jkr forces back to zero
    simulation.cell_locations = new_locations
    simulation.cell_jkr_force = np.zeros((simulation.number_cells, 3), dtype=float)


@backend.record_time
def nearest(simulation):
    """ looks at cells within a given radius a determines
        the closest cells of certain types
    """
    # radius of search for nearest cells
    nearest_distance = 0.000015

    # if a static variable has not been created to hold the maximum number of cells in a bin, create one
    if not hasattr(nearest, "max_cells"):
        # begin with a low number of cells that can be revalued if the max number of cells exceeds this value
        nearest.max_cells = 5

    # calls the function that generates an array of bins that generalize the cell locations in addition to a
    # creating a helper array that assists the search method in counting cells in a particular bin
    bins, bins_help, max_cells = backend.assign_bins(simulation, nearest_distance, nearest.max_cells)

    # update the value of the max number of cells in a bin and double it
    nearest.max_cells = max_cells

    # turn the following arrays into True/False
    if_diff = simulation.cell_states == "Differentiated"
    gata6_high = simulation.cell_fds[:, 2] == 1
    nanog_high = simulation.cell_fds[:, 3] == 1

    # call the nvidia gpu version
    if simulation.parallel:
        # turn the following into arrays that can be interpreted by the gpu
        locations_cuda = cuda.to_device(simulation.cell_locations)
        bins_cuda = cuda.to_device(bins)
        bins_help_cuda = cuda.to_device(bins_help)
        distance_cuda = cuda.to_device(nearest_distance)
        if_diff_cuda = cuda.to_device(if_diff)
        gata6_high_cuda = cuda.to_device(gata6_high)
        nanog_high_cuda = cuda.to_device(nanog_high)
        nearest_gata6_cuda = cuda.to_device(simulation.cell_nearest_gata6)
        nearest_nanog_cuda = cuda.to_device(simulation.cell_nearest_nanog)
        nearest_diff_cuda = cuda.to_device(simulation.cell_nearest_diff)

        # allocate threads and blocks for gpu memory
        threads_per_block = 72
        blocks_per_grid = math.ceil(simulation.number_cells / threads_per_block)

        # call the cuda kernel with given parameters
        backend.nearest_gpu[blocks_per_grid, threads_per_block](locations_cuda, bins_cuda, bins_help_cuda,
                                                                distance_cuda, if_diff_cuda, gata6_high_cuda,
                                                                nanog_high_cuda, nearest_gata6_cuda, nearest_nanog_cuda,
                                                                nearest_diff_cuda)
        # return the arrays back from the gpu
        gata6 = nearest_gata6_cuda.copy_to_host()
        nanog = nearest_nanog_cuda.copy_to_host()
        diff = nearest_diff_cuda.copy_to_host()

    # call the cpu version
    else:
        gata6, nanog, diff = backend.nearest_cpu(simulation.number_cells, simulation.cell_locations, bins, bins_help,
                                                 nearest_distance, if_diff, gata6_high, nanog_high,
                                                 simulation.cell_nearest_gata6, simulation.cell_nearest_nanog,
                                                 simulation.cell_nearest_diff)

    # revalue the array holding the indices of nearest cells of given type
    simulation.cell_nearest_gata6 = gata6
    simulation.cell_nearest_nanog = nanog
    simulation.cell_nearest_diff = diff


@backend.record_time
def nearest_cluster(simulation):
    """ find the nearest nanog high cells outside the cluster
        that the cell is currently in
    """
    # radius of search for the nearest pluripotent cell not in the same cluster
    nearest_distance = 0.0002

    # create a copy of the neighbor graph and get the edges
    nanog_graph = copy.deepcopy(simulation.neighbor_graph)
    edges = np.array(nanog_graph.get_edgelist())

    # create an array to hold edges to delete
    length = len(edges)
    delete = np.zeros(length, dtype=int)
    delete_help = np.zeros(length, dtype=bool)

    # use parallel jit function to find differentiated/gata6 nodes/edges
    if length != 0:
        delete, delete_help = backend.remove_gata6_edges(length, simulation.cell_fds, edges, delete, delete_help)

    # only delete edges meant to be deleted by the help array
    delete = delete[delete_help]
    nanog_graph.delete_edges(delete)

    # get the membership to corresponding clusters
    members = np.array(nanog_graph.clusters().membership)

    # if a static variable has not been created to hold the maximum number of cells in a bin, create one
    if not hasattr(nearest_cluster, "max_cells"):
        # begin with a low number of cells that can be revalued if the max number of cells exceeds this value
        nearest_cluster.max_cells = 5

    # calls the function that generates an array of bins that generalize the cell locations in addition to a
    # creating a helper array that assists the search method in counting cells in a particular bin
    bins, bins_help, max_cells = backend.assign_bins(simulation, nearest_distance, nearest_cluster.max_cells)

    # update the value of the max number of cells in a bin and double it
    nearest_cluster.max_cells = max_cells

    # turn the following array into True/False
    if_nanog = simulation.cell_fds[:, 3] == 1

    # call the nvidia gpu version
    if simulation.parallel:
        # turn the following into arrays that can be interpreted by the gpu
        distance_cuda = cuda.to_device(nearest_distance)
        bins_cuda = cuda.to_device(bins)
        bins_help_cuda = cuda.to_device(bins_help)
        locations_cuda = cuda.to_device(simulation.cell_locations)
        if_nanog_cuda = cuda.to_device(if_nanog)
        cell_nearest_cluster_cuda = cuda.to_device(simulation.cell_nearest_cluster)
        members_cuda = cuda.to_device(members)

        # allocate threads and blocks for gpu memory
        threads_per_block = 72
        blocks_per_grid = math.ceil(simulation.number_cells / threads_per_block)

        # call the cuda kernel with given parameters
        backend.nearest_cluster_gpu[blocks_per_grid, threads_per_block](locations_cuda, bins_cuda, bins_help_cuda,
                                                                        distance_cuda, if_nanog_cuda,
                                                                        cell_nearest_cluster_cuda, members_cuda)

        # return the array back from the gpu
        nearest_cell = cell_nearest_cluster_cuda.copy_to_host()

    # call the cpu version
    else:
        nearest_cell = backend.nearest_cluster_cpu(simulation.number_cells, simulation.cell_locations, bins, bins_help,
                                                   nearest_distance, if_nanog, simulation.cell_nearest_cluster, members)

    # revalue the array holding the indices of nearest nanog high cells outside cluster
    simulation.cell_cluster_nearest = nearest_cell


@backend.record_time
def highest_fgf4(simulation):
    """ Search for the highest concentration of
        fgf4 within a fixed radius
    """
    # call the nvidia gpu version
    if simulation.parallel:
        # make sure the gradient array is contiguous
        fgf4_values = np.ascontiguousarray(simulation.fgf4_values)

        # turn the following into arrays that can be interpreted by the gpu
        locations_cuda = cuda.to_device(simulation.cell_locations)
        diffuse_bins_cuda = cuda.to_device(simulation.diffuse_bins)
        diffuse_bins_help_cuda = cuda.to_device(simulation.diffuse_bins_help)
        diffuse_locations_cuda = cuda.to_device(simulation.diffuse_locations)
        distance_cuda = cuda.to_device(simulation.diffuse_radius)
        highest_fgf4_cuda = cuda.to_device(simulation.cell_highest_fgf4)
        fgf4_values_cuda = cuda.to_device(fgf4_values)

        # allocate threads and blocks for gpu memory
        threads_per_block = 72
        blocks_per_grid = math.ceil(simulation.number_cells / threads_per_block)

        # call the cuda kernel with given parameters
        backend.highest_fgf4_gpu[blocks_per_grid, threads_per_block](locations_cuda, diffuse_bins_cuda,
                                                                     diffuse_bins_help_cuda, diffuse_locations_cuda,
                                                                     distance_cuda, highest_fgf4_cuda, fgf4_values_cuda)
        # return the array back from the gpu
        cell_highest_fgf4 = highest_fgf4_cuda.copy_to_host()

    # call the cpu version
    else:
        cell_highest_fgf4 = backend.highest_fgf4_cpu(simulation.number_cells, simulation.cell_locations,
                                                     simulation.diffuse_bins, simulation.diffuse_bins_help,
                                                     simulation.diffuse_locations, simulation.diffuse_radius,
                                                     simulation.cell_highest_fgf4, simulation.fgf4_values)

    # revalue the array holding the indices of diffusion points of highest fgf4
    simulation.cell_highest_fgf4 = cell_highest_fgf4


def setup_diffusion_bins(simulation):
    """ This function will put the diffusion points
        into corresponding bins that will be used to
        find values of diffusion within a radius
    """
    # if a static variable has not been created to hold the maximum diffusion points in a bin, create one
    if not hasattr(setup_diffusion_bins, "max_points"):
        # begin with a low number of points that can be revalued if the max number of points exceeds this value
        setup_diffusion_bins.max_points = 10

    # get the dimensions of the array representing the diffusion points
    shape = simulation.fgf4_values.shape

    # set up the locations of the diffusion points
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    x, y, z = x * simulation.dx, y * simulation.dy, z * simulation.dz
    simulation.diffuse_locations = np.stack((x, y, z), axis=3)

    # if there is enough space for all points that should be in a bin, break out of the loop. if there isn't
    # enough space update the amount of needed space and re-put the points in bins. this will run once if the prediction
    # of points is correct, twice if it isn't the first time
    while True:
        # calculate the size of the array used to represent the bins and the bins helper array, include extra bins
        # for points that may fall outside of the space
        bins_help_size = np.ceil(simulation.size / simulation.diffuse_radius).astype(int) + np.array([5, 5, 5])
        bins_size = np.append(bins_help_size, [setup_diffusion_bins.max_points, 3])

        # create the arrays for "bins" and "bins_help"
        diffuse_bins = np.empty(bins_size, dtype=int)
        diffuse_bins_help = np.zeros(bins_help_size, dtype=int)

        # assign the points to bins via the jit function
        diffuse_bins, diffuse_bins_help = backend.setup_diffuse_bins_cpu(simulation.diffuse_locations, shape,
                                                                         simulation.diffuse_radius, diffuse_bins,
                                                                         diffuse_bins_help)

        # either break the loop if all points were accounted for or revalue the maximum number of points based on
        # the output of the function call. this will only be changed at most once
        max_points = np.amax(diffuse_bins_help)
        if setup_diffusion_bins.max_points >= max_points:
            break
        else:
            setup_diffusion_bins.max_points = max_points

    # update the diffuse bins for the simulation instance
    simulation.diffuse_bins = diffuse_bins
    simulation.diffuse_bins_help = diffuse_bins_help


@backend.record_time
def update_diffusion(simulation):
    """ goes through all extracellular gradients and
        approximates the diffusion of that molecule
    """
    # if a static variable for holding step time hasn't been created, create one
    if not hasattr(update_diffusion, "steps"):
        # get the total amount of times the cells will be incrementally moved during the step
        update_diffusion.steps = math.ceil(simulation.step_dt / simulation.dt)

    # go through all gradients and update the diffusion of each
    for gradient, temp in simulation.extracellular_names:
        # divide the temporary gradient by the number of steps to simulate the incremental increase in concentration
        simulation.__dict__[temp] /= update_diffusion.steps

        # get the dimensions of an array that is 2 bigger along all axes
        size = np.array(simulation.__dict__[gradient].shape) + 2 * np.ones(3, dtype=int)

        # create arrays that will give the gradient arrays a border of zeros
        # gradient_base = np.ones(size) * np.mean(simulation.__dict__[gradient])
        gradient_base = np.zeros(size)
        gradient_base[1:-1, 0, 1] = simulation.__dict__[gradient][:, 0, 0]
        gradient_base[1:-1, -1, 1] = simulation.__dict__[gradient][:, -1, 0]
        gradient_base[0, 1:-1, 1] = simulation.__dict__[gradient][0, :, 0]
        gradient_base[-1, 1:-1, 1] = simulation.__dict__[gradient][-1, :, 0]
        temp_base = np.zeros(size)

        # add the gradient array and the temp array to the middle of the base arrays so create border of zeros
        gradient_base[1:-1, 1:-1, 1:-1] = simulation.fgf4_values
        temp_base[1:-1, 1:-1, 1:-1] = simulation.fgf4_values_temp

        # return the gradient base after it has been updated by the finite differences method
        gradient_base = backend.update_diffusion_cpu(gradient_base, temp_base, update_diffusion.steps, simulation.dt,
                                                     simulation.dx2, simulation.dy2, simulation.dz2, simulation.diffuse,
                                                     simulation.size)
        # get the gradient
        simulation.__dict__[gradient] = gradient_base[1:-1, 1:-1, 1:-1]
        simulation.__dict__[temp][:, :, :] = 0
