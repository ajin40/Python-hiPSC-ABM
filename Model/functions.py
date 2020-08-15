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
    # records the real time value of when a step begins
    simulation.step_start = time.time()

    # prints the current step number and the number of cells
    print("Step: " + str(simulation.current_step))
    print("Number of cells: " + str(simulation.number_cells))


def cell_update(simulation):
    """ loops over all indices of cells and updates
        them accordingly
    """
    # start time of the function
    simulation.cell_update_time = -1 * time.time()

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

    # calculate the total time elapsed for the function
    simulation.cell_update_time += time.time()


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
    # checks to see if the non-moving cell should divide or increase its division counter
    if not simulation.cell_motion[index]:
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
    half_index_x = simulation.cell_locations[index][0] // (simulation.dx / 2)
    half_index_y = simulation.cell_locations[index][1] // (simulation.dy / 2)
    half_index_z = simulation.cell_locations[index][2] // (simulation.dz / 2)
    index_x = math.ceil(half_index_x / 2)
    index_y = math.ceil(half_index_y / 2)
    index_z = math.ceil(half_index_z / 2)

    # if the diffusion point value is less than the max FGF4 it can hold and the cell is NANOG high
    # increase the FGF4 value by 1
    if simulation.cell_fds[index][3] == 1 and simulation.fgf4_values[index_x][index_y][index_z] < simulation.max_fgf4:
        simulation.fgf4_values_temp[index_x][index_y][index_z] += 1

    # activate the following pathway based on if dox (after 24 hours) has been induced yet
    if simulation.current_step >= 49:
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
            new_fgf4 = x5 % num_states
            new_fgfr = (x1 * x4) % num_states
            new_erk = x2 % num_states
            new_gata6 = (1 + x5 + x5 * x4) % num_states
            new_nanog = ((x3 + 1) * (x4 + 1)) % num_states

            # updates self.booleans with the new boolean values
            simulation.cell_fds[index] = np.array([new_fgfr, new_erk, new_gata6, new_nanog])

        # if no fds update, maintain the same fgf4 boolean value
        else:
            new_fgf4 = fgf4_bool

        # increase the finite dynamical system counter
        simulation.cell_fds_counter[index] += 1

        # if the temporary FGFR value is 0 and the FGF4 value is 1 decrease the amount of FGF4 by 1
        # this simulates FGFR using FGF4
        if temp_fgfr == 0 and new_fgf4 == 1:
            if simulation.fgf4_values[index_x][index_y][index_z] > 1:
                simulation.fgf4_values_temp[index_x][index_y][index_z] -= 1

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


def update_queue(simulation):
    """ add and removes cells to and from the simulation
        either all at once or in "groups"
    """
    # start time of the function
    simulation.update_queue_time = -1 * time.time()

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

    # calculate the total time elapsed for the function
    simulation.update_queue_time += time.time()


def check_neighbors(simulation):
    """ for all cells, determines which cells fall within a fixed
        radius to denote a neighbor then stores this information
        in a graph (uses a bin/bucket sorting method)
    """
    # start time of the function
    simulation.check_neighbors_time = -1 * time.time()

    # radius of search (meters) in which all cells within are classified as neighbors
    neighbor_distance = 0.000015

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

    # update the value of the max number of cells in a bin
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
        # based on the output of the function call
        max_neighbors = np.amax(edge_count)
        if check_neighbors.max_neighbors >= max_neighbors:
            break
        else:
            check_neighbors.max_neighbors = max_neighbors

    # reduce the edges to only nonzero edges
    edge_holder = edge_holder[if_nonzero]

    # add the edges to the neighbor graph
    simulation.neighbor_graph.add_edges(edge_holder)

    # calculate the total time elapsed for the function
    simulation.check_neighbors_time += time.time()


def handle_movement(simulation):
    """ runs the following functions together for the time period
        of the step. resets the motility force array to zero after
        movement is done
    """
    # start time of the function
    simulation.handle_movement_time = -1 * time.time()

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

    # calculate the total time elapsed for the function
    simulation.handle_movement_time += time.time()


def jkr_neighbors(simulation):
    """ for all cells, determines which cells will have physical
        interactions with other cells returns this information
        as an array of edges
    """
    # start time of the function
    simulation.jkr_neighbors_time = -1 * time.time()

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
        # based on the output of the function call
        max_neighbors = np.amax(edge_count)
        if jkr_neighbors.max_neighbors >= max_neighbors:
            break
        else:
            jkr_neighbors.max_neighbors = max_neighbors

    # reduce the edges to only nonzero edges
    edge_holder = edge_holder[if_nonzero]

    # add the edges and simplify the graph as this is a running graph that is never cleared due to its use
    # for holding adhesive JKR bonds from step to step
    simulation.jkr_graph.add_edges(edge_holder)
    simulation.jkr_graph.simplify()

    # calculate the total time elapsed for the function
    simulation.jkr_neighbors_time += time.time()


def get_forces(simulation):
    """ goes through all of "JKR" edges and quantifies any
        resulting adhesive or repulsion forces between
        pairs of cells
    """
    # start time of the function
    simulation.get_forces_time = -1 * time.time()

    # parameters that rarely change
    adhesion_const = 0.000107    # the adhesion constant in kg/s from P Pathmanathan et al.
    poisson = 0.5    # Poisson's ratio for the cells, 0.5 means incompressible
    youngs = 1000    # Young's modulus for the cells in kPa

    # get the edges as a numpy array, count them, and create an array used to delete edges
    jkr_edges = np.array(simulation.jkr_graph.get_edgelist())
    number_edges = len(jkr_edges)
    delete_edges = np.zeros(number_edges, dtype=int)

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
        delete_edges = delete_edges[delete_edges != 0]
        simulation.jkr_graph.delete_edges(delete_edges)
        simulation.cell_jkr_force = forces

    # calculate the total time elapsed for the function
    simulation.get_forces_time += time.time()


def apply_forces(simulation):
    """ Turns the active motility/division forces and
        inactive JKR forces into movement
    """
    # start time of the function
    simulation.apply_forces_time = -1 * time.time()

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

    # calculate the total time elapsed for the function
    simulation.apply_forces_time += time.time()


def nearest(simulation):
    """ looks at cells within a given radius a determines
        the closest cells of certain types
    """
    # start time of the function
    simulation.nearest_time = -1 * time.time()

    # radius of search for nearest cells
    nearest_distance = 0.000025

    # if a static variable has not been created to hold the maximum number of cells in a bin, create one
    if not hasattr(nearest, "max_cells"):
        # begin with a low number of cells that can be revalued if the max number of cells exceeds this value
        nearest.max_cells = 5

    # calls the function that generates an array of bins that generalize the cell locations in addition to a
    # creating a helper array that assists the search method in counting cells in a particular bin
    bins, bins_help, max_cells = backend.assign_bins(simulation, nearest_distance, nearest.max_cells)

    # update the value of the max number of cells in a bin
    nearest.max_cells = max_cells

    # call the nvidia gpu version
    if simulation.parallel:
        states = simulation.cells_states == "Differentiated"
        a[:, 4] == 1

        # turn the following into arrays that can be interpreted by the gpu
        locations_cuda = cuda.to_device(simulation.cell_locations)
        bins_cuda = cuda.to_device(bins)
        bins_help_cuda = cuda.to_device(bins_help)
        distance_cuda = cuda.to_device(nearest_distance)
        states_cuda = cuda.to_device(states)
        fds_cuda = cuda.to_device(simulation.cell_fds)
        nearest_gata6_cuda = cuda.to_device(simulation.cell_nearest_gata6)
        nearest_nanog_cuda = cuda.to_device(simulation.cell_nearest_nanog)
        nearest_diff_cuda = cuda.to_device(simulation.cell_nearest_diff)

        # allocate threads and blocks for gpu memory
        threads_per_block = 72
        blocks_per_grid = math.ceil(simulation.number_cells / threads_per_block)

        # call the cuda kernel with given parameters
        backend.nearest_gpu[blocks_per_grid, threads_per_block](locations_cuda, bins_cuda, bins_help_cuda,
                                                                distance_cuda, states_cuda, fds_cuda,
                                                                nearest_gata6_cuda, nearest_nanog_cuda,
                                                                nearest_diff_cuda)
        # return the arrays back from the gpu
        gata6 = nearest_gata6_cuda.copy_to_host()
        nanog = nearest_nanog_cuda.copy_to_host()
        diff = nearest_diff_cuda.copy_to_host()

    # call the cpu version
    else:
        gata6, nanog, diff = backend.nearest_cpu(simulation.number_cells, nearest_distance, bins, bins_help,
                                                 simulation.cell_locations, simulation.cell_nearest_gata6,
                                                 simulation.cell_nearest_nanog, simulation.cell_nearest_diff,
                                                 simulation.cell_states, simulation.cell_fds)

    # revalue the array holding the indices of nearest cells of given type
    simulation.cell_nearest_gata6 = gata6
    simulation.cell_nearest_nanog = nanog
    simulation.cell_nearest_diff = diff

    # calculate the total time elapsed for the function
    simulation.nearest_time += time.time()


def cell_motility(simulation):
    """ gives the cells a motive force depending on
        set rules for the cell types.
    """
    # start time of the function
    simulation.cell_motility_time = -1 * time.time()

    # this is the motility force of the cells
    motility_force = 0.000000005

    # loop over all of the cells
    for i in range(simulation.number_cells):
        # get the neighbors of the cell
        neighbors = simulation.neighbor_graph.neighbors(i)

        # check whether differentiated or pluripotent
        if simulation.cell_states[i] == "Differentiated":
            count = 0
            for index in neighbors:
                if simulation.cell_states[index] == "Differentiated":
                    count += 1

            if count >= simulation.diff_move_thresh:
                simulation.cell_motion[i] = False

            if simulation.cell_motion[i]:
                # create a vector to hold the sum of normal vectors between a cell and its neighbors
                vector_holder = np.array([0.0, 0.0, 0.0])

                # loop over the neighbors getting the normal and adding to the holder
                count = 0
                for j in range(len(neighbors)):
                    if simulation.cell_states[neighbors[j]] == "Pluripotent":
                        count += 1
                        vector = simulation.cell_locations[neighbors[j]] - simulation.cell_locations[i]
                        vector_holder += vector

                if count > 0:
                    # get the normal vector
                    normal = backend.normal_vector(vector_holder)

                    # move in direction opposite to pluripotent cells
                    simulation.cell_motility_force[i] += motility_force * normal * -1 * 1

                else:
                    simulation.cell_motility_force[i] += backend.random_vector(simulation) * motility_force

        # for pluripotent cells
        else:
            count = 0
            for index in neighbors:
                if simulation.cell_states[index] == "Pluripotent":
                    count += 1

            if count >= simulation.move_thresh:
                simulation.cell_motion[i] = False

            if simulation.cell_motion[i]:

                simulation.cell_motility_force[i] += backend.random_vector(simulation) * motility_force

            else:
                if not np.isnan(simulation.cell_cluster_nearest[i]):
                    pluri_index = int(simulation.cell_cluster_nearest[i])

                    vector = simulation.cell_locations[pluri_index] - simulation.cell_locations[i]
                    normal = backend.normal_vector(vector)

                    # calculate the motility force
                    simulation.cell_motility_force[i] += normal * motility_force * 0.05

            # # apply movement if the cell is "in motion"
            # if simulation.cell_motion[i]:
            #     # GATA6 high cell
            #     if simulation.cell_fds[i][2] == 1:
            #         # continue if using Guye et al. movement and if there exists differentiated cells
            #         if simulation.guye_move and not np.isnan(simulation.cell_nearest_diff[i]):
            #             # get the differentiated neighbors
            #             guye_neighbor = int(simulation.cell_nearest_diff[i])
            #
            #             # get the normal vector
            #             vector = simulation.cell_locations[guye_neighbor] - simulation.cell_locations[i]
            #             normal = backend.normal_vector(vector)
            #
            #             # calculate the motility force
            #             simulation.cell_motility_force[i] += normal * motility_force
            #
            #     # NANOG high cell
            #     elif simulation.cell_fds[i][3] == 1:
            #         # move based on fgf4 concentrations
            #         if simulation.fgf4_move:
            #             # makes sure not the numpy nan type, proceed if actual value
            #             if (np.isnan(simulation.cell_highest_fgf4[i]) == np.zeros(3, dtype=bool)).all():
            #                 # get the location of the diffusion point and move toward it
            #                 x = int(simulation.cell_highest_fgf4[i][0])
            #                 y = int(simulation.cell_highest_fgf4[i][1])
            #                 z = int(simulation.cell_highest_fgf4[i][2])
            #                 vector = simulation.cell_locations[i] - simulation.diffuse_locations[x][y][z]
            #                 normal = backend.normal_vector(vector)
            #                 simulation.cell_motility_force[i] += normal * motility_force
            #
            #         # move based on Eunbi's model
            #         elif simulation.eunbi_move:
            #             # if there is a gata6 high cell nearby, move away from it
            #             if not np.isnan(simulation.cell_nearest_gata6[i]):
            #                 nearest_index = int(simulation.cell_nearest_gata6[i])
            #                 vector = simulation.cell_locations[nearest_index] - simulation.cell_locations[i]
            #                 normal = backend.normal_vector(vector)
            #                 simulation.cell_motility_force[i] += normal * motility_force * -1
            #
            #             # if there is a nanog high cell nearby, move to it
            #             elif not np.isnan(simulation.cell_nearest_nanog[i]):
            #                 nearest_index = int(simulation.cell_nearest_nanog[i])
            #                 vector = simulation.cell_locations[nearest_index] - simulation.cell_locations[i]
            #                 normal = backend.normal_vector(vector)
            #                 simulation.cell_motility_force[i] += normal * motility_force
            #
            #             # if nothing else, move randomly
            #             else:
            #                 simulation.cell_motility_force[i] += backend.random_vector(simulation) * motility_force
            #         # if nothing else, move randomly
            #         else:
            #             simulation.cell_motility_force[i] += backend.random_vector(simulation) * motility_force
            #     # if nothing else, move randomly
            #     else:
            #         simulation.cell_motility_force[i] += backend.random_vector(simulation) * motility_force

    # calculate the total time elapsed for the function
    simulation.cell_motility_time += time.time()


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
    x_steps = simulation.fgf4_values.shape[0]
    y_steps = simulation.fgf4_values.shape[1]
    z_steps = simulation.fgf4_values.shape[2]

    # set up the locations of the diffusion points
    x, y, z = np.meshgrid(np.arange(x_steps), np.arange(y_steps), np.arange(z_steps), indexing='ij')
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
        diffuse_bins, diffuse_bins_help = backend.setup_diffuse_bins_cpu(simulation.diffuse_locations, x_steps, y_steps,
                                                                         z_steps, simulation.diffuse_radius,
                                                                         diffuse_bins, diffuse_bins_help)

        # either break the loop if all points were accounted for or revalue the maximum number of points based on
        # the output of the function call
        max_points = np.amax(diffuse_bins_help)
        if setup_diffusion_bins.max_points >= max_points:
            break
        else:
            setup_diffusion_bins.max_points = max_points

    # update the diffuse bins for the simulation instance
    simulation.diffuse_bins = diffuse_bins
    simulation.diffuse_bins_help = diffuse_bins_help


def update_diffusion(simulation):
    """ goes through all extracellular gradients and
        approximates the diffusion of that molecule
    """
    # start time of the function
    simulation.update_diffusion_time = -1 * time.time()

    # calculate the number of times the finite differences diffusion is run
    diffusion_steps = math.ceil(simulation.step_dt / simulation.dt)

    # go through all gradients and update the diffusion of each
    for gradient, temp in simulation.extracellular_names:
        # divide the temporary gradient by the number of steps to simulate the incremental increase in concentration
        simulation.__dict__[temp] /= diffusion_steps

        # get the dimensions of an array that is 2 bigger in all directions
        size = np.array(simulation.__dict__[gradient].shape) + 2 * np.ones(3, dtype=int)

        # create arrays that will give the gradient arrays a border of zeros
        gradient_base = np.zeros(size)
        temp_base = np.zeros(size)

        # add the gradient array and the temp array to the middle of the base arrays
        gradient_base[1:-1, 1:-1, 1:-1] = simulation.fgf4_values
        temp_base[1:-1, 1:-1, 1:-1] = simulation.fgf4_values_temp

        # return the gradient base after it has been updated by the finite differences method
        gradient_base = backend.update_diffusion_cpu(gradient_base, temp_base, diffusion_steps, simulation.dt,
                                                     simulation.dx2, simulation.dy2, simulation.dz2, simulation.diffuse,
                                                     simulation.size)
        # get the gradient
        simulation.__dict__[gradient] = gradient_base[1:-1, 1:-1, 1:-1]

    # calculate the total time elapsed for the function
    simulation.update_diffusion_time += time.time()


def highest_fgf4(simulation):
    """ Search for the highest concentration of
        fgf4 within a fixed radius
    """
    simulation.cell_highest_fgf4 = backend.highest_fgf4_cpu(simulation.diffuse_radius, simulation.diffuse_bins,
                                                            simulation.diffuse_bins_help, simulation.diffuse_locations,
                                                            simulation.cell_locations, simulation.number_cells,
                                                            simulation.cell_highest_fgf4, simulation.fgf4_values)


def outside_cluster(simulation):
    """ Find pluripotent cells outside the cluster
        that the cell is currently in
    """
    # create a copy of the neighbor graph and remove edges with differentiated cells
    pluri_graph = copy.deepcopy(simulation.neighbor_graph)
    edges = np.array(pluri_graph.get_edgelist())
    delete = np.zeros(len(edges), dtype=int)
    delete = backend.remove_diff_edges(simulation.cell_states, edges, delete)
    delete = delete[delete != 0]
    pluri_graph.delete_edges(delete)

    # get the membership to corresponding clusters
    members = np.array(pluri_graph.clusters().membership)

    # radius of search for the nearest pluripotent cell not in the same cluster
    nearest_distance = 0.0002

    # calls the function that generates an array of bins that generalize the cell locations in addition to a
    # helper array that assists the search method in counting cells in a particular bin
    bins, bins_help = backend.assign_bins(simulation, nearest_distance)

    if simulation.parallel:
        nearest_distance_cuda = cuda.to_device(nearest_distance)
        bins_cuda = cuda.to_device(bins)
        bins_help_cuda = cuda.to_device(bins_help)
        locations_cuda = cuda.to_device(simulation.cell_locations)

        a = copy.deepcopy(simulation.cell_states)
        cell_states = a == "Pluripotent"
        states_cuda = cuda.to_device(cell_states)
        cell_cluster_nearest_cuda = cuda.to_device(simulation.cell_cluster_nearest)
        members_cuda = cuda.to_device(members)

        # allocate threads and blocks for gpu memory
        threads_per_block = 72
        blocks_per_grid = math.ceil(simulation.number_cells / threads_per_block)

        backend.outside_cluster_gpu[blocks_per_grid, threads_per_block](nearest_distance_cuda, bins_cuda,
                                                                        bins_help_cuda, locations_cuda, states_cuda,
                                                                        cell_cluster_nearest_cuda, members_cuda)

        simulation.cell_cluster_nearest = cell_cluster_nearest_cuda.copy_to_host()

    else:
        nearest_outside = backend.outside_cluster_cpu(simulation.number_cells, nearest_distance, bins, bins_help,
                                                      simulation.cell_locations, simulation.cell_states,
                                                      simulation.cell_cluster_nearest, members)

        # revalue the array holding the indices of nearest pluripotent cells outside cluster
        simulation.cell_cluster_nearest = nearest_outside
