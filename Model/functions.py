import numpy as np
from numba import cuda
import math
import random as r

import backend


@backend.record_time
def cell_death(simulation):
    """ Marks the cell for removal if it meets
        the criteria for cell death.
    """
    for index in range(simulation.number_cells):
        # checks to see if cell is pluripotent
        if simulation.states[index] == "Pluripotent":

            # gets the number of neighbors for a cell, increasing the death counter if not enough neighbors
            if len(simulation.neighbor_graph.neighbors(index)) < simulation.lonely_thresh:
                simulation.death_counters[index] += 1

            # if not, reset the death counter back to zero
            else:
                simulation.death_counters[index] = 0

            # add cell to removal array if it meets the parameters
            if simulation.death_counters[index] >= simulation.death_thresh:
                simulation.cells_to_remove = np.append(simulation.cells_to_remove, index)


@backend.record_time
def cell_diff_surround(simulation):
    """ Simulates differentiated cells inducing the
        differentiation of a pluripotent cell.
    """
    for index in range(simulation.number_cells):
        # checks to see if cell is pluripotent and GATA6 low/medium
        if simulation.states[index] == "Pluripotent" and simulation.GATA6[index] < simulation.NANOG[index]:

            # get the list of neighbors for the cell
            neighbors = simulation.neighbor_graph.neighbors(index)

            # loop over neighbors, counting the ones that are differentiated
            diff_neighbors = 0
            for neighbor_index in neighbors:
                # checks to see if current neighbor is differentiated and if so add to the counter
                if simulation.states[neighbor_index] == "Differentiated":
                    diff_neighbors += 1

                # if the number of differentiated meets the threshold, set the cell as gata6 high and nanog low
                if diff_neighbors >= 6:
                    simulation.GATA6[index] = simulation.field - 1
                    simulation.NANOG[index] = 0
                    break


@backend.record_time
def cell_division(simulation):
    """ Increases the cell division counter and if the
        cell meets criteria mark it for division.
    """
    for index in range(simulation.number_cells):
        # stochastically increase the division counter by either 0 or 1
        simulation.div_counters[index] += r.randint(0, 1)

        # pluripotent cell
        if simulation.states[index] == "Pluripotent":
            # check the division counter against the threshold, add to array if dividing
            if simulation.div_counters[index] >= simulation.pluri_div_thresh:
                simulation.cells_to_divide = np.append(simulation.cells_to_divide, index)

        # differentiated cell
        else:
            # check the division counter against the threshold, add to array if dividing
            if simulation.div_counters[index] >= simulation.diff_div_thresh:

                # check for contact inhibition since differentiated
                if len(simulation.neighbor_graph.neighbors(index)) < 6:
                    simulation.cells_to_divide = np.append(simulation.cells_to_divide, index)


@backend.record_time
def cell_growth(simulation):
    """ Simulates the growth of a cell currently linear,
        radius-based growth.
    """
    for index in range(simulation.number_cells):
        # increase the cell radius based on the state and whether or not it has reached the max size
        if simulation.radii[index] < simulation.max_radius:
            # pluripotent growth
            if simulation.states[index] == "Pluripotent":
                radius = simulation.pluri_growth * simulation.div_counters[index] + simulation.min_radius

            # differentiated growth
            else:
                radius = simulation.diff_growth * simulation.div_counters[index] + simulation.min_radius

            # update the radius for the index
            simulation.radii[index] = radius


@backend.record_time
def cell_pathway(simulation):
    """ Updates finite dynamical system variables and
        extracellular conditions.
    """
    for index in range(simulation.number_cells):
        # add FGF4 to the gradient based on the cell's value of NANOG
        if simulation.NANOG[index] > 0:
            # get the amount
            amount = simulation.NANOG[index]

            # add it to the normal FGF4 gradient and the alternative FGF4 gradient
            backend.adjust_morphogens(simulation, "fgf4_values", index, amount, "nearest")
            # backend.adjust_morphogens(simulation, "fgf4_alt", index, amount, "distance")

        # activate the following pathway based on if doxycycline  has been induced yet (after 24 hours/48 steps)
        if simulation.current_step >= simulation.dox_step:
            # get an FGF4 value for the FDS based on the concentration of FGF4
            fgf4_value = backend.get_concentration(simulation, "fgf4_values", index)

            # if FDS is boolean
            if simulation.field == 2:
                # base thresholds on the maximum concentrations
                if fgf4_value < 0.5 * simulation.max_concentration:
                    fgf4_fds = 0   # FGF4 low
                else:
                    fgf4_fds = 1    # FGF4 high

            # otherwise assume ternary for now
            else:
                # base thresholds on the maximum concentrations
                if fgf4_value < 1/3 * simulation.max_concentration:
                    fgf4_fds = 0    # FGF4 low
                elif fgf4_value < 2/3 * simulation.max_concentration:
                    fgf4_fds = 1    # FGF4 medium
                else:
                    fgf4_fds = 2    # FGF4 high

            # temporarily hold the FGFR value
            temp_fgfr = simulation.FGFR[index]

            # if updating the FDS values this step
            if simulation.fds_counters[index] % simulation.fds_thresh == 0:
                # get the current FDS values of the cell
                x1 = fgf4_fds
                x2 = simulation.FGFR[index]
                x3 = simulation.ERK[index]
                x4 = simulation.GATA6[index]
                x5 = simulation.NANOG[index]

                # if the FDS is boolean
                if simulation.field == 2:
                    # update boolean values based on FDS functions
                    new_fgfr = (x1 * x4) % 2
                    new_erk = x2 % 2
                    new_gata6 = (1 + x5 + x5 * x4) % 2
                    new_nanog = ((x3 + 1) * (x4 + 1)) % 2

                # otherwise assume ternary
                else:
                    # update ternary values based on FDS functions
                    new_fgfr = (x1 * x4 * ((2*x1 + 1) * (2*x4 + 1) + x1 * x4)) % 3
                    new_erk = x2 % 3
                    new_gata6 = ((x4**2) * (x5 + 1) + (x5**2) * (x4 + 1) + 2*x5 + 1) % 3
                    new_nanog = (x5**2 + x5 * (x5 + 1) * (x3 * (2*x4**2 + 2*x3 + 1) + x4*(2*x3**2 + 2*x4 + 1)) +
                                 (2*x3**2 + 1) * (2*x4**2 + 1)) % 3

                # if the amount of FGFR has increased, subtract that much FGF4 from the gradient
                fgfr_change = new_fgfr - temp_fgfr
                if fgfr_change > 0:
                    backend.adjust_morphogens(simulation, "fgf4_values", index, -1 * fgfr_change, "nearest")

                # update the FDS values of the cell
                simulation.FGFR[index] = new_fgfr
                simulation.ERK[index] = new_erk
                simulation.GATA6[index] = new_gata6
                simulation.NANOG[index] = new_nanog

            # increase the finite dynamical system counter
            simulation.fds_counters[index] += 1

            # if the cell is GATA6 high and pluripotent
            if simulation.GATA6[index] > simulation.NANOG[index] and simulation.states[index] == "Pluripotent":

                # increase the differentiation counter by 0 or 1
                simulation.diff_counters[index] += r.randint(0, 1)

                # if the differentiation counter is greater than or equal to the threshold, differentiate
                if simulation.diff_counters[index] >= simulation.pluri_to_diff:
                    # change the state to differentiated
                    simulation.states[index] = "Differentiated"

                    # make sure NANOG is low
                    simulation.NANOG[index] = 0

                    # allow the cell to actively move again
                    simulation.motion[index] = True


@backend.record_time
def cell_motility(simulation):
    """ Gives the cells a motive force depending on
        set rules for the cell types.
    """
    # this is the motility force of the cells
    motility_force = 0.000000002

    # loop over all of the cells
    for index in range(simulation.number_cells):
        # get the neighbors of the cell
        neighbors = simulation.neighbor_graph.neighbors(index)

        # if not surrounded 6 or more cells, calculate motility forces
        if len(neighbors) < 6:
            # if the cell state is differentiated
            if simulation.states[index] == "Differentiated":
                # create a vector to hold the sum of normal vectors between a cell and its neighbors
                vector_holder = np.array([0.0, 0.0, 0.0])

                # loop over the neighbors
                count = 0
                for i in range(len(neighbors)):
                    # if neighbor is nanog high, add vector to the cell to the holder
                    if simulation.NANOG[neighbors[i]] > simulation.GATA6[neighbors[i]]:
                        count += 1
                        vector = simulation.locations[neighbors[i]] - simulation.locations[index]
                        vector_holder += vector

                # if there is at least one nanog high cell move away from it
                if count > 0:
                    # get the normalized vector
                    normal = backend.normal_vector(vector_holder)

                    # move in direction opposite to nanog high cells
                    random = backend.random_vector(simulation)
                    simulation.motility_forces[index] += (normal * -0.8 + random * 0.2) * motility_force

                # if no nanog high cells around, move randomly
                else:
                    simulation.motility_forces[index] += backend.random_vector(simulation) * motility_force

            # if the cell is gata6 high and nanog low
            elif simulation.GATA6[index] > simulation.NANOG[index]:
                # continue if using Guye et al. movement and if there exists differentiated cells
                if simulation.guye_move and simulation.nearest_diff[index] != -1:
                    # get the differentiated neighbor
                    guye_neighbor = simulation.nearest_diff[index]

                    # get the normal vector
                    vector = simulation.locations[guye_neighbor] - simulation.locations[index]
                    normal = backend.normal_vector(vector)

                    # calculate the motility force
                    random = backend.random_vector(simulation)
                    simulation.motility_forces[index] += (normal * 0.8 + random * 0.2) * motility_force

                # if no Guye movement or no differentiated cells nearby, move randomly
                else:
                    simulation.motility_forces[index] += backend.random_vector(simulation) * motility_force

            # if the cell is nanog high and gata6 low
            elif simulation.NANOG[index] > simulation.GATA6[index]:
                # move randomly
                simulation.motility_forces[index] += backend.random_vector(simulation) * motility_force

            # if both gata6/nanog high or both low
            else:
                # move randomly
                simulation.motility_forces[index] += backend.random_vector(simulation) * motility_force


@backend.record_time
def eunbi_motility(simulation):
    """ Gives the cells a motive force depending on
        set rules for the cell types where these rules
        are closer to Eunbi's model.
    """
    # this is the motility force of the cells
    motility_force = 0.000000002

    # loop over all of the cells
    for index in range(simulation.number_cells):
        # see if the cell is moving or not
        if simulation.motion[index]:
            # get the neighbors of the cell if the cell is actively moving
            neighbors = simulation.neighbor_graph.neighbors(index)

            # if cell is not surrounded by 6 or more other cells, calculate motility forces
            if len(neighbors) < 6:
                # if differentiated
                if simulation.states[index] == "Differentiated":
                    # if there is a nanog high cell nearby, move away from it
                    if simulation.nearest_nanog[index] != -1:
                        nearest_index = simulation.nearest_nanog[index]
                        vector = simulation.locations[nearest_index] - simulation.cell_locations[index]
                        normal = backend.normal_vector(vector)
                        random = backend.random_vector(simulation)
                        simulation.motility_forces[index] += (normal * -0.8 + random * 0.2) * motility_force

                    # if no nearby nanog high cells, move randomly
                    else:
                        simulation.motility_forces[index] += backend.random_vector(simulation) * motility_force

                # if the cell is gata6 high and nanog low
                elif simulation.GATA6[index] > simulation.NANOG[index]:
                    # if there is a differentiated cell nearby, move toward it
                    if simulation.nearest_diff[index] != -1:
                        nearest_index = simulation.nearest_diff[index]
                        vector = simulation.locations[nearest_index] - simulation.locations[index]
                        normal = backend.normal_vector(vector)
                        random = backend.random_vector(simulation)
                        simulation.motility_forces[index] += (normal * 0.8 + random * 0.2) * motility_force

                    # if no nearby differentiated cells, move randomly
                    else:
                        simulation.motility_forces[index] += backend.random_vector(simulation) * motility_force

                # if the cell is nanog high and gata6 low
                elif simulation.NANOG[index] > simulation.GATA6[index]:
                    # if there is a nanog high cell nearby, move toward it
                    if simulation.nearest_nanog[index] != -1:
                        nearest_index = simulation.nearest_nanog[index]
                        vector = simulation.locations[nearest_index] - simulation.locations[index]
                        normal = backend.normal_vector(vector)
                        random = backend.random_vector(simulation)
                        simulation.motility_forces[index] += (normal * 0.8 + random * 0.2) * motility_force

                    # if there is a gata6 high cell nearby, move away from it
                    elif simulation.nearest_gata6[index] != -1:
                        nearest_index = simulation.nearest_gata6[index]
                        vector = simulation.locations[nearest_index] - simulation.locations[index]
                        normal = backend.normal_vector(vector)
                        random = backend.random_vector(simulation)
                        simulation.motility_forces[index] += (normal * -0.8 + random * 0.2) * motility_force

                    else:
                        simulation.motility_forces[index] += backend.random_vector(simulation) * motility_force

                # if both gata6/nanog high or both low, move randomly
                else:
                    simulation.motility_forces[index] += backend.random_vector(simulation) * motility_force


@backend.record_time
def get_neighbors(simulation):
    """ For all cells, determines which cells fall within a fixed
        radius to denote a neighbor then stores this information
        in a graph (uses a bin/bucket sorting method).
    """
    # radius of search (meters) in which all cells within are classified as neighbors
    neighbor_distance = 0.000015

    # if a static variable has not been created to hold the maximum number of neighbors for a cell, create one
    if not hasattr(get_neighbors, "max_neighbors"):
        # begin with a low number of neighbors that can be revalued if the max number of neighbors exceeds this value
        get_neighbors.max_neighbors = 5

    # if a static variable has not been created to hold the maximum number of cells in a bin, create one
    if not hasattr(get_neighbors, "max_cells"):
        # begin with a low number of cells that can be revalued if the max number of cells exceeds this value
        get_neighbors.max_cells = 5

    # clear all of the edges in the neighbor graph
    simulation.neighbor_graph.delete_edges(None)

    # calls the function that generates an array of bins that generalize the cell locations in addition to a
    # creating a helper array that assists the search method in counting cells for a particular bin
    bins, bins_help, bin_locations, max_cells = backend.assign_bins(simulation, neighbor_distance,
                                                                    get_neighbors.max_cells)

    # update the value of the max number of cells in a bin
    get_neighbors.max_cells = max_cells

    # this will run once if all edges are included in edge_holder, breaking the loop. if not, this will
    # run a second time with an updated value for the number of predicted neighbors such that all edges are included
    while True:
        # create an array used to hold edges, an array to say if edge exists, and an array to count the edges per cell
        length = simulation.number_cells * get_neighbors.max_neighbors
        edge_holder = np.zeros((length, 2), dtype=int)
        if_edge = np.zeros(length, dtype=bool)
        edge_count = np.zeros(simulation.number_cells, dtype=int)

        # call the nvidia gpu version
        if simulation.parallel:
            # turn the following into arrays that can be interpreted by the gpu
            bin_locations_cuda = cuda.to_device(bin_locations)
            locations_cuda = cuda.to_device(simulation.locations)
            bins_cuda = cuda.to_device(bins)
            bins_help_cuda = cuda.to_device(bins_help)
            distance_cuda = cuda.to_device(neighbor_distance)
            edge_holder_cuda = cuda.to_device(edge_holder)
            if_edge_cuda = cuda.to_device(if_edge)
            edge_count_cuda = cuda.to_device(edge_count)
            max_neighbors_cuda = cuda.to_device(get_neighbors.max_neighbors)

            # allocate threads and blocks for gpu memory "threads per block" and "blocks per grid"
            tpb = 72
            bpg = math.ceil(simulation.number_cells / tpb)

            # call the cuda kernel with given parameters
            backend.get_neighbors_gpu[bpg, tpb](bin_locations_cuda, locations_cuda, bins_cuda, bins_help_cuda,
                                                distance_cuda, edge_holder_cuda, if_edge_cuda, edge_count_cuda,
                                                max_neighbors_cuda)

            # return the arrays back from the gpu
            edge_holder = edge_holder_cuda.copy_to_host()
            if_edge = if_edge_cuda.copy_to_host()
            edge_count = edge_count_cuda.copy_to_host()

        # call the jit cpu version
        else:
            edge_holder, if_edge, edge_count = backend.get_neighbors_cpu(simulation.number_cells, bin_locations,
                                                                         simulation.locations, bins, bins_help,
                                                                         neighbor_distance, edge_holder, if_edge,
                                                                         edge_count, get_neighbors.max_neighbors)

        # either break the loop if all neighbors were accounted for or revalue the maximum number of neighbors
        # based on the output of the function call and double it for future calls
        max_neighbors = np.amax(edge_count)
        if get_neighbors.max_neighbors >= max_neighbors:
            break
        else:
            get_neighbors.max_neighbors = max_neighbors * 2

    # reduce the edges to only edges that actually exist
    edge_holder = edge_holder[if_edge]

    # add the edges to the neighbor graph
    simulation.neighbor_graph.add_edges(edge_holder)


@backend.record_time
def nearest(simulation):
    """ Determines the nearest GATA6 high, NANOG high, and
        differentiated cell within a fixed radius for each
        cell.
    """
    # radius of search (meters) for nearest cells of the three types
    nearest_distance = 0.000015

    # if a static variable has not been created to hold the maximum number of cells in a bin, create one
    if not hasattr(nearest, "max_cells"):
        # begin with a low number of cells that can be revalued if the max number of cells exceeds this value
        nearest.max_cells = 5

    # calls the function that generates an array of bins that generalize the cell locations in addition to a
    # creating a helper array that assists the search method in counting cells for a particular bin
    bins, bins_help, bin_locations, max_cells = backend.assign_bins(simulation, nearest_distance, nearest.max_cells)

    # update the value of the max number of cells in a bin
    nearest.max_cells = max_cells

    # turn the following array into True/False instead of strings
    if_diff = simulation.states == "Differentiated"

    # call the nvidia gpu version
    if simulation.parallel:
        # turn the following into arrays that can be interpreted by the gpu
        bin_locations_cuda = cuda.to_device(bin_locations)
        locations_cuda = cuda.to_device(simulation.locations)
        bins_cuda = cuda.to_device(bins)
        bins_help_cuda = cuda.to_device(bins_help)
        distance_cuda = cuda.to_device(nearest_distance)
        if_diff_cuda = cuda.to_device(if_diff)
        gata6_cuda = cuda.to_device(simulation.GATA6)
        nanog_cuda = cuda.to_device(simulation.NANOG)
        nearest_gata6_cuda = cuda.to_device(simulation.nearest_gata6)
        nearest_nanog_cuda = cuda.to_device(simulation.nearest_nanog)
        nearest_diff_cuda = cuda.to_device(simulation.nearest_diff)

        # allocate threads and blocks for gpu memory "threads per block" and "blocks per grid"
        tpb = 72
        bpg = math.ceil(simulation.number_cells / tpb)

        # call the cuda kernel with given parameters
        backend.nearest_gpu[bpg, tpb](bin_locations_cuda, locations_cuda, bins_cuda, bins_help_cuda, distance_cuda,
                                      if_diff_cuda, gata6_cuda, nanog_cuda, nearest_gata6_cuda, nearest_nanog_cuda,
                                      nearest_diff_cuda)

        # return the new nearest arrays back from the gpu
        gata6 = nearest_gata6_cuda.copy_to_host()
        nanog = nearest_nanog_cuda.copy_to_host()
        diff = nearest_diff_cuda.copy_to_host()

    # call the cpu version
    else:
        gata6, nanog, diff = backend.nearest_cpu(simulation.number_cells, bin_locations, simulation.locations,
                                                 bins, bins_help, nearest_distance, if_diff, simulation.GATA6,
                                                 simulation.NANOG, simulation.nearest_gata6, simulation.nearest_nanog,
                                                 simulation.nearest_diff)

    # revalue the array holding the indices of nearest cells of given type
    simulation.nearest_gata6 = gata6
    simulation.nearest_nanog = nanog
    simulation.nearest_diff = diff


@backend.record_time
def handle_movement(simulation):
    """ Runs the following functions together for the time period
        of the step. Resets the motility force array to zero after
        movement is done.
    """
    # if a static variable for holding the number of steps hasn't been created, create one
    if not hasattr(handle_movement, "steps"):
        # get the total amount of times the cells will be incrementally moved during the step
        handle_movement.steps = math.ceil(simulation.step_dt / simulation.move_dt)

    # run the following movement functions consecutively
    for _ in range(handle_movement.steps):
        # determines which cells will have physical interactions and save this to a graph
        jkr_neighbors(simulation)

        # go through the edges found in the above function and calculate resulting JKR forces
        get_forces(simulation)

        # apply all forces such as motility and JKR to the cells
        apply_forces(simulation)

    # reset motility forces back to zero
    simulation.motility_forces[:][:] = 0


@backend.record_time
def jkr_neighbors(simulation):
    """ For all cells, determines which cells will have physical
        interactions with other cells and puts this information
        into a graph.
    """
    # radius of search (meters) in which neighbors will have physical interactions, double the max cell radius
    jkr_distance = 2 * simulation.max_radius

    # if a static variable has not been created to hold the maximum number of neighbors for a cell, create one
    if not hasattr(jkr_neighbors, "max_neighbors"):
        # begin with a low number of neighbors that can be revalued if the max number of neighbors exceeds this value
        jkr_neighbors.max_neighbors = 5

    # if a static variable has not been created to hold the maximum number of cells in a bin, create one
    if not hasattr(jkr_neighbors, "max_cells"):
        # begin with a low number of cells that can be revalued if the max number of cells exceeds this value
        jkr_neighbors.max_cells = 5

    # this will run once if all edges are included in edge_holder, breaking the loop. if not, this will
    # run a second time with an updated value for the number of predicted neighbors such that all edges are included
    bins, bins_help, bin_locations, max_cells = backend.assign_bins(simulation, jkr_distance, jkr_neighbors.max_cells)

    # update the value of the max number of cells in a bin
    jkr_neighbors.max_cells = max_cells

    # this will run once and if all edges are included in edge_holder, the loop will break. if not this will
    # run a second time with an updated value for number of predicted neighbors such that all edges are included
    while True:
        # create an array used to hold edges, an array to say where edges are, and an array to count the edges per cell
        length = simulation.number_cells * jkr_neighbors.max_neighbors
        edge_holder = np.zeros((length, 2), dtype=int)
        if_edge = np.zeros(length, dtype=bool)
        edge_count = np.zeros(simulation.number_cells, dtype=int)

        # call the nvidia gpu version
        if simulation.parallel:
            # turn the following into arrays that can be interpreted by the gpu
            bin_locations_cuda = cuda.to_device(bin_locations)
            locations_cuda = cuda.to_device(simulation.locations)
            radii_cuda = cuda.to_device(simulation.radii)
            bins_cuda = cuda.to_device(bins)
            bins_help_cuda = cuda.to_device(bins_help)
            edge_holder_cuda = cuda.to_device(edge_holder)
            if_edge_cuda = cuda.to_device(if_edge)
            edge_count_cuda = cuda.to_device(edge_count)
            max_neighbors_cuda = cuda.to_device(jkr_neighbors.max_neighbors)

            # allocate threads and blocks for gpu memory "threads per block" and "blocks per grid"
            tpb = 72
            bpg = math.ceil(simulation.number_cells / tpb)

            # call the cuda kernel with given parameters
            backend.jkr_neighbors_gpu[bpg, tpb](bin_locations_cuda, locations_cuda, radii_cuda, bins_cuda,
                                                bins_help_cuda, edge_holder_cuda, if_edge_cuda, edge_count_cuda,
                                                max_neighbors_cuda)

            # return the arrays back from the gpu
            edge_holder = edge_holder_cuda.copy_to_host()
            if_edge = if_edge_cuda.copy_to_host()
            edge_count = edge_count_cuda.copy_to_host()

        # call the jit cpu version
        else:
            edge_holder, if_edge, edge_count = backend.jkr_neighbors_cpu(simulation.number_cells, bin_locations,
                                                                         simulation.locations, simulation.radii, bins,
                                                                         bins_help, edge_holder, if_edge, edge_count,
                                                                         jkr_neighbors.max_neighbors)

        # either break the loop if all neighbors were accounted for or revalue the maximum number of neighbors
        # based on the output of the function call and double it
        max_neighbors = np.amax(edge_count)
        if jkr_neighbors.max_neighbors >= max_neighbors:
            break
        else:
            jkr_neighbors.max_neighbors = max_neighbors * 2

    # reduce the edges to only nonzero edges
    edge_holder = edge_holder[if_edge]

    # add the edges and simplify the graph as this graph is never cleared due to its use for holding adhesive JKR
    # bonds from step to step
    simulation.jkr_graph.add_edges(edge_holder)
    simulation.jkr_graph.simplify()


@backend.record_time
def get_forces(simulation):
    """ Goes through all of "JKR" edges and quantifies any
        resulting adhesive or repulsion forces between
        pairs of cells.
    """
    # contact mechanics parameters that rarely change
    adhesion_const = 0.000107    # the adhesion constant in kg/s from P Pathmanathan et al.
    poisson = 0.5    # Poisson's ratio for the cells, 0.5 means incompressible
    youngs = 1000    # Young's modulus for the cells in Pa

    # get the edges as a numpy array, count them, and create an array used to delete edges from the JKR graph
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
            locations_cuda = cuda.to_device(simulation.locations)
            radii_cuda = cuda.to_device(simulation.radii)
            forces_cuda = cuda.to_device(simulation.jkr_forces)
            poisson_cuda = cuda.to_device(poisson)
            youngs_cuda = cuda.to_device(youngs)
            adhesion_const_cuda = cuda.to_device(adhesion_const)

            # allocate threads and blocks for gpu memory "threads per block" and "blocks per grid"
            tpb = 72
            bpg = math.ceil(number_edges / tpb)

            # call the cuda kernel with given parameters
            backend.get_forces_gpu[bpg, tpb](jkr_edges_cuda, delete_edges_cuda, locations_cuda, radii_cuda, forces_cuda,
                                             poisson_cuda, youngs_cuda, adhesion_const_cuda)

            # return the new forces and the edges to be deleted
            forces = forces_cuda.copy_to_host()
            delete_edges = delete_edges_cuda.copy_to_host()

        # call the cpu version
        else:
            forces, delete_edges = backend.get_forces_cpu(number_edges, jkr_edges, delete_edges, simulation.locations,
                                                          simulation.radii, simulation.jkr_forces, poisson, youngs,
                                                          adhesion_const)

        # update the jkr edges to remove any edges that have be broken and update the JKR forces array
        delete_edges_indices = np.arange(number_edges)[delete_edges]
        simulation.jkr_graph.delete_edges(delete_edges_indices)
        simulation.jkr_forces = forces


@backend.record_time
def apply_forces(simulation):
    """ Turns the motility and JKR forces acting on
        a cell into movement.
    """
    # contact mechanics parameters that rarely change
    viscosity = 10000    # the viscosity of the medium in Ns/m used for stokes friction

    # call the nvidia gpu version
    if simulation.parallel:
        # turn the following into arrays that can be interpreted by the gpu
        jkr_forces_cuda = cuda.to_device(simulation.jkr_forces)
        motility_forces_cuda = cuda.to_device(simulation.motility_forces)
        locations_cuda = cuda.to_device(simulation.locations)
        radii_cuda = cuda.to_device(simulation.radii)
        viscosity_cuda = cuda.to_device(viscosity)
        size_cuda = cuda.to_device(simulation.size)
        move_dt_cuda = cuda.to_device(simulation.move_dt)

        # allocate threads and blocks for gpu memory "threads per block" and "blocks per grid"
        tpb = 72
        bpg = math.ceil(simulation.number_cells / tpb)

        # call the cuda kernel with given parameters
        backend.apply_forces_gpu[bpg, tpb](jkr_forces_cuda, motility_forces_cuda, locations_cuda, radii_cuda,
                                           viscosity_cuda, size_cuda, move_dt_cuda)

        # return the new cell locations from the gpu
        new_locations = locations_cuda.copy_to_host()

    # call the cpu version
    else:
        new_locations = backend.apply_forces_cpu(simulation.number_cells, simulation.jkr_forces,
                                                 simulation.motility_forces, simulation.locations,
                                                 simulation.radii, viscosity, simulation.size,
                                                 simulation.move_dt)

    # update the locations and reset the jkr forces back to zero
    simulation.locations = new_locations
    simulation.jkr_forces[:][:] = 0.0


@backend.record_time
def update_diffusion(simulation):
    """ Goes through all indicated extracellular gradients and
        approximates the diffusion of the morphogen.
    """
    # go through all gradients and update the diffusion equation to each
    for gradient_name in simulation.gradient_names:
        # the simulation holds all gradients are 3D arrays for simplicity, so get the gradient as a 2D array instead
        gradient = simulation.__dict__[gradient_name][:, :, 0]

        # set max and min concentration values
        gradient[gradient > simulation.max_concentration] = simulation.max_concentration
        gradient[gradient < 0] = 0

        # create a slightly larger array to hold ghost points for initial conditions
        size = np.array(gradient.shape) + 2
        base = np.zeros(size, dtype=float)

        # paste the gradient into the middle of the base array
        base[1:-1, 1:-1] += gradient

        # call the JIT diffusion function
        gradient = backend.update_diffusion_jit(base, simulation.step_dt, simulation.diffuse_dt, simulation.spat_res2,
                                                simulation.diffuse_const)

        # set max and min concentration values again
        gradient[gradient > simulation.max_concentration] = simulation.max_concentration
        gradient[gradient < 0] = 0

        # update the simulation gradient array
        simulation.__dict__[gradient_name][:, :, 0] = gradient


@backend.record_time
def update_queue(simulation):
    """ Adds and removes cells to and from the simulation
        either all at once or in "groups".
    """
    # get the number of cells being added and removed
    num_added = len(simulation.cells_to_divide)
    num_removed = len(simulation.cells_to_remove)

    # print how many cells are being added/removed during a given step
    print("Adding " + str(num_added) + " cells...")
    print("Removing " + str(num_removed) + " cells...")

    # -------------------- Division --------------------
    # extend each of the arrays by how many cells being added
    for name in simulation.cell_array_names:
        # copy the indices of the cell array for the dividing cells
        copies = simulation.__dict__[name][simulation.cells_to_divide]

        # if the instance variable is 1-dimensional
        if simulation.__dict__[name].ndim == 1:
            # add the copies to the end of the array
            simulation.__dict__[name] = np.concatenate((simulation.__dict__[name], copies))

        # if the instance variable is 2-dimensional
        else:
            # add the copies to the end of the array
            simulation.__dict__[name] = np.concatenate((simulation.__dict__[name], copies), axis=0)

    # go through each of the dividing cells
    for i in range(num_added):
        # get the indices of the mother cell and the daughter cell
        mother_index = simulation.cells_to_divide[i]
        daughter_index = simulation.number_cells

        # move the cells to new positions
        division_position = backend.random_vector(simulation) * (simulation.max_radius - simulation.min_radius)
        simulation.locations[mother_index] += division_position
        simulation.locations[daughter_index] -= division_position

        # reduce both radii to minimum size (representative of a divided cell) and set the division counters to zero
        simulation.radii[mother_index] = simulation.radii[daughter_index] = simulation.min_radius
        simulation.div_counters[mother_index] = simulation.div_counters[daughter_index] = 0

        # go through each graph adding the number of dividing cells
        for graph_name in simulation.graph_names:
            simulation.__dict__[graph_name].add_vertex()

        # update the number of cells in the simulation
        simulation.number_cells += 1

        # if not adding all of the cells at once
        if simulation.group != 0:
            # Cannot add all of the new cells, otherwise several cells are likely to be added in
            #   close proximity to each other at later time steps. Such addition, coupled with
            #   handling collisions, make give rise to sudden changes in overall positions of
            #   cells within the simulation. Instead, collisions are handled after 'group' number
            #   of cells are added. - Daniel Cruz

            # if the current number added is divisible by the group number
            if (i + 1) % simulation.group == 0:
                # call the handle movement function to better simulate asynchronous division
                handle_movement(simulation)

    # -------------------- Death --------------------
    # get the indices of the cells leaving the simulation
    indices = simulation.cells_to_remove

    # go through the cell arrays remove the indices
    for name in simulation.cell_array_names:
        # if the array is 1-dimensional
        if simulation.__dict__[name].ndim == 1:
            simulation.__dict__[name] = np.delete(simulation.__dict__[name], indices)
        # if the array is 2-dimensional
        else:
            simulation.__dict__[name] = np.delete(simulation.__dict__[name], indices, axis=0)

    # automatically update the graphs and change the number of cells
    for graph_name in simulation.graph_names:
        simulation.__dict__[graph_name].delete_vertices(indices)
    simulation.number_cells -= num_removed

    # clear the arrays for the next step
    simulation.cells_to_divide = np.array([], dtype=int)
    simulation.cells_to_remove = np.array([], dtype=int)
