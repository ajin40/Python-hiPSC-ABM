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
        assign_bins.max_cells = 5

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
        bins, bins_help = Backend.assign_bins_cpu(simulation.number_cells, distance, bins, bins_help,
                                                  simulation.cell_locations)

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
    """ for all cells, determines which cells fall within
        a fixed radius to denote a neighbor then stores
        this information in a graph
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
            edge_holder, max_array = Backend.check_neighbors_cpu(simulation.number_cells, simulation.cell_locations,
                                                                 bins, bins_help, simulation.neighbor_distance,
                                                                 edge_holder, check_neighbors.max_neighbors, max_array)

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


def handle_movement(simulation):
    """ runs the following functions together for the time
        period of the step. resets the motility force array
        to zero too.
    """
    # start time of the function
    simulation.handle_movement_time = -1 * time.time()

    # get the total amount of times the cells will be incrementally moved during the step
    steps = math.ceil(simulation.time_step_value / simulation.move_time_step)

    # run the following functions consecutively for the given amount of steps
    for i in range(steps):
        # update the jkr neighbors
        jkr_neighbors(simulation)

        # calculate the forces acting on each cell
        get_forces(simulation)

        # turn the forces into movement
        apply_forces(simulation)

    # reset all forces back to zero vectors
    simulation.cell_motility_force = np.zeros((simulation.number_cells, 3))

    # calculate the total time elapsed for the function
    simulation.handle_movement_time += time.time()


def jkr_neighbors(simulation):
    """ for all cells, determines which cells will have
        physical interactions with other cells returns
        this information as an array of edges
    """
    # start time of the function
    simulation.jkr_neighbors_time = -1 * time.time()

    # if a static variable has not been created to hold the maximum number of neighbors, create one
    if not hasattr(jkr_neighbors, "max_neighbors"):
        # begin with a low number of neighbors that can be revalued if the max number of neighbors exceeds this value
        jkr_neighbors.max_neighbors = 10

    # calls the function that generates an array of bins that generalize the cell locations in addition to a
    # helper array that assists the search method in counting cells in a particular bin
    bins, bins_help = assign_bins(simulation, simulation.jkr_distance)

    # this will run once and if all edges are included in edge_holder, the loop will break. if not this will
    # run a second time with an updated value for number of predicted neighbors such that all edges are included
    while True:
        # create a 3D array used to hold edges for each of the cells
        edge_holder = np.zeros((simulation.number_cells, jkr_neighbors.max_neighbors, 2), dtype=int)
        max_array = np.zeros(simulation.number_cells, dtype=int)

        # call the gpu version
        if simulation.parallel:
            # turn the following into arrays that can be interpreted by the gpu
            bins_cuda = cuda.to_device(bins)
            bins_help_cuda = cuda.to_device(bins_help)
            distance_cuda = cuda.to_device(simulation.jkr_distance)
            edge_holder_cuda = cuda.to_device(edge_holder)
            locations_cuda = cuda.to_device(simulation.cell_locations)
            radii_cuda = cuda.to_device(simulation.cell_radii)
            max_array_cuda = cuda.to_device(max_array)

            # sets up the correct allocation of threads and blocks
            threads_per_block = 72
            blocks_per_grid = math.ceil(simulation.number_cells / threads_per_block)

            # calls the cuda function with the given inputs
            Backend.jkr_neighbors_gpu[blocks_per_grid, threads_per_block](locations_cuda, radii_cuda, bins_cuda,
                                                                          bins_help_cuda, distance_cuda,
                                                                          edge_holder_cuda, max_array_cuda)
            # returns the array back from the gpu
            edge_holder = edge_holder_cuda.copy_to_host()
            max_array = max_array_cuda.copy_to_host()

        # call the cpu version
        else:
            edge_holder, max_array = Backend.jkr_neighbors_cpu(simulation.number_cells, simulation.jkr_distance,
                                                               edge_holder, bins, bins_help, simulation.cell_locations,
                                                               simulation.cell_radii, jkr_neighbors.max_neighbors,
                                                               max_array)

        # either break the loop if all neighbors were accounted for or revalue the maximum number of neighbors
        # based on the output of the function call
        max_neighbors = np.amax(max_array)
        if jkr_neighbors.max_neighbors >= max_neighbors:
            break
        else:
            jkr_neighbors.max_neighbors = max_neighbors

    # reshape the array so that the output is compatible with the igraph library and remove leftover zero columns
    edge_holder = edge_holder.reshape((-1, 2))
    edge_holder = edge_holder[~np.all(edge_holder == 0, axis=1)]

    # sort the array to remove duplicate edges produced by the parallel search method
    edge_holder = np.sort(edge_holder)
    edge_holder = np.sort(edge_holder, axis=0)
    edge_holder = edge_holder[::2]

    if len(edge_holder) > 0:
        # add the edges to the neighbor graph and simplify the graph to remove any extraneous loops or repeated edges
        simulation.jkr_edges = np.append(simulation.jkr_edges, edge_holder, axis=0)
        # print(len(simulation.jkr_edges))
        simulation.jkr_edges = np.unique(simulation.jkr_edges, axis=0)
        # print(len(simulation.jkr_edges))

    # calculate the total time elapsed for the function
    simulation.jkr_neighbors_time += time.time()


def get_forces(simulation):
    """ goes through all of the cells and quantifies any forces
        arising from adhesion or repulsion between cells
    """
    # start time of the function
    simulation.get_forces_time = -1 * time.time()

    # count the number of edges and create an array used to record the deletion of edges
    number_edges = len(simulation.jkr_edges)
    delete_edges = np.ones(number_edges, dtype=bool)

    # only continue if exists, otherwise the just-in-time functions will throw errors
    if number_edges > 0:
        # call the gpu version
        if simulation.parallel:
            # convert these arrays into a form able to be read by the GPU
            jkr_edges_cuda = cuda.to_device(simulation.jkr_edges)
            delete_edges_cuda = cuda.to_device(delete_edges)
            poisson_cuda = cuda.to_device(simulation.poisson)
            youngs_cuda = cuda.to_device(simulation.youngs_mod)
            adhesion_const_cuda = cuda.to_device(simulation.adhesion_const)
            forces_cuda = cuda.to_device(simulation.cell_jkr_force)
            locations_cuda = cuda.to_device(simulation.cell_locations)
            radii_cuda = cuda.to_device(simulation.cell_radii)

            # sets up the correct allocation of threads and blocks
            threads_per_block = 72
            blocks_per_grid = math.ceil(number_edges / threads_per_block)

            # call the cuda function
            Backend.get_forces_gpu[blocks_per_grid, threads_per_block](jkr_edges_cuda, delete_edges_cuda,
                                                                       locations_cuda, radii_cuda, forces_cuda,
                                                                       poisson_cuda, youngs_cuda, adhesion_const_cuda)
            # return the new forces and the edges to be deleted
            forces = forces_cuda.copy_to_host()
            delete_edges = delete_edges_cuda.copy_to_host()

        # call the cpu version
        else:
            forces, delete_edges = Backend.get_forces_cpu(simulation.jkr_edges, delete_edges, simulation.poisson,
                                                          simulation.youngs_mod, simulation.adhesion_const,
                                                          simulation.cell_locations, simulation.cell_radii,
                                                          simulation.cell_jkr_force)

        # update the jkr edges to remove any edges that have be broken and update the cell jkr forces
        # print(len(simulation.jkr_edges))
        simulation.jkr_edges = simulation.jkr_edges[delete_edges]
        # print(len(simulation.jkr_edges))
        simulation.cell_jkr_force = forces

    # calculate the total time elapsed for the function
    simulation.get_forces_time += time.time()


def apply_forces(simulation):
    """ Turns the active motility/division forces
        and inactive JKR forces into movement
    """
    # start time of the function
    simulation.apply_forces_time = -1 * time.time()

    # call the gpu version
    if simulation.parallel:
        # turn these arrays into gpu array
        jkr_forces_cuda = cuda.to_device(simulation.cell_jkr_force)
        motility_forces_cuda = cuda.to_device(simulation.cell_motility_force)
        locations_cuda = cuda.to_device(simulation.cell_locations)
        radii_cuda = cuda.to_device(simulation.cell_radii)
        viscosity_cuda = cuda.to_device(simulation.viscosity)
        size_cuda = cuda.to_device(simulation.size)
        move_time_step_cuda = cuda.to_device(simulation.move_time_step)

        # sets up the correct allocation of threads and blocks
        threads_per_block = 72
        blocks_per_grid = math.ceil(simulation.number_cells / threads_per_block)

        # call the cuda function
        Backend.apply_forces_gpu[blocks_per_grid, threads_per_block](jkr_forces_cuda, motility_forces_cuda,
                                                                     locations_cuda, radii_cuda, viscosity_cuda,
                                                                     size_cuda, move_time_step_cuda)
        # return the new cell locations from the gpu
        new_locations = locations_cuda.copy_to_host()

    # call the cpu version
    else:
        new_locations = Backend.apply_forces_cpu(simulation.number_cells, simulation.cell_jkr_force,
                                                 simulation.cell_motility_force, simulation.cell_locations,
                                                 simulation.cell_radii, simulation.viscosity, simulation.size,
                                                 simulation.move_time_step)

    # update the locations and reset the jkr forces back to zero
    simulation.cell_locations = new_locations
    simulation.cell_jkr_force = np.zeros((simulation.number_cells, 3))

    # calculate the total time elapsed for the function
    simulation.apply_forces_time += time.time()
