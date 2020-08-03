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
    # enough space update the amount of needed space and re-put the cells in bins. this will run once if the prediction
    # of max neighbors is correct, twice if it isn't the first time
    while True:
        # calculate the size of the array used to represent the bins and the bins helper array, include extra bins
        # for cells that may fall outside of the space
        bins_help_size = np.ceil(simulation.size / distance).astype(int) + np.array([5, 5, 5], dtype=int)
        bins_size = np.append(bins_help_size, assign_bins.max_cells)

        # create the arrays for "bins" and "bins_help"
        bins_help = np.zeros(bins_help_size, dtype=int)
        bins = np.empty(bins_size, dtype=int)

        # assign the cells to bins so that the searches may be parallel
        bins, bins_help = Backend.assign_bins_cpu(simulation.number_cells, simulation.cell_locations, distance, bins,
                                                  bins_help)

        # either break the loop if all cells were accounted for or revalue the maximum number of cells based on
        # the output of the function call
        max_cells = np.amax(bins_help)
        if assign_bins.max_cells >= max_cells:
            break
        else:
            assign_bins.max_cells = max_cells

    # return the two arrays
    return bins, bins_help


def clean_edges(edges):
    """ takes edges from "check_neighbors" and "jkr_neighbors"
        search methods and returns an array of edges that is
        free of duplicates and loops
    """
    # reshape the array so that the output is compatible with the igraph library and remove leftover zero columns
    edges = edges.reshape((-1, 2))
    edges = edges[~np.all(edges == 0, axis=1)]

    # sort the array to remove duplicate edges produced by the parallel search method
    edges = np.sort(edges)
    edges = edges[np.lexsort(np.fliplr(edges).T)]
    edges = edges[::2]

    # send the edges back
    return edges


def check_neighbors(simulation):
    """ for all cells, determines which cells fall within a fixed
        radius to denote a neighbor then stores this information
        in a graph (uses a bin/bucket sorting method)
    """
    # start time of the function
    simulation.check_neighbors_time = -1 * time.time()

    # if a static variable has not been created to hold the maximum number of neighbors, create one
    if not hasattr(check_neighbors, "max_neighbors"):
        # begin with a low number of neighbors that can be revalued if the max number of neighbors exceeds this value
        check_neighbors.max_neighbors = 5

    # clear all of the edges in the neighbor graph
    simulation.neighbor_graph.delete_edges(None)

    # calls the function that generates an array of bins that generalize the cell locations in addition to a
    # helper array that assists the search method in counting cells in a particular bin
    bins, bins_help = assign_bins(simulation, simulation.neighbor_distance)

    # this will run once and if all edges are included in edge_holder, the loop will break. if not this will
    # run a second time with an updated value for number of predicted neighbors such that all edges are included
    while True:
        # create a 3D array used to hold edges for all cells
        edge_holder = np.zeros((simulation.number_cells, check_neighbors.max_neighbors, 2), dtype=int)
        edge_count = np.zeros(simulation.number_cells, dtype=int)

        # call the nvidia gpu version
        if simulation.parallel:
            # turn the following into arrays that can be interpreted by the gpu
            bins_cuda = cuda.to_device(bins)
            bins_help_cuda = cuda.to_device(bins_help)
            distance_cuda = cuda.to_device(simulation.neighbor_distance)
            edge_holder_cuda = cuda.to_device(edge_holder)
            locations_cuda = cuda.to_device(simulation.cell_locations)
            edge_count_cuda = cuda.to_device(edge_count)

            # allocate threads and blocks for gpu memory
            threads_per_block = 72
            blocks_per_grid = math.ceil(simulation.number_cells / threads_per_block)

            # call the cuda kernel with given parameters
            Backend.check_neighbors_gpu[blocks_per_grid, threads_per_block](locations_cuda, bins_cuda, bins_help_cuda,
                                                                            distance_cuda, edge_holder_cuda,
                                                                            edge_count_cuda)
            # return the arrays back from the gpu
            edge_holder = edge_holder_cuda.copy_to_host()
            edge_count = edge_count_cuda.copy_to_host()

        # call the cpu version
        else:
            edge_holder, edge_count = Backend.check_neighbors_cpu(simulation.number_cells, simulation.cell_locations,
                                                                  bins, bins_help, simulation.neighbor_distance,
                                                                  edge_holder, edge_count,
                                                                  check_neighbors.max_neighbors)

        # either break the loop if all neighbors were accounted for or revalue the maximum number of neighbors
        # based on the output of the function call
        max_neighbors = np.amax(edge_count)
        if check_neighbors.max_neighbors >= max_neighbors:
            break
        else:
            check_neighbors.max_neighbors = max_neighbors

    # remove duplicates and loops from the array as these slow down "add_edges"
    edge_holder = clean_edges(edge_holder)

    # add the edges to the neighbor graph and simplify the graph to remove any extraneous loops or repeated edges
    simulation.neighbor_graph.add_edges(edge_holder)

    # calculate the total time elapsed for the function
    simulation.check_neighbors_time += time.time()


def handle_movement(simulation):
    """ runs the following functions together for the time
        period of the step. resets the motility force array
        to zero
    """
    # start time of the function
    simulation.handle_movement_time = -1 * time.time()

    # get the total amount of times the cells will be incrementally moved during the step
    steps = math.ceil(simulation.time_step_value / simulation.move_time_step)

    # run the following functions consecutively for the given amount of steps
    for i in range(steps):
        jkr_neighbors(simulation)
        get_forces(simulation)
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
        jkr_neighbors.max_neighbors = 5

    # calls the function that generates an array of bins that generalize the cell locations in addition to a
    # helper array that assists the search method in counting cells in a particular bin
    bins, bins_help = assign_bins(simulation, simulation.jkr_distance)

    # this will run once and if all edges are included in edge_holder, the loop will break. if not this will
    # run a second time with an updated value for number of predicted neighbors such that all edges are included
    while True:
        # create a 3D array used to hold edges for each of the cells
        edge_holder = np.zeros((simulation.number_cells, jkr_neighbors.max_neighbors, 2), dtype=int)
        edge_count = np.zeros(simulation.number_cells, dtype=int)

        # call the nvidia gpu version
        if simulation.parallel:
            # turn the following into arrays that can be interpreted by the gpu
            bins_cuda = cuda.to_device(bins)
            bins_help_cuda = cuda.to_device(bins_help)
            distance_cuda = cuda.to_device(simulation.jkr_distance)
            edge_holder_cuda = cuda.to_device(edge_holder)
            locations_cuda = cuda.to_device(simulation.cell_locations)
            radii_cuda = cuda.to_device(simulation.cell_radii)
            edge_count_cuda = cuda.to_device(edge_count)

            # allocate threads and blocks for gpu memory
            threads_per_block = 72
            blocks_per_grid = math.ceil(simulation.number_cells / threads_per_block)

            # call the cuda kernel with given parameters
            Backend.jkr_neighbors_gpu[blocks_per_grid, threads_per_block](locations_cuda, radii_cuda, bins_cuda,
                                                                          bins_help_cuda, distance_cuda,
                                                                          edge_holder_cuda, edge_count_cuda)
            # return the arrays back from the gpu
            edge_holder = edge_holder_cuda.copy_to_host()
            edge_count = edge_count_cuda.copy_to_host()

        # call the cpu version
        else:
            edge_holder, edge_count = Backend.jkr_neighbors_cpu(simulation.number_cells, simulation.cell_locations,
                                                                simulation.cell_radii, bins, bins_help,
                                                                simulation.jkr_distance, edge_holder, edge_count,
                                                                jkr_neighbors.max_neighbors)

        # either break the loop if all neighbors were accounted for or revalue the maximum number of neighbors
        # based on the output of the function call
        max_neighbors = np.amax(edge_count)
        if jkr_neighbors.max_neighbors >= max_neighbors:
            break
        else:
            jkr_neighbors.max_neighbors = max_neighbors

    # remove duplicates and loops from the array as these slow down "add_edges"
    edge_holder = clean_edges(edge_holder)

    # add the edges and simplify the graph as this is a running graph that is never cleared
    simulation.jkr_graph.add_edges(edge_holder)
    simulation.jkr_graph.simplify()

    # calculate the total time elapsed for the function
    simulation.jkr_neighbors_time += time.time()


def get_forces(simulation):
    """ goes through all of the cells and quantifies any forces
        arising from adhesion or repulsion between cells
    """
    # start time of the function
    simulation.get_forces_time = -1 * time.time()

    # count the number of edges and create an array used to record the deletion of edges
    jkr_edges = np.array(simulation.jkr_graph.get_edgelist())
    number_edges = len(jkr_edges)
    delete_edges = np.zeros(number_edges, dtype=bool)

    # only continue if exists, otherwise the just-in-time functions will throw errors
    if number_edges > 0:
        # call the gpu version
        if simulation.parallel:
            # convert these arrays into a form able to be read by the GPU
            jkr_edges_cuda = cuda.to_device(jkr_edges)
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
            forces, delete_edges = Backend.get_forces_cpu(jkr_edges, delete_edges, simulation.poisson,
                                                          simulation.youngs_mod, simulation.adhesion_const,
                                                          simulation.cell_locations, simulation.cell_radii,
                                                          simulation.cell_jkr_force)

        # update the jkr edges to remove any edges that have be broken and update the cell jkr forces
        delete_edges = delete_edges[delete_edges != 0]
        simulation.jkr_graph.delete_edges(delete_edges)
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
