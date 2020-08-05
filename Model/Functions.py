import numpy as np
import time
from numba import cuda
import math

import Backend


def info(simulation):
    """ gives an idea of how the simulation is running
        and records the beginning of the step in real time
    """
    # records the real time value of when a step begins
    simulation.step_start = time.time()

    # prints the current step number and the number of cells
    print("Step: " + str(simulation.current_step))
    print("Number of cells: " + str(simulation.number_cells))


def update_queue(simulation):
    """ add and removes cells to and from the simulation
        either all at once or some at a time
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
        Backend.divide_cell(simulation, index)

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
        Backend.remove_cell(simulation, index)

        # adjusts the indices as deleting part of the array may alter the indices to remove
        for j in range(i + 1, len(simulation.cells_to_remove)):
            # if the current cell being deleted falls after the index, shift the indices by 1
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

    # if a static variable has not been created to hold the maximum number of neighbors, create one
    if not hasattr(check_neighbors, "max_neighbors"):
        # begin with a low number of neighbors that can be revalued if the max number of neighbors exceeds this value
        check_neighbors.max_neighbors = 5

    # clear all of the edges in the neighbor graph
    simulation.neighbor_graph.delete_edges(None)

    # calls the function that generates an array of bins that generalize the cell locations in addition to a
    # helper array that assists the search method in counting cells in a particular bin
    bins, bins_help = Backend.assign_bins(simulation, simulation.neighbor_distance)

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
    edge_holder = Backend.clean_edges(edge_holder)

    # add the edges to the neighbor graph and simplify the graph to remove any extraneous loops or repeated edges
    simulation.neighbor_graph.add_edges(edge_holder)

    # calculate the total time elapsed for the function
    simulation.check_neighbors_time += time.time()


def handle_movement(simulation):
    """ runs the following functions together for the time
        period of the step. resets the motility force array
        to zero after movement is done
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
    bins, bins_help = Backend.assign_bins(simulation, simulation.jkr_distance)

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
    edge_holder = Backend.clean_edges(edge_holder)

    # add the edges and simplify the graph as this is a running graph that is never cleared
    simulation.jkr_graph.add_edges(edge_holder)
    simulation.jkr_graph.simplify()

    # calculate the total time elapsed for the function
    simulation.jkr_neighbors_time += time.time()


def get_forces(simulation):
    """ goes through all of "JKR" edges and quantifies
        any resulting adhesive or repulsion forces between
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
    delete_edges = np.zeros(number_edges, dtype=bool)

    # only continue if edges exist, otherwise the compiled functions will raise errors
    if number_edges > 0:
        # call the nvidia gpu version
        if simulation.parallel:
            # turn the following into arrays that can be interpreted by the gpu
            jkr_edges_cuda = cuda.to_device(jkr_edges)
            delete_edges_cuda = cuda.to_device(delete_edges)
            poisson_cuda = cuda.to_device(poisson)
            youngs_cuda = cuda.to_device(youngs)
            adhesion_const_cuda = cuda.to_device(adhesion_const)
            forces_cuda = cuda.to_device(simulation.cell_jkr_force)
            locations_cuda = cuda.to_device(simulation.cell_locations)
            radii_cuda = cuda.to_device(simulation.cell_radii)

            # allocate threads and blocks for gpu memory
            threads_per_block = 72
            blocks_per_grid = math.ceil(number_edges / threads_per_block)

            # call the cuda kernel with given parameters
            Backend.get_forces_gpu[blocks_per_grid, threads_per_block](jkr_edges_cuda, delete_edges_cuda,
                                                                       locations_cuda, radii_cuda, forces_cuda,
                                                                       poisson_cuda, youngs_cuda, adhesion_const_cuda)
            # return the new forces and the edges to be deleted
            forces = forces_cuda.copy_to_host()
            delete_edges = delete_edges_cuda.copy_to_host()

        # call the cpu version
        else:
            forces, delete_edges = Backend.get_forces_cpu(jkr_edges, delete_edges, simulation.cell_locations,
                                                          simulation.cell_radii, simulation.cell_jkr_force, poisson,
                                                          youngs, adhesion_const)

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
        move_time_step_cuda = cuda.to_device(simulation.move_time_step)

        # allocate threads and blocks for gpu memory
        threads_per_block = 72
        blocks_per_grid = math.ceil(simulation.number_cells / threads_per_block)

        # call the cuda kernel with given parameters
        Backend.apply_forces_gpu[blocks_per_grid, threads_per_block](jkr_forces_cuda, motility_forces_cuda,
                                                                     locations_cuda, radii_cuda, viscosity_cuda,
                                                                     size_cuda, move_time_step_cuda)
        # return the new cell locations from the gpu
        new_locations = locations_cuda.copy_to_host()

    # call the cpu version
    else:
        new_locations = Backend.apply_forces_cpu(simulation.number_cells, simulation.cell_jkr_force,
                                                 simulation.cell_motility_force, simulation.cell_locations,
                                                 simulation.cell_radii, viscosity, simulation.size,
                                                 simulation.move_time_step)

    # update the locations and reset the jkr forces back to zero
    simulation.cell_locations = new_locations
    simulation.cell_jkr_force = np.zeros((simulation.number_cells, 3))

    # calculate the total time elapsed for the function
    simulation.apply_forces_time += time.time()


def nearest(simulation):
    """ looks at cells within a given radius a determines
        the closest cells of important types
    """
    # start time of the function
    simulation.nearest_time = -1 * time.time()

    # calls the function that generates an array of bins that generalize the cell locations in addition to a
    # helper array that assists the search method in counting cells in a particular bin
    bins, bins_help = Backend.assign_bins(simulation, simulation.nearest_distance)

    # call the nvidia gpu version
    if simulation.parallel:
        pass

    # call the cpu version
    else:
        # find the nearest cell of each type with the external method, no gpu function yet
        gata6, nanog, diff = Backend.nearest_cpu(simulation.number_cells, simulation.nearest_distance, bins, bins_help,
                                                 simulation.cell_locations, simulation.cell_nearest_gata6,
                                                 simulation.cell_nearest_nanog, simulation.cell_nearest_diff,
                                                 simulation.cell_states, simulation.cell_fds)

    # revalue the array holding the indices of nearest cells of given type
    simulation.cell_nearest_gata6 = gata6
    simulation.cell_nearest_nanog = nanog
    simulation.cell_nearest_diff = diff

    # calculate the total time elapsed for the function
    simulation.nearest_time += time.time()


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
        diffuse_bins, diffuse_bins_help = Backend.setup_diffuse_bins_cpu(simulation.diffuse_locations, x_steps, y_steps,
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

    # parameters that rarely change
    diffuse = 0.0000000000000001    # the diffusion constant of the extracellular molecules
    dt = 1    # the time step for the diffusion approximation

    # calculate how many steps for the approximation
    time_steps = math.ceil(simulation.time_step_value / dt)

    # go through all gradients and update the diffusion of each
    for gradient in simulation.extracellular_names:
        simulation.__dict__[gradient] = Backend.update_diffusion_cpu(simulation.__dict__[gradient], time_steps, dt,
                                                                     simulation.dx2, simulation.dy2,
                                                                     simulation.dz2, diffuse, simulation.size)
    # calculate the total time elapsed for the function
    simulation.update_diffusion_time += time.time()
