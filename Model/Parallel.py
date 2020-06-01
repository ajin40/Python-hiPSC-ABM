from numba import cuda
import math
import numpy as np


# def update_gradient_gpu(extracellular, simulation):
#     """ This is a near identical function to the
#         non-parallel one; however, this uses
#         cupy which is identical to numpy, but
#         it is run on a cuda gpu instead
#     """
#     # get the number of times this will be run
#     time_steps = int(simulation.time_step_value / extracellular.dt)
#
#     # make the variable name smaller for easier writing
#     a = cp.asarray(extracellular.diffuse_values)
#
#     # perform the following operations on the diffusion points at each time step
#     for i in range(time_steps):
#
#         x = (a[2:][1:-1][1:-1] - 2 * a[1:-1][1:-1][1:-1] + a[:-2][1:-1][1:-1]) / extracellular.dx2
#         y = (a[1:-1][2:][1:-1] - 2 * a[1:-1][1:-1][1:-1] + a[1:-1][:-2][1:-1]) / extracellular.dy2
#         z = (a[1:-1][1:-1][2:] - 2 * a[1:-1][1:-1][1:-1] + a[1:-1][1:-1][:-2]) / extracellular.dz2
#
#         # update the array, assign a variable for ease of writing
#         new_value = a[1:-1][1:-1][1:-1] + extracellular.diffuse_const * extracellular.dt * (x + y + z)
#         a[1:-1][1:-1][1:-1] = new_value
#
#     # turn it back into a numpy array
#     extracellular.diffuse_values = cp.asnumpy(a)


@cuda.jit(device=True)
def magnitude(location_one, location_two):
    """ This is a cuda device function for
        finding magnitude give two vectors
    """
    total = 0
    for i in range(0, 3):
        total += (location_one[i] - location_two[i]) ** 2

    return total ** 0.5


def check_neighbors_gpu(simulation, distance, edge_holder, max_neighbors):
    """ The GPU parallelized version of check_neighbors()
        from the Simulation class.
    """
    # turn the following into arrays that can be interpreted by the gpu
    distance_cuda = cuda.to_device(distance)
    max_neighbors_cuda = cuda.to_device(max_neighbors)
    edge_holder_cuda = cuda.to_device(edge_holder)

    # divides the space into bins and gives a holder of fixed size for each bin
    bins_size = simulation.size // distance + np.array([3, 3, 3])
    bins_size_help = tuple(bins_size.astype(int))
    bins_size = np.append(bins_size, 100)
    bins_size = tuple(bins_size.astype(int))

    # assigns values of -1 to denote a lack of cells
    bins = np.ones(bins_size) * -1

    # an array used to accelerate the cuda function by telling the function how many cells are in a given bin
    bins_help = np.zeros(bins_size_help, dtype=np.int)

    # assigns cells to bins as a general location
    for i in range(len(simulation.cells)):
        # offset bins by 1 to avoid missing cells
        block_location = simulation.cells[i].location // distance + np.array([1, 1, 1])
        block_location = block_location.astype(int)
        x, y, z = block_location[0], block_location[1], block_location[2]

        # tries to place the cell in the holder for the bin. if the holder's value is other than -1 it will move
        # to the next spot to see if it's empty
        place = bins_help[x][y][z]

        # gives the cell's array location
        bins[x][y][z][place] = i

        # updates the total amount cells in a bin
        bins_help[x][y][z] += 1

    # turn the bins array and the blocks_help array into a format to be sent to the gpu
    bins_cuda = cuda.to_device(bins)
    bins_help_cuda = cuda.to_device(bins_help)

    # loops over all over the cells and puts their locations into a holder array and turns it into a gpu array
    locations = np.zeros((len(simulation.cells), 3))
    for i in range(len(simulation.cells)):
        locations[i] = simulation.cells[i].location
    locations_cuda = cuda.to_device(locations)

    # sets up the correct allocation of threads and blocks
    threads_per_block = 72
    blocks_per_grid = math.ceil(len(locations) / threads_per_block)

    # calls the cuda function with the given inputs
    check_neighbors_cuda[blocks_per_grid, threads_per_block](locations_cuda, bins_cuda, bins_help_cuda,
                                                             distance_cuda, edge_holder_cuda, max_neighbors_cuda)
    # return the array back from the GPU
    new_edges = edge_holder_cuda.copy_to_host()

    # return the edges that will be added to the graph
    return new_edges


@cuda.jit
def check_neighbors_cuda(locations, bins, bins_help, distance, edge_holder, max_neighbors):
    """ This is the parallelized function for checking
        neighbors that is run numerous times.
    """
    # a provides the location on the array as it runs, essentially loops over the cells
    index_1 = cuda.grid(1)

    # identify the location on the edge holder where the function will begin writing edges
    place = cuda.grid(1) * max_neighbors[0]

    # checks to see that position is in the array, double-check as GPUs can be weird sometimes
    if index_1 < locations.shape[0]:
        # gets the block location based on how they were inputted
        location_x = int(locations[index_1][0] / distance[0]) + 1
        location_y = int(locations[index_1][1] / distance[0]) + 1
        location_z = int(locations[index_1][2] / distance[0]) + 1

        # looks at the blocks surrounding the current block as these are the ones containing the neighbors
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # gets the number of cells in each block thanks to the helper array, int to prevent problems
                    number_cells = int(bins_help[location_x + i][location_y + j][location_z + k])

                    # loops over the cell indices in the current block
                    for l in range(number_cells):
                        # gets the index of the potential neighbor
                        index_2 = int(bins[location_x + i][location_y + j][location_z + k][l])

                        # get the magnitude via the device function and make sure not the same cell
                        if magnitude(locations[index_1], locations[index_2]) <= distance[0] and \
                                index_1 != index_2:
                            # assign the array location showing that this cell is a neighbor
                            edge_holder[place][0] = index_1
                            edge_holder[place][1] = index_2
                            place += 1


def get_forces_gpu(simulation, edges, jkr_edges, add_jkr_edges, delete_jkr_edges, poisson, youngs, adhesion_const):
    """ The GPU parallelized version of forces_to_movement()
        from the Simulation class.
    """
    bond = np.empty((0, 1), bool)
    for i in range(len(edges)):
        if edges[i] in jkr_edges:
            bond = np.append(bond, True)
        else:
            bond = np.append(bond, False)

    bond_cuda = cuda.to_device(bond)

    # convert these arrays into a form able to be read by the GPU
    edges_cuda = cuda.to_device(edges)
    jkr_edges_cuda = cuda.to_device(jkr_edges)
    add_jkr_edges_cuda = cuda.to_device(add_jkr_edges)
    delete_jkr_edges_cuda = cuda.to_device(delete_jkr_edges)
    poisson_cuda = cuda.to_device([poisson])
    youngs_cuda = cuda.to_device([youngs])
    adhesion_const_cuda = cuda.to_device([adhesion_const])
    forces_cuda = cuda.to_device(np.zeros((len(simulation.cells), 3)))

    # loops over all over the cells and puts their locations into a holder array and turns it into a gpu array
    locations = np.empty((0, 3), float)
    radii = np.empty((0, 1), float)

    for i in range(len(simulation.cells)):
        locations = np.append(locations, [simulation.cells[i].location], axis=0)
        radii = np.append(radii, simulation.cells[i].radius)

    locations_cuda = cuda.to_device(locations)
    radii_cuda = cuda.to_device(radii)

    # sets up the correct allocation of threads and blocks
    threads_per_block = 72
    blocks_per_grid = math.ceil(len(edges) / threads_per_block)

    get_forces_cuda[blocks_per_grid, threads_per_block](edges_cuda, jkr_edges_cuda, add_jkr_edges_cuda,
                                                        delete_jkr_edges_cuda, locations_cuda, radii_cuda,
                                                        forces_cuda, poisson_cuda, youngs_cuda, adhesion_const_cuda,
                                                        bond_cuda)

    forces_output = forces_cuda.copy_to_host()

    for i in range(len(simulation.cells)):
        simulation.cells[i].inactive_force += forces_output[i]

    add_jkr_edges_output = add_jkr_edges_cuda.copy_to_host()
    delete_jkr_edges_output = delete_jkr_edges_cuda.copy_to_host()

    return add_jkr_edges_output, delete_jkr_edges_output


@cuda.jit
def get_forces_cuda(edges, jkr_edges, add_jkr_edges, delete_jkr_edges, locations, radii, forces, poisson, youngs,
                    adhesion_const, bond):

    # a provides the location on the array as it runs, essentially loops over the cells
    edge_index = cuda.grid(1)

    # checks to see that position is in the array, double-check as GPUs can be weird sometimes
    if edge_index < edges.shape[0]:

        # get the indices of the cells in the edge
        index_1 = edges[edge_index][0]
        index_2 = edges[edge_index][1]

        # get the normal vector
        location_1 = locations[index_1]
        location_2 = locations[index_2]
        vector_x = location_1[0] - location_2[0]
        vector_y = location_1[1] - location_2[1]
        vector_z = location_1[2] - location_2[2]
        displacement = magnitude(location_1, location_2)

        if displacement != 0:
            normal_x = vector_x / displacement
            normal_y = vector_y / displacement
            normal_z = vector_z / displacement
        else:
            normal_x, normal_y, normal_z = 0, 0, 0

        # get the total overlap of the cells used later in calculations
        overlap = radii[index_1] + radii[index_2] - displacement

        # indicate that an adhesive bond has formed between the cells
        if overlap >= 0:
            add_jkr_edges[edge_index][0] = index_1
            add_jkr_edges[edge_index][1] = index_2

        # gets two values used for JKR
        e_hat = (((1 - poisson[0] ** 2) / youngs[0]) + ((1 - poisson[0] ** 2) / youngs[0])) ** -1
        r_hat = ((1 / radii[index_1]) + (1 / radii[index_2])) ** -1

        # used to calculate the max adhesive distance after bond has been already formed
        overlap_ = (((math.pi * adhesion_const[0]) / e_hat) ** (2 / 3)) * (r_hat ** (1 / 3))

        # get the nondimensionalized overlap, used for later calculations and checks
        # also for the use of a polynomial approximation of the force
        d = overlap / overlap_

        # used to see if the adhesive bond once formed has broken
        overlap_condition = d > -0.360562

        bond_condition = bond[edge_index]

        # check to see if the cells will have a force interaction
        if overlap_condition and bond_condition:
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
        elif bond_condition:
            delete_jkr_edges[edge_index] = edge_index


def apply_forces_gpu(simulation):
    """ The GPU parallelized version of apply_forces()
        from the Simulation class.
    """
    # get the length of the arrays and create them so that they can later be updated
    length = len(simulation.cells)
    inactive_forces = np.empty((length, 3), np.float)
    active_forces = np.empty((length, 3), np.float)
    locations = np.empty((length, 3), np.float)
    radii = np.empty(length, np.float)

    # loop over all cells and input their values to the arrays
    for i in range(len(simulation.cells)):
        inactive_forces[i] = simulation.cells[i].inactive_force
        active_forces[i] = simulation.cells[i].active_force
        locations[i] = simulation.cells[i].location
        radii[i] = simulation.cells[i].radius

    # turn those arrays into gpu arrays
    inactive_forces_cuda = cuda.to_device(inactive_forces)
    active_forces_cuda = cuda.to_device(active_forces)
    locations_cuda = cuda.to_device(locations)
    radii_cuda = cuda.to_device(radii)
    viscosity_cuda = cuda.to_device(simulation.viscosity)
    size_cuda = cuda.to_device(simulation.size)
    move_time_step_cuda = cuda.to_device(simulation.move_time_step)

    # sets up the correct allocation of threads and blocks
    threads_per_block = 72
    blocks_per_grid = math.ceil(len(locations) / threads_per_block)

    # call the cuda function
    apply_forces_cuda[blocks_per_grid, threads_per_block](inactive_forces_cuda, active_forces_cuda, locations_cuda,
                                                          radii_cuda, viscosity_cuda, size_cuda, move_time_step_cuda)

    # return the new cell locations from the gpu
    new_locations = locations_cuda.copy_to_host()

    # loop over all cells updating the locations and setting the inactive_force back to zero
    for i in range(len(simulation.cells)):
        simulation.cells[i].location = new_locations[i]
        simulation.cells[i].inactive_force = np.array([0.0, 0.0, 0.0])


@cuda.jit
def apply_forces_cuda(inactive_forces, active_forces, locations, radii, viscosity, size, move_time_step):
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
