from numba import cuda
import math
import numpy as np



def initialize_grid_gpu(simulation):
    """ the parallel form of "initialize_grid"
    """
    # sets up the correct allocation of threads and blocks
    threads_per_block = (10, 10, 10)
    blocks_per_grid_x = math.ceil(simulation.grid.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(simulation.grid.shape[1] / threads_per_block[1])
    blocks_per_grid_z = math.ceil(simulation.grid.shape[2] / threads_per_block[2])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

    # turns the grid into a form able to send to the gpu
    cuda_grid = cuda.to_device(simulation.grid)

    # calls the cuda function, gives the correct allocation and sends the grid
    initialize_grid_cuda[blocks_per_grid, threads_per_block](cuda_grid)

    # returns the grid from the cuda function and updates the simulation instance variable
    simulation.grid = cuda_grid.copy_to_host()

@cuda.jit
def initialize_grid_cuda(grid_array):
    """ this is the function being run in parallel
    """
    # a, b, and c provide the location on the array as this parallel function runs
    # the "3" tells cuda to return all three positions
    a, b, c = cuda.grid(3)

    # checks that the location of the current thread is within the array and updates the value
    if a < grid_array.shape[0] and b < grid_array.shape[1] and c < grid_array.shape[2]:
        grid_array[a][b][c] += 5.0


def update_grid_gpu(simulation):
    """ the parallel form of "update_grid"
    """
    # sets up the correct allocation of threads and blocks
    threads_per_block = (10, 10, 10)
    blocks_per_grid_x = math.ceil(simulation.grid.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(simulation.grid.shape[1] / threads_per_block[1])
    blocks_per_grid_z = math.ceil(simulation.grid.shape[2] / threads_per_block[2])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

    # turns the grid into a form able to send to the gpu
    cuda_grid = cuda.to_device(simulation.grid)

    # calls the cuda function, gives the correct allocation and sends the grid
    initialize_grid_cuda[blocks_per_grid, threads_per_block](cuda_grid)

    # returns the grid from the cuda function and updates the simulation instance variable
    simulation.grid = cuda_grid.copy_to_host()

@cuda.jit
def update_grid_cuda(grid_array):
    """ this is the function being run in parallel
    """
    # a, b, and c provide the location on the array as this parallel function runs
    # the "3" tells cuda to return all three positions
    a, b, c = cuda.grid(3)

    # checks that the location of the current thread is within the array and updates the value
    if a < grid_array.shape[0] and b < grid_array.shape[1] and c < grid_array.shape[2] and grid_array[a][b][c] > 0:
        grid_array[a][b][c] -= 1.0


def check_neighbors_gpu(simulation):
    """ the parallel form of "check_neighbors"
    """
    # loops over all over the cells and puts their locations into a holder array
    location_array = np.empty((0, 3), int)

    for i in range(len(simulation.cells)):
        location_array = np.append(location_array, np.array([simulation.cells[i].location]), axis=0)

    # an array to represent the connections between cells
    edges_array = np.zeros((len(simulation.cells), len(simulation.cells)))

    # turns the arrays and value into a form able to send to the gpu
    location_array_cuda = cuda.to_device(location_array)
    edges_array_cuda = cuda.to_device(edges_array)
    distance_cuda = cuda.to_device(simulation.neighbor_distance)

    # sets up the correct allocation of threads and blocks
    threads_per_block = 32
    blocks_per_grid = math.ceil(location_array.size / threads_per_block)

    # calls the cuda function
    check_neighbors_cuda[blocks_per_grid, threads_per_block](location_array_cuda, edges_array_cuda, distance_cuda)

    # returns the array back from the gpu
    output = edges_array_cuda.copy_to_host()

    # turns the array into an upper triangular matrix the code will create duplicate edges and this corrects that
    output = np.triu(output)

    # looks for 1's designating edges gives back an array with indices in simulation.cells locating the cell
    # example [[15 392] [732 4] [923 284]]
    edges = np.argwhere(output == 1)

    # re-adds the cells as nodes
    for i in range(len(simulation.cells)):
        simulation.network.add_node(simulation.cells[i])

    # forms an edge between cells based on the results from edges
    for i in range(len(edges)):
        simulation.network.add_edge(simulation.cells[edges[i][0]], simulation.cells[edges[i][1]])

@cuda.jit
def check_neighbors_cuda(locations, edges, distance):
    """ this is the function being run in parallel
    """
    # i provides the location on the array as it runs
    i = cuda.grid(1)

    # checks to see that position is in the array
    if i < locations.shape[0]:

        # loops over all cells
        for j in range(locations.shape[0]):

            # if the distance between the cells is less than or equal to the neighbor distance
            if magnitude(locations[i], locations[j]) <= distance[0]:
                # no edge for the cell and itself. "1" denotes an edge
                if i != j:
                    edges[i][j] = 1


def handle_collisions_gpu(simulation):
    """ the parallel form of "handle_collisions"
    """
    # the while loop controls the amount of time steps for movement
    time_counter = 0

    # arrays that hold the cell locations, radii, and masses
    location_array = np.empty((0, 3), int)
    velocities_array = np.empty((0, 3), int)
    radius_array = np.array([])
    mass_array = np.array([])

    # loops over all over the cells and puts their locations into a holder array
    for i in range(len(simulation.cells)):
        location_array = np.append(location_array, np.array([simulation.cells[i].location]), axis=0)
        velocities_array = np.append(velocities_array, np.array([simulation.cells[i].velocity]), axis=0)
        radius_array = np.append(radius_array, np.array([simulation.cells[i].radius]), axis=0)
        mass_array = np.append(mass_array, np.array([simulation.cells[i].mass]), axis=0)

    # turns the arrays into a form able to send to the gpu
    location_array = cuda.to_device(location_array)
    velocities_array = cuda.to_device(velocities_array)
    radius_array = cuda.to_device(radius_array)
    mass_array = cuda.to_device(mass_array)
    energy_kept = cuda.to_device(simulation.energy_kept)
    spring_constant = cuda.to_device(simulation.spring_constant)
    time_step = cuda.to_device(simulation.time_step)
    grid_size = cuda.to_device(simulation.size)
    friction = cuda.to_device(simulation.friction)

    # sets up the correct allocation of threads and blocks
    threads_per_block = 36
    blocks_per_grid = math.ceil(location_array.size / threads_per_block)

    while time_counter <= simulation.move_max_time:
        # smaller the time step, less error from missing collisions
        time_counter += simulation.move_time_step

        handle_collisions_cuda[blocks_per_grid, threads_per_block](location_array, radius_array, mass_array,
                                                                   energy_kept, spring_constant, velocities_array)

        update_locations_cuda[blocks_per_grid, threads_per_block](location_array, velocities_array, time_step,
                                                                  grid_size, friction)

    # returns the array
    new_velocities = velocities_array.copy_to_host()
    new_locations = location_array.copy_to_host()

    # updates the velocities
    for i in range(len(simulation.cells)):
        simulation.cells[i].velocity = new_velocities[i]
        simulation.cells[i].location = new_locations[i]

@cuda.jit
def handle_collisions_cuda(locations, radius, mass, energy, spring, velocities):
    """ this is the function being run in parallel
    """
    # i provides the location on the array as it runs
    i = cuda.grid(1)

    # checks to see that position is in the array
    if i < locations.shape[0]:

        # loops over all cells
        for j in range(locations.shape[0]):

            # gets the magnitude of the vector between the cell locations
            mag = magnitude(locations[i], locations[j])

            # if the cells are overlapping then proceed
            if mag <= radius[i] + radius[j]:

                # loops over all directions
                for k in range(0, 3):

                    # gets the distance between the cells for x, y, and z
                    displacement = locations[i][k] - locations[j][k]

                    # gets a normal vector of the vector between the centers of both cells
                    if mag == 0.0:
                        normal = 0.0

                    else:
                        normal = displacement / mag

                    # find the displacement of the membrane overlap for each cell
                    obj1_displacement = radius[i] * normal
                    obj2_displacement = radius[j] * normal

                    # find the overlap vector
                    overlap = ((obj1_displacement + obj2_displacement) - displacement) / 2

                    # converts the spring energy into kinetic energy in opposing directions
                    velocities[i][k] += overlap * (energy[0] * spring[0] / mass[i]) ** 0.5
                    velocities[j][k] -= overlap * (energy[0] * spring[0] / mass[j]) ** 0.5

@cuda.jit
def update_locations_cuda(locations, velocities, time_step, grid_size, friction):
    """ this is the function being run in parallel
    """
    # a provides the location on the array as it runs
    a = cuda.grid(1)

    # checks to see that position is in the array
    if a < locations.shape[0]:
        v = velocities[a]

        # loops over all directions of space
        for i in range(0, 3):
            movement = v[i] * time_step[i]
            new_location = locations[a][i] + movement

            # check if new location is in environment space if not simulation a collision with the bounds
            if new_location >= grid_size[i]:
                velocities[a][i] *= -0.5
                locations[a][i] = grid_size[i] - 0.001
            elif new_location < 0:
                velocities[a][i] *= -0.5
                locations[a][i] = 0.0
            else:
                locations[a][i] = new_location

            # subtracts the work from the kinetic energy and recalculates a new velocity
            sign = math.copysign(1.0, velocities[a][i])
            velocities[a][i] = sign * max(velocities[a][i] ** 2 - 2 * friction[i] * abs(movement), 0.0) ** 0.5


@cuda.jit(device=True)
def magnitude(location_one, location_two):
    total = 0
    for i in range(0, 3):
        total += (location_one[i] - location_two[i]) ** 2

    return total ** 0.5