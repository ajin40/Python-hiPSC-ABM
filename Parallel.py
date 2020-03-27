from numba import cuda
import math
import numpy as np


def initialize_grid_gpu(self):
    """ the parallel form of "initialize_grid"
    """
    # sets up the correct allocation of threads and blocks
    threads_per_block = (10, 10, 10)
    blocks_per_grid_x = math.ceil(self.grid.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(self.grid.shape[1] / threads_per_block[1])
    blocks_per_grid_z = math.ceil(self.grid.shape[2] / threads_per_block[2])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

    # turns the grid into a form able to send to the gpu
    cuda_grid = cuda.to_device(self.grid)

    # calls the cuda function
    initialize_grid_cuda[blocks_per_grid, threads_per_block](cuda_grid)

    # returns the grid and reassigns the new grid
    self.grid = cuda_grid.copy_to_host()

@cuda.jit
def initialize_grid_cuda(grid_array):
    """ this is the function being run in parallel
    """
    # a and b provide the location on the array as it runs
    a, b, c = cuda.grid(3)

    # checks that the location is within the array
    if a < grid_array.shape[0] and b < grid_array.shape[1] and c < grid_array.shape[2]:
        grid_array[a][b][c] += 5.0


def update_grid_gpu(self):
    """ the parallel form of "update_grid"
    """
    # sets up the correct allocation of threads and blocks
    threads_per_block = (10, 10, 10)
    blocks_per_grid_x = math.ceil(self.grid.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(self.grid.shape[1] / threads_per_block[1])
    blocks_per_grid_z = math.ceil(self.grid.shape[2] / threads_per_block[2])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

    # turns the grid into a form able to send to the gpu
    cuda_grid = cuda.to_device(self.grid)

    # calls the cuda function
    update_grid_cuda[blocks_per_grid, threads_per_block](cuda_grid)

    # returns the grid and reassigns the new grid
    self.grid = cuda_grid.copy_to_host()

@cuda.jit
def update_grid_cuda(grid_array):
    """ this is the function being run in parallel
    """
    # a and b provide the location on the array as it runs
    a, b, c = cuda.grid(3)

    # checks that the location is within the array and that the concentration is larger than zero
    if a < grid_array.shape[0] and b < grid_array.shape[1] and c < grid_array.shape[2] and grid_array[a][b][c] > 0:
        grid_array[a][b][c] -= 1.0


def check_neighbors_gpu(self):
    """ the parallel form of "check_neighbors"
    """
    # arrays that will hold the cell locations and the output of edges
    location_array = np.empty((0, 3), int)
    edges_array = np.zeros((len(self.cells), len(self.cells)))

    # loops over all over the cells and puts their locations into a holder array
    for i in range(len(self.cells)):
        location_array = np.append(location_array, np.array([self.cells[i].location]), axis=0)

    # turns the arrays into a form able to send to the gpu
    location_array_cuda = cuda.to_device(location_array)
    edges_array_cuda = cuda.to_device(edges_array)
    distance_cuda = cuda.to_device(self.neighbor_distance)

    # sets up the correct allocation of threads and blocks
    threads_per_block = 32
    blocks_per_grid = math.ceil(location_array.size / threads_per_block)

    # calls the cuda function
    check_neighbors_cuda[blocks_per_grid, threads_per_block](location_array_cuda, edges_array_cuda, distance_cuda)

    # returns the array
    output = edges_array_cuda.copy_to_host()

    # turns the array into an upper triangular matrix as we don't need to double count edges
    output = np.triu(output)

    # gives pairs where there should be a connection
    edges = np.argwhere(output == 1)

    # re-adds the cells as nodes
    for i in range(len(self.cells)):
        self.network.add_node(self.cells[i])

    # forms an edge between cells that are close enough
    for i in range(len(edges)):
        self.network.add_edge(self.cells[edges[i][0]], self.cells[edges[i][1]])

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

def handle_collisions_gpu(self):
    """ the parallel form of "handle_collisions"
    """
    # the while loop controls the amount of time steps for movement
    time_counter = 0
    while time_counter <= self.move_max_time:
        # smaller the time step, less error from missing collisions
        time_counter += self.move_time_step

        # arrays that hold the cell locations, radii, and masses
        location_array = np.empty((0, 3), int)
        velocities_array = np.empty((0, 3), int)
        radius_array = np.array([])
        mass_array = np.array([])

        # arrays that hold the energy and spring values
        energy_kept = np.array([self.energy_kept])
        spring_constant = np.array([self.spring_constant])
        time_step = np.array([self.move_time_step])
        friction = np.array([self.friction])

        # loops over all over the cells and puts their locations into a holder array
        for i in range(len(self.cells)):
            location_array = np.append(location_array, np.array([self.cells[i].location]), axis=0)
            velocities_array = np.append(velocities_array, np.array([self.cells[i].velocity]), axis=0)
            radius_array = np.append(radius_array, np.array([self.cells[i].radius]), axis=0)
            mass_array = np.append(mass_array, np.array([self.cells[i].mass]), axis=0)

        # turns the arrays into a form able to send to the gpu
        location_array = cuda.to_device(location_array)
        velocities_array = cuda.to_device(velocities_array)
        radius_array = cuda.to_device(radius_array)
        mass_array = cuda.to_device(mass_array)
        energy_kept = cuda.to_device(energy_kept)
        spring_constant = cuda.to_device(spring_constant)
        time_step = cuda.to_device(time_step)
        grid_size = cuda.to_device(self.size)
        friction = cuda.to_device(friction)

        # sets up the correct allocation of threads and blocks
        threads_per_block = 32
        blocks_per_grid = math.ceil(location_array.size / threads_per_block)

        handle_collisions_cuda[blocks_per_grid, threads_per_block](location_array, radius_array,
                                                                   mass_array, energy_kept, spring_constant,
                                                                   velocities_array)

        update_locations_cuda[blocks_per_grid, threads_per_block](location_array, velocities_array, time_step,
                                                                  grid_size, friction)

        # returns the array
        new_velocities = velocities_array.copy_to_host()
        new_locations = location_array.copy_to_host()

        # updates the velocities
        for i in range(len(self.cells)):
            self.cells[i].velocity = new_velocities[i]
            self.cells[i].location = new_locations[i]

    # Determines if two cells are close enough together to designate a neighbor
    self.check_neighbors()

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

            # gets the distance between the cells
            displacement_x = locations[i][0] - locations[j][0]
            displacement_y = locations[i][1] - locations[j][1]
            displacement_z = locations[i][2] - locations[j][2]
            mag = magnitude(locations[i], locations[j])

            # if the cells are overlapping then proceed
            if mag <= radius[i] + radius[j]:

                # gets a normal vector of the vector between the centers of both cells
                if mag == 0.0:
                    normal_x = 0.0
                    normal_y = 0.0
                    normal_z = 0.0
                else:
                    normal_x = displacement_x / mag
                    normal_y = displacement_y / mag
                    normal_z = displacement_z / mag

                # find the displacement of the membrane overlap for each cell
                obj1_displacement_x = radius[i] * normal_x
                obj1_displacement_y = radius[i] * normal_y
                obj1_displacement_z = radius[i] * normal_z

                obj2_displacement_x = radius[j] * normal_x
                obj2_displacement_y = radius[j] * normal_y
                obj2_displacement_z = radius[j] * normal_z

                overlap_x = (displacement_x - (obj1_displacement_x + obj2_displacement_x)) / 2
                overlap_y = (displacement_y - (obj1_displacement_y + obj2_displacement_y)) / 2
                overlap_z = (displacement_z - (obj1_displacement_z + obj2_displacement_z)) / 2

                # converts the spring energy into kinetic energy in opposing directions
                velocities[i][0] -= overlap_x * (energy[0] * spring[0] / mass[i]) ** 0.5
                velocities[i][1] -= overlap_y * (energy[0] * spring[0] / mass[i]) ** 0.5
                velocities[i][2] -= overlap_z * (energy[0] * spring[0] / mass[i]) ** 0.5

                velocities[j][0] += overlap_x * (energy[0] * spring[0] / mass[j]) ** 0.5
                velocities[j][1] += overlap_y * (energy[0] * spring[0] / mass[j]) ** 0.5
                velocities[j][2] += overlap_z * (energy[0] * spring[0] / mass[j]) ** 0.5

@cuda.jit
def update_locations_cuda(locations, velocities, time_step, grid_size, friction):
    """ this is the function being run in parallel
    """
    # i provides the location on the array as it runs
    i = cuda.grid(1)

    # checks to see that position is in the array
    if i < locations.shape[0]:
        v = velocities[i]
        movement_x = v[0] * time_step[0]
        movement_y = v[1] * time_step[0]
        movement_z = v[2] * time_step[0]

        new_location_x = locations[i][0] + movement_x
        new_location_y = locations[i][1] + movement_y
        new_location_z = locations[i][2] + movement_z

        if new_location_x >= grid_size[0]:
            velocities[i][0] *= -0.5
            locations[i][0] = grid_size[0] - 0.001
        elif new_location_x < 0:
            velocities[i][0] *= -0.5
            locations[i][0] = 0.0
        else:
            locations[i][0] = new_location_x

        if new_location_y >= grid_size[1]:
            velocities[i][1] *= -0.5
            locations[i][1] = grid_size[1] - 0.001
        elif new_location_y < 0:
            velocities[i][1] *= -0.5
            locations[i][1] = 0.0
        else:
            locations[i][1] = new_location_y

        if new_location_z >= grid_size[2]:
            velocities[i][2] *= -0.5
            locations[i][2] = grid_size[2] - 0.001
        elif new_location_z < 0:
            velocities[i][2] *= -0.5
            locations[i][2] = 0.0
        else:
            locations[i][2] = new_location_z

        # subtracts the work from the kinetic energy and recalculates a new velocity
        velocities[i][0] = math.copysign(1.0, velocities[i][0]) * max(velocities[i][0] ** 2 - 2 * friction[0] * abs(movement_x), 0.0) ** 0.5
        velocities[i][1] = math.copysign(1.0, velocities[i][1]) * max(velocities[i][1] ** 2 - 2 * friction[0] * abs(movement_y), 0.0) ** 0.5
        velocities[i][2] = math.copysign(1.0, velocities[i][2]) * max(velocities[i][2] ** 2 - 2 * friction[0] * abs(movement_z), 0.0) ** 0.5


@cuda.jit(device=True)
def magnitude(location_one, location_two):
    displacement_x = location_one[0] - location_two[0]
    displacement_y = location_one[1] - location_two[1]
    displacement_z = location_one[2] - location_two[2]

    mag = (displacement_x ** 2 + displacement_y ** 2 + displacement_z ** 2) ** 0.5

    return mag