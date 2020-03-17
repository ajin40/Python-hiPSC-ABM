#########################################################
# Name:    Parallel                                     #
# Author:  Jack Toppen                                  #
# Date:    3/17/20                                      #
#########################################################
from numba import cuda
import math
import numpy as np



def initialize_grid_gpu(self):
    array = cuda.to_device(self.grid)

    threads_per_block = (32, 32)
    blocks_per_grid_x = math.ceil(self.grid.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(self.grid.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    initialize_grid_cuda[blocks_per_grid, threads_per_block](array)

    self.grid = array.copy_to_host()


def update_grid_gpu(self):
    array = cuda.to_device(self.grid)

    threads_per_block = (32, 32)
    blocks_per_grid_x = math.ceil(self.grid.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(self.grid.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    initialize_grid_cuda[blocks_per_grid, threads_per_block](array)

    self.grid = array.copy_to_host()


@cuda.jit
def initialize_grid_cuda(grid_array):
    a, b = cuda.grid(2)
    if a < grid_array.shape[0] and b < grid_array.shape[1]:
        grid_array[a, b] += 10


@cuda.jit
def update_grid_cuda(grid_array):
    a, b = cuda.grid(2)
    if a < grid_array.shape[0] and b < grid_array.shape[1] and grid_array[a, b] >= 1:
        grid_array[a, b] -= 1


def check_neighbors_gpu(self):

    location_array = np.empty((0, 2), int)
    edges_array = np.zeros((len(self.objects), len(self.objects)))

    for i in range(len(self.objects)):
        location_array = np.append(location_array, np.array([self.objects[i].location]), axis=0)

    location_array_device_in = cuda.to_device(location_array)
    edges_array_device_in = cuda.to_device(edges_array)

    threadsperblock = 32
    blockspergrid = math.ceil(location_array.size / threadsperblock)

    check_neighbors_cuda[blockspergrid, threadsperblock](location_array_device_in, edges_array_device_in)

    output = edges_array_device_in.copy_to_host()
    output = np.triu(output)
    edges = np.argwhere(output == 1)

    for i in range(len(self.objects)):
        self.network.add_node(self.objects[i])

    for i in range(len(edges)):
        self.network.add_edge(self.objects[edges[i][0]], self.objects[edges[i][1]])

@cuda.jit
def check_neighbors_cuda(locations, edges):
    pos = cuda.grid(1)
    if pos < len(locations):
        for i in range(len(locations)):
            if ((locations[pos][0] - locations[i][0]) ** 2 + (locations[pos][1] - locations[i][1]) ** 2) ** 0.5 <= 24:
                if pos != i:
                    edges[pos][i] = 1

def handle_collisions_gpu(self):
    time_counter = 0
    while time_counter <= self.move_max_time:
        time_counter += self.move_time_step

        location_array = np.empty((0, 2), int)
        nuclear_array = np.array([])
        cytoplasm_array = np.array([])
        mass_array = np.array([])

        energy_kept = np.array([self.energy_kept])
        spring_constant = np.array([self.spring_constant])


        velocities_array = np.zeros((len(self.objects), 2))

        for i in range(len(self.objects)):
            location_array = np.append(location_array, np.array([self.objects[i].location]), axis=0)
            nuclear_array = np.append(nuclear_array, np.array([self.objects[i].nuclear_radius]), axis=0)
            cytoplasm_array = np.append(cytoplasm_array, np.array([self.objects[i].cytoplasm_radius]), axis=0)
            mass_array = np.append(mass_array, np.array([self.objects[i].mass]), axis=0)


        location_array = cuda.to_device(location_array)
        nuclear_array = cuda.to_device(nuclear_array)
        cytoplasm_array = cuda.to_device(cytoplasm_array)
        mass_array = cuda.to_device(mass_array)

        energy_kept = cuda.to_device(energy_kept)
        spring_constant = cuda.to_device(spring_constant)

        velocities_array = cuda.to_device(velocities_array)

        threadsperblock = 32
        blockspergrid = (location_array.size + (threadsperblock - 1)) // threadsperblock
        handle_collisions_cuda[blockspergrid, threadsperblock](location_array, nuclear_array, cytoplasm_array, mass_array, energy_kept, spring_constant, velocities_array)

        output = velocities_array.copy_to_host()

        for i in range(len(self.objects)):
            velocity = output[i]


            movement = velocity * self.move_time_step
            self.objects[i].disp_vec += movement

            new_velocity = np.array([0.0, 0.0])

            new_velocity[0] = np.sign(velocity[0]) * max((velocity[0]) ** 2 - 2 * self.friction * abs(movement[0]),
                                                         0.0) ** 0.5
            new_velocity[1] = np.sign(velocity[0]) * max((velocity[1]) ** 2 - 2 * self.friction * abs(movement[1]),
                                                         0.0) ** 0.5

            self.objects[i].velocity = new_velocity


            self.objects[i].location += self.objects[i].disp_vec

            if not 0 <= self.objects[i].location[0] < 1000:
                self.objects[i].location[0] -= 2 * self.objects[i].disp_vec[0]

            if not 0 <= self.objects[i].location[1] < 1000:
                self.objects[i].location[1] -= 2 * self.objects[i].disp_vec[1]

            # resets the movement vector to [0,0]
            self.objects[i].disp_vec = np.array([0.0, 0.0])



@cuda.jit
def handle_collisions_cuda(locations, nuclear, cytoplasm, mass, energy, spring, velocities):
    i = cuda.grid(1)
    if i < len(locations):
        for j in range(len(locations)):
            displacement_x = locations[i][0] - locations[j][0]
            displacement_y = locations[i][1] - locations[j][1]

            mag = (displacement_x ** 2 + displacement_y ** 2) ** 0.5

            if mag <= nuclear[i] + nuclear[j] + cytoplasm[i] + cytoplasm[j]:
                normal_x = displacement_x / mag
                normal_y = displacement_y / mag


                obj1_displacement_x = (nuclear[i] + cytoplasm[i]) * normal_x
                obj1_displacement_y = (nuclear[i] + cytoplasm[i]) * normal_y

                obj2_displacement_x = (nuclear[j] + cytoplasm[j]) * normal_x
                obj2_displacement_y = (nuclear[j] + cytoplasm[j]) * normal_y

                real_displacement_x = (displacement_x - (obj1_displacement_x + obj2_displacement_x)) / 2
                real_displacement_y = (displacement_y - (obj1_displacement_y + obj2_displacement_y)) / 2

                velocities[i][0] -= real_displacement_x * (energy[0] * spring[0] / mass[i]) ** 0.5
                velocities[i][1] -= real_displacement_y * (energy[0] * spring[0] / mass[i]) ** 0.5

                velocities[j][0] += real_displacement_x * (energy[0] * spring[0] / mass[j]) ** 0.5
                velocities[j][1] += real_displacement_y * (energy[0] * spring[0] / mass[j]) ** 0.5
