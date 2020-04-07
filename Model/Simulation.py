import numpy as np
import networkx as nx

from Model import Parallel


class Simulation:
    """ called once holds important information about the
        simulation
    """

    def __init__(self, path, end_time, time_step, pluri_div_thresh, diff_div_thresh, pluri_to_diff, size,
                 diff_surround_value, functions, parallel, death_threshold, move_time_step, move_max_time,
                 spring_constant, friction, energy_kept, neighbor_distance, density, n, quality,
                 group, speed):

        """ Initialization function for the simulation setup.
            name: the simulation name
            path: the path to save the simulation information to
            start_time: the start time for the simulation
            end_time: the end time for the simulation
            time_step: the time step to increment the simulation by
            pluri_div_thresh: threshold for pluripotent cells to divide
            diff_div_thresh:  threshold for differentiated cells to divide
            pluri_to_diff: threshold for pluripotent cells to differentiate
            size: the size of the grid (dimension, rows, columns)
            spring_max: represents the max distance for spring and other interactions
            diff_surround_value: the amount of differentiated cells needed to surround
                a pluripotent cell inducing its differentiation
            functions: the boolean functions as a string from Model_Setup
            parallel: whether some aspects are run on the gpu
            spring_constant: strength of spring
            friction: resistance to moving in the environment
            energy_kept: percent of energy left after turning spring energy into kinetic
            neighbor_distance: how close cells are to be neighbors
            density: the density of a cell
            n: the number of boolean states (0, 1, 2,...)
            quality: the quality of the images in pixel dimensions times 1500
            group: how many cells are removed or added at a time
            speed: magnitude of random movement speed
        """
        self.path = path
        self.end_time = end_time
        self.time_step = time_step
        self.pluri_div_thresh = pluri_div_thresh
        self.diff_div_thresh = diff_div_thresh
        self.pluri_to_diff = pluri_to_diff
        self.size = size
        self.diff_surround_value = diff_surround_value
        self.functions = functions
        self.parallel = parallel
        self.death_threshold = death_threshold
        self.move_time_step = move_time_step
        self.move_max_time = move_max_time
        self.spring_constant = spring_constant
        self.friction = friction
        self.energy_kept = energy_kept
        self.neighbor_distance = neighbor_distance
        self.density = density
        self.n = n
        self.quality = quality
        self.group = group
        self.speed = speed

        # counts how many times an image is created for making videos
        self.image_counter = 0

        # keeps a running count of the time
        self.time_counter = 0.0

        # array to hold all of the cell objects
        self.cells = np.array([], dtype=np.object)

        # array to hold all of the gradient objects
        self.gradients = np.array([], dtype=np.object)

        # graph representing all the cells and their connections to other cells
        self.network = nx.Graph()

        # holds the objects until they are added or removed from the simulation
        self.cells_to_remove = np.array([], dtype=np.object)
        self.cells_to_add = np.array([], dtype=np.object)


    def info(self):
        """ prints information about the simulation as it
            runs. May include more information later
        """
        print("Time: " + str(self.time_counter))
        print("Number of objects: " + str(len(self.cells)))

    def initialize_gradients(self):
        """ adds initial concentrations to each gradient grid
        """
        for i in range(len(self.gradients)):
            self.gradients[i].initialize_grid()

    def update_gradients(self):
        """ currently degrades the concentrations of molecules
            in the grids
        """
        for i in range(len(self.gradients)):
            self.gradients[i].update_grid()

    def update_cells(self):
        """ updates each cell by allowing them to divide
            and differentiate among other things
        """
        for i in range(len(self.cells)):
            self.cells[i].update_cell(self)

    def kill_cells(self):
        """ kills the cells that are alone for too long
        """
        for i in range(len(self.cells)):
            self.cells[i].kill_cell(self)

    def diff_surround_cells(self):
        """ increases the differentiation counter if enough
            differentiated cells surround a pluripotent cell
        """
        for i in range(len(self.cells)):
            self.cells[i].diff_surround(self)

    def change_size_cells(self):
        """ updates the cell's radius and mass based on
            the division counter
        """
        for i in range(len(self.cells)):
            self.cells[i].change_size(self)

    def randomly_move_cells(self):
        """ has the objects that are in motion
            move in a random way
        """
        for i in range(len(self.cells)):
            self.cells[i].randomly_move(self)

    def add_cell(self, cell):
        """ Adds the specified object to the array
            and the graph
        """
        # adds it to the array
        self.cells = np.append(self.cells, cell)

        # adds it to the graph
        self.network.add_node(cell)

    def remove_cell(self, cell):
        """ Removes the specified object from the array
            and the graph
        """
        # removes it from the array
        self.cells = self.cells[self.cells != cell]

        # removes it from the graph
        self.network.remove_node(cell)

    def update_cell_queue(self):
        """ Updates the object add and remove queue
        """
        print("Adding " + str(len(self.cells_to_add)) + " objects...")
        print("Removing " + str(len(self.cells_to_remove)) + " objects...")

        # loops over all objects to remove
        for i in range(len(self.cells_to_remove)):
            self.remove_cell(self.cells_to_remove[i])

            # can't add all the cells together or you get a mess

            if (i + 1) % self.group == 0:

                self.handle_collisions()

        # loops over all objects to add
        for i in range(len(self.cells_to_add)):
            self.add_cell(self.cells_to_add[i])

            # can't add all the cells together or you get a mess
            if (i + 1) % self.group == 0:
                self.handle_collisions()

        # clear the arrays
        self.cells_to_remove = np.array([], dtype=np.object)
        self.cells_to_add = np.array([], dtype=np.object)

    # old version about 50-100X slower
    # def check_neighbors(self):
    #     """ checks all of the distances between cells
    #         if it is less than a set value create a
    #         connection between two cells.
    #     """
    #     # clears the current graph to prevent existing edges remaining
    #     self.network.clear()
    #
    #     # tries to run the parallel version of the function
    #     if self.parallel:
    #         Parallel.check_neighbors_gpu(self)
    #     else:
    #         # loops over all objects
    #         for i in range(len(self.cells)):
    #
    #             # adds all of the cells to the simulation
    #             self.network.add_node(self.cells[i])
    #
    #             # loops over all objects not check already
    #             for j in range(i + 1, len(self.cells)):
    #
    #                 # get the distance between cells
    #                 dist_vec = self.cells[i].location - self.cells[j].location
    #
    #                 # get the magnitude of the distance vector
    #                 dist = np.linalg.norm(dist_vec)
    #
    #                 # if the cells are close enough, add an edge between them
    #                 if dist <= self.neighbor_distance:
    #                     self.network.add_edge(self.cells[i], self.cells[j])


    def check_neighbors(self):
        """ checks all of the distances between cells
            if it is less than a set value create a
            connection between two cells.
        """
        # clears the current graph to prevent existing edges remaining
        self.network.clear()

        # tries to run the parallel version of the function
        if self.parallel:
            Parallel.check_neighbors_gpu(self)
        else:
            # divides the grid into a series of blocks
            distance = self.neighbor_distance
            x = int(self.size[0] / distance + 3)
            y = int(self.size[1] / distance + 3)
            z = int(self.size[2] / distance + 3)
            blocks = np.empty((x, y, z), dtype=object)

            # gives each block a array as a cell holder
            for i in range(x):
                for j in range(y):
                    for k in range(z):
                        blocks[i][j][k] = np.array([])

            # assigns each cell to a block
            for i in range(len(self.cells)):
                # adds all of the cells to the simulation
                self.network.add_node(self.cells[i])

                # offset blocks by 1 to help when searching over blocks
                location_x = int(self.cells[i].location[0] / distance) + 1
                location_y = int(self.cells[i].location[1] / distance) + 1
                location_z = int(self.cells[i].location[2] / distance) + 1

                # adds the cell to a given block
                current_block = blocks[location_x][location_y][location_z]
                blocks[location_x][location_y][location_z] = np.append(current_block, self.cells[i])

            # loops over all cells and gets block location
            for h in range(len(self.cells)):
                location_x = int(self.cells[h].location[0] / distance) + 1
                location_y = int(self.cells[h].location[1] / distance) + 1
                location_z = int(self.cells[h].location[2] / distance) + 1

                # looks at the blocks surrounding a given block that houses the cell
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        for k in range(-1, 2):
                            cells_in_block = blocks[location_x + i][location_y + j][location_z + k]

                            # looks at the cells in a block and decides if they are neighbors
                            for l in range(len(cells_in_block)):
                                if cells_in_block[l] != self.cells[h]:
                                    if np.linalg.norm(cells_in_block[l].location - self.cells[h].location) <= distance:
                                        self.network.add_edge(self.cells[h], cells_in_block[l])

    def handle_collisions(self):
        """ Move the cells in small increments and manages
            any collisions that will arise
        """
        # tries to run the parallel version of the function
        if self.parallel:
            Parallel.handle_collisions_gpu(self)

        else:
            self.check_neighbors()

            # the while loop controls the amount of time steps for movement
            time_counter = 0
            while time_counter <= self.move_max_time:

                # smaller the time step, less error from missing collisions
                time_counter += self.move_time_step

                # gets all of the neighbor connections
                edges = list(self.network.edges())

                # loops over the connections as these cells are close together
                for i in range(len(edges)):
                    cell_1 = edges[i][0]
                    cell_2 = edges[i][1]

                    # vector between the center of each cell for the edge
                    displacement_vec = cell_1.location - cell_2.location

                    # addition of total cell radius
                    cell_1_total_radius = cell_1.radius
                    cell_2_total_radius = cell_2.radius
                    total_radii = cell_1_total_radius + cell_2_total_radius

                    # checks to see if the cells are overlapping
                    if np.linalg.norm(displacement_vec) < total_radii:

                        # find the displacement of the membrane overlap for each cell
                        mag = np.linalg.norm(displacement_vec)
                        if mag == 0.0:
                            displacement_normal = np.array([0.0, 0.0, 0.0])
                        else:
                            displacement_normal = displacement_vec / mag

                        overlap = (displacement_vec - (total_radii * displacement_normal)) / 2

                        # converts the spring energy into kinetic energy in opposing directions
                        cell_1.velocity[0] -= overlap[0] * (self.energy_kept * self.spring_constant / cell_1.mass)**0.5
                        cell_1.velocity[1] -= overlap[1] * (self.energy_kept * self.spring_constant / cell_1.mass)**0.5
                        cell_1.velocity[2] -= overlap[2] * (self.energy_kept * self.spring_constant / cell_1.mass)**0.5

                        cell_2.velocity[0] += overlap[0] * (self.energy_kept * self.spring_constant / cell_2.mass)**0.5
                        cell_2.velocity[1] += overlap[1] * (self.energy_kept * self.spring_constant / cell_2.mass)**0.5
                        cell_2.velocity[2] += overlap[2] * (self.energy_kept * self.spring_constant / cell_2.mass)**0.5

                for i in range(len(self.cells)):

                    # multiplies the time step by the velocity and adds that vector to the cell's holder
                    v = self.cells[i].velocity

                    movement = v * self.move_time_step
                    location = self.cells[i].location

                    new_location = location + movement

                    if new_location[0] >= self.size[0]:
                        self.cells[i].velocity[0] *= -0.5
                        self.cells[i].location[0] = self.size[0] - 0.001
                    elif new_location[0] < 0:
                        self.cells[i].velocity[0] *= -0.5
                        self.cells[i].location[0] = 0.0
                    else:
                        self.cells[i].location[0] = new_location[0]

                    if new_location[1] >= self.size[1]:
                        self.cells[i].velocity[1] *= -0.5
                        self.cells[i].location[1] = self.size[1] - 0.001
                    elif new_location[1] < 0:
                        self.cells[i].velocity[1] *= -0.5
                        self.cells[i].location[1] = 0.0
                    else:
                        self.cells[i].location[1] = new_location[1]

                    if new_location[2] >= self.size[2]:
                        self.cells[i].velocity[2] *= -0.5
                        self.cells[i].location[2] = self.size[2] - 0.001
                    elif new_location[2] < 0:
                        self.cells[i].velocity[2] *= -0.5
                        self.cells[i].location[2] = 0.0
                    else:
                        self.cells[i].location[2] = new_location[2]

                    # subtracts the work from the kinetic energy and recalculates a new velocity
                    new_velocity_x = np.sign(v[0]) * max(v[0] ** 2 - 2 * self.friction * abs(movement[0]), 0.0) ** 0.5
                    new_velocity_y = np.sign(v[1]) * max(v[1] ** 2 - 2 * self.friction * abs(movement[1]), 0.0) ** 0.5
                    new_velocity_z = np.sign(v[2]) * max(v[2] ** 2 - 2 * self.friction * abs(movement[2]), 0.0) ** 0.5

                    # assign new velocity
                    self.cells[i].velocity = np.array([new_velocity_x, new_velocity_y, new_velocity_z])

                # checks neighbors after the cells move for re-evaluation of collisions
                self.check_neighbors()