import numpy as np
import networkx as nx

import Parallel

class Simulation:
    """ Initialization called once for each simulation. Class holds all information about each simulation as a whole
    """

    def __init__(self, path, end_time, time_step, pluri_div_thresh, diff_div_thresh, pluri_to_diff, size,
                 diff_surround_value, functions, parallel, death_threshold, move_time_step, move_max_time,
                 spring_constant, friction, energy_kept, neighbor_distance, density, num_states, quality,
                 group, speed, max_radius, slices, adhesion_const):

        """ Initialization function for the simulation setup.
            path: the path to save the simulation information to
            end_time: the end time for the simulation
            time_step: the time step to increment the simulation by
            pluri_div_thresh: threshold for pluripotent cells to divide
            diff_div_thresh:  threshold for differentiated cells to divide
            pluri_to_diff: threshold for pluripotent cells to differentiate
            size: the size of the grid (dimension, rows, columns)
            diff_surround_value: the amount of differentiated cells needed to surround
                a pluripotent cell inducing its differentiation
            functions: the finite dynamical system functions as a string from Model_Setup
            parallel: true / false which determines whether some tasks are run on the GPU
            death_threshold: the value at which a cell dies
            move_time_step: the time value in which the cells are moved incrementally
            move_max_time: the max time for movement allow enough time for cells to reach equilibrium
            spring_constant: spring constant for modeling interactions between cells with spring energy
            friction: friction constant for modeling loss of energy
            energy_kept: percent of energy (as a decimal) left after turning spring energy into kinetic
            neighbor_distance: how close cells need to be in order to be considered 'neighbors'
            density: the density of a cell
            num_states: the number of states for the finite dynamical system (positive integer).
                Currently 2 because the system is a Boolean network
            quality: the 'quality" of the images as pixel dimensions times 1500
            group: how many cells are removed or added at once per time step
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
        self.num_states = num_states
        self.quality = quality
        self.group = group
        self.speed = speed
        self.max_radius = max_radius
        self.slices = slices
        self.adhesion_const = adhesion_const

        # counts how many times an image is created for making videos
        self.image_counter = 0

        # keeps a running count of the time
        self.time_counter = 0.0

        # array to hold all of the Cell objects
        self.cells = np.array([], dtype=np.object)

        # array to hold all of the Extracellular objects
        self.extracellular = np.array([], dtype=np.object)

        # graph representing cells and neighbors
        self.neighbor_graph = nx.Graph()

        # graph representing the presence of JKR adhesion bonds between cells
        self.jkr_graph = nx.Graph()

        # holds the objects until they are added or removed from the simulation
        self.cells_to_remove = np.array([], dtype=np.object)
        self.cells_to_add = np.array([], dtype=np.object)


    def info(self):
        """ prints information about the simulation as it
            runs. May include more information later
        """
        print("Time: " + str(self.time_counter))
        print("Number of objects: " + str(len(self.cells)))

    def initialize_diffusion(self):
        """ see Cell.py for definition
        """
        for i in range(len(self.extracellular)):
            self.extracellular[i].initialize()

    def update_diffusion(self):
        """ see Cell.py for definition
        """
        for i in range(len(self.extracellular)):
            self.extracellular[i].update(self)

    def update_cells(self):
        """ see Cell.py for definition
        """
        for i in range(len(self.cells)):
            self.cells[i].update_cell(self)

    def kill_cells(self):
        """ kills the cells that are alone for too long
        """
        for i in range(len(self.cells)):
            self.cells[i].kill_cell(self)

    def diff_surround_cells(self):
        """ see Cell.py for definition
        """
        for i in range(len(self.cells)):
            self.cells[i].diff_surround(self)

    def change_size_cells(self):
        """ see Cell.py for definition
        """
        for i in range(len(self.cells)):
            self.cells[i].change_size(self)

    def randomly_move_cells(self):
        """ see Cell.py for definition
        """
        for i in range(len(self.cells)):
            self.cells[i].randomly_move(self)

    def add_cell(self, cell):
        """ Adds the specified object to the array
            and the neighbor graph
        """
        # adds it to the array
        self.cells = np.append(self.cells, cell)

        # adds it to the graph
        self.neighbor_graph.add_node(cell)

    def remove_cell(self, cell):
        """ Removes the specified object from the array
            and the neighbor graph
        """
        # removes it from the array
        self.cells = self.cells[self.cells != cell]

        # removes it from the graph
        self.neighbor_graph.remove_node(cell)

    def update_cell_queue(self):
        """ Updates the queues for adding and removing cell objects
        """
        print("Adding " + str(len(self.cells_to_add)) + " cell objects...")
        print("Removing " + str(len(self.cells_to_remove)) + " cell objects...")

        # loops over all objects to remove
        for i in range(len(self.cells_to_remove)):
            self.remove_cell(self.cells_to_remove[i])

            # Cannot add all of the new cell objects, otherwise several cells are likely to be added
            #   in close proximity to each other at later time steps. Such object addition, coupled
            #   with handling collisions, make give rise to sudden changes in overall positions of
            #   cells within the simulation. Instead, collisions are handled after 'group' number
            #   of cell objects are added.
            # if (i + 1) % self.group == 0:
            #     self.handle_forces()

        # loops over all objects to add
        for i in range(len(self.cells_to_add)):
            self.add_cell(self.cells_to_add[i])

            # can't add all the cells together or you get a mess
            # if (i + 1) % self.group == 0:
            #     self.handle_forces()

        # clear the arrays
        self.cells_to_remove = np.array([], dtype=np.object)
        self.cells_to_add = np.array([], dtype=np.object)

    def check_neighbors(self):
        """ checks all of the distances between cells
            if it is less than a fixed value create a
            connection between two cells.
        """
        # clears the current graph to prevent existing edges from remaining
        self.neighbor_graph.clear()

        # tries to run the parallel version of this function
        if self.parallel:
            Parallel.check_neighbors_gpu(self)
        else:
            # divides the environment into blocks
            distance = self.neighbor_distance
            x = int(self.size[0] / distance + 3)
            y = int(self.size[1] / distance + 3)
            z = int(self.size[2] / distance + 3)
            blocks = np.empty((x, y, z), dtype=object)

            # gives each block an array as a cell holder
            for i in range(x):
                for j in range(y):
                    for k in range(z):
                        blocks[i][j][k] = np.array([])

            # assigns each cell to a block by rounding its coordinates up to the nearest integer
            # loops over all cells and gets block location
            for h in range(len(self.cells)):

                # adds all of the cells to the simulation
                self.neighbor_graph.add_node(self.cells[h])

                # offset blocks by 1 to help when searching over blocks
                location_x = int(self.cells[h].location[0] / distance) + 1
                location_y = int(self.cells[h].location[1] / distance) + 1
                location_z = int(self.cells[h].location[2] / distance) + 1

                # adds the cell to a given block
                current_block = blocks[location_x][location_y][location_z]
                blocks[location_x][location_y][location_z] = np.append(current_block, self.cells[h])

                # looks at the blocks surrounding a given block that houses the cell
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        for k in range(-1, 2):
                            cells_in_block = blocks[location_x + i][location_y + j][location_z + k]

                            # looks at the cells in a block and decides if they are neighbors
                            for l in range(len(cells_in_block)):
                                if cells_in_block[l] != self.cells[h]:
                                    if np.linalg.norm(cells_in_block[l].location - self.cells[h].location) <= distance:
                                        self.neighbor_graph.add_edge(self.cells[h], cells_in_block[l])

    def adhesion_and_repulsion(self):
        """ goes through all of the cells and applies any forces arising
            from adhesion or repulsion between the cells
        """
        # list of the neighbors as these will only be the cells in physical contact
        edges = list(self.neighbor_graph.edges())

        # loops over the pairs of neighbors
        for i in range(len(edges)):
            # assigns the nodes of each edge to a variable
            cell_1 = edges[i][0]
            cell_2 = edges[i][1]

            # hold the vector between the centers of the cells and the magnitude of this vector
            disp_vector = cell_1.location - cell_2.location
            magnitude = np.linalg.norm(disp_vector)

            # get the total overlap of the cells used later in calculations
            overlap = cell_1.radius + cell_2.radius - magnitude

            # gets two values used for the Hertzian contact and JKR adhesion
            e_hat = (((1 - cell_1.poisson ** 2) / cell_1.youngs_mod) + (
                        (1 - cell_2.poisson ** 2) / cell_2.youngs_mod)) ** -1
            r_hat = ((1 / cell_1.radius) + (1 / cell_2.radius)) ** -1

            # used to calculate the max adhesive distance after an adhesion has been already formed
            overlap_ = (((3.14159 * self.adhesion_const) / e_hat) ** 2 / 3) * (r_hat ** 1 / 3)

            # used to see if the adhesive bond once formed has broken
            overlap_condition = overlap / overlap_ > -0.360562

            if overlap >= 0 or (overlap_condition and self.jkr_graph.has_edge(cell_1, cell_2)):
                # JKR adhesion
                # we nondimensionalize the overlap to allow for the use of a polynomial approximation
                d = overlap / ((((3.14159 * self.adhesion_const) / e_hat) ** 2 / 3) * (r_hat ** 1 / 3))

                # plug the value of d into the nondimensionalized equation for adhesion force
                f = (-0.0204 * d ** 3) + (0.4942 * d ** 2) + (1.0801 * d) - 1.324

                # convert from the nondimensionalization to find the adhesive force
                adhesive = f * 3.14159 * self.adhesion_const * r_hat

                # adds the adhesive force as a vector in opposite directions to each cell's force holder
                cell_1.force += adhesive * disp_vector
                cell_2.force -= adhesive * disp_vector

                if overlap >= 0:
                    # if the cells touch or overlap there is an adhesive bond formed
                    self.jkr_graph.add_edge(cell_1, cell_2)

                    # Hertzian contact for repulsion
                    # finds the repulsive force scalar
                    repulsive = (4 / 3) * e_hat * (r_hat ** 0.5) * (overlap ** 1.5)

                    # adds the repulsive force as a vector in opposite directions to each cell's force holder
                    cell_1.force += repulsive * disp_vector
                    cell_2.force -= repulsive * disp_vector

            elif not overlap_condition:
                self.jkr_graph.remove_edge(cell_1, cell_2)