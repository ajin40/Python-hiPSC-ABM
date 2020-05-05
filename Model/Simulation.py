import numpy as np
import networkx as nx
import math

import Parallel

class Simulation:
    """ Initialization called once for each simulation. Class holds all information about each simulation as a whole
    """

    def __init__(self, path, parallel, size, resolution, num_states, functions, neighbor_distance, time_step_value,
                 end_time, move_time_step, pluri_div_thresh, pluri_to_diff, diff_div_thresh, boolean_thresh,
                 diff_surround, death_thresh, adhesion_const, viscosity, group, slices, image_quality, background_color,
                 bound_color, pluri_gata6_high_color, pluri_nanog_high_color, pluri_both_high_color, diff_color,
                 lonely_cell, contact_inhibit, guye_move, guye_distance, motility_force):

        """ path: the path to save the simulation information to
            parallel: true / false which determines whether some tasks are run on the GPU
            size: the size of the space (x, y, z)
            resolution: the spatial resolution of the space
            num_states: the number of states for the finite dynamical system (positive integer).
                Currently 2 because the system is a Boolean network
            functions: the finite dynamical system functions as a string from the template file
            neighbor_distance: how close cells need to be in order to be considered 'neighbors'
            time_step: the time step to increment the simulation by
            end_time: the end time for the simulation
            move_time_step: the time value in which the cells are moved incrementally
            pluri_div_thresh: threshold for pluripotent cells to divide
            pluri_to_diff: threshold for pluripotent cells to differentiate
            diff_div_thresh:  threshold for differentiated cells to divide
            diff_surround: the amount of differentiated cells needed to surround
                a pluripotent cell inducing its differentiation
            death_thresh: the value at which a cell dies
            adhesion_const: JKR work of adhesion
            group: how many cells are removed or added at once per time step
            slices: the amount of slices taken in the z direction
            image_quality: the dimensions of the output images in pixels
            background_color: the color of the image background
            bound_color: the colors of the bounding lines of the image
            pluri_cell_color: the color of pluripotent cells
            diff_cell_color: the color of differentiated cells
        """
        self.path = path
        self.parallel = parallel
        self.size = size
        self.resolution = resolution
        self.num_states = num_states
        self.functions = functions
        self.neighbor_distance = neighbor_distance
        self.time_step_value = time_step_value
        self.end_time = end_time
        self.move_time_step = move_time_step
        self.pluri_div_thresh = pluri_div_thresh
        self.pluri_to_diff = pluri_to_diff
        self.diff_div_thresh = diff_div_thresh
        self.boolean_thresh = boolean_thresh
        self.diff_surround = diff_surround
        self.death_thresh = death_thresh
        self.adhesion_const = adhesion_const
        self.viscosity = viscosity
        self.group = group
        self.slices = slices
        self.image_quality = image_quality
        self.background_color = background_color
        self.bound_color = bound_color
        self.pluri_gata6_high_color = pluri_gata6_high_color
        self.pluri_nanog_high_color = pluri_nanog_high_color
        self.pluri_both_high_color = pluri_both_high_color
        self.diff_color = diff_color
        self.lonely_cell = lonely_cell
        self.contact_inhibit = contact_inhibit
        self.guye_move = guye_move
        self.guye_distance = guye_distance
        self.motility_force = motility_force

        # counts how many times an image is created for making videos
        self.image_counter = 0

        # keeps a running count of the simulation steps
        self.steps_counter = 0

        # array to hold all of the Cell objects
        self.cells = np.array([], dtype=np.object)

        # array to hold all of the Extracellular objects
        self.extracellular = np.array([], dtype=np.object)

        # graph representing cells and neighbors
        self.neighbor_graph = nx.Graph()

        # graph representing the presence of JKR adhesion bonds between cells
        self.jkr_graph = nx.Graph()

        # graph used to locate nearest differentiated neighbors
        self.diff_graph = nx.Graph()

        # holds the objects until they are added or removed from the simulation
        self.cells_to_remove = np.array([], dtype=np.object)
        self.cells_to_add = np.array([], dtype=np.object)

        # holds all of the differentiated cells
        self.diff_cells = np.array([], dtype=np.object)

    def info(self):
        """ prints information about the simulation as it
            runs. May include more information later
        """
        print("Time: " + str(self.steps_counter))
        print("Number of objects: " + str(len(self.cells)))

    def initialize_diffusion(self):
        """ see Extracellular.py for description
        """
        for i in range(len(self.extracellular)):
            self.extracellular[i].initialize()

    def update_diffusion(self):
        """ see Extracellular.py for description
        """
        for i in range(len(self.extracellular)):
            self.extracellular[i].update(self)

    def update_cells(self):
        """ see Cell.py for description
        """
        for i in range(len(self.cells)):
            self.cells[i].update_cell(self)

    def kill_cells(self):
        """ see Cell.py for description
        """
        for i in range(len(self.cells)):
            self.cells[i].kill_cell(self)

    def diff_surround_cells(self):
        """ see Cell.py for description
        """
        for i in range(len(self.cells)):
            self.cells[i].diff_surround(self)

    def motility_cells(self):
        """ see Cell.py for description
        """
        for i in range(len(self.cells)):
            self.cells[i].motility(self)

    def add_cell(self, cell):
        """ Adds the cell to both the neighbor graph
            and the JKR adhesion graph
        """
        # adds it to the array holding the cell objects
        self.cells = np.append(self.cells, cell)

        # adds it to the neighbor graph
        self.neighbor_graph.add_node(cell)

        # adds it to the adhesion graph
        self.jkr_graph.add_node(cell)

        self.diff_graph.add_node(cell)

    def remove_cell(self, cell):
        """ Adds the cell to both the neighbor graph
            and the JKR adhesion graph
        """
        # removes it from the array holding the cell objects
        self.cells = self.cells[self.cells != cell]

        # removes it from the neighbor graph
        self.neighbor_graph.remove_node(cell)

        # removes it from the adhesion graph
        self.jkr_graph.remove_node(cell)

        self.diff_graph.remove_node(cell)

    def get_differentiated(self):
        """ Finds all of the differentiated cells
        """
        for i in range(len(self.cells)):
            if self.cells[i].state == "Differentiated":
                self.diff_cells = np.append(self.diff_cells, self.cells[i])

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
            if self.group != 0:
                if (i + 1) % self.group == 0:
                    self.handle_movement()

        # loops over all objects to add
        for i in range(len(self.cells_to_add)):
            self.add_cell(self.cells_to_add[i])

            # can't add all the cells together or you get a mess
            if self.group != 0:
                if (i + 1) % self.group == 0:
                    self.handle_movement()

        # clear the arrays
        self.cells_to_remove = np.array([], dtype=np.object)
        self.cells_to_add = np.array([], dtype=np.object)

    def check_neighbors(self):
        """ checks all of the distances between cells if it
            is less than a fixed value create a connection
            between two cells.
        """
        # removes the edges from the graph. Simply no function from networkx exists to do this
        edges = list(self.neighbor_graph.edges())
        self.neighbor_graph.remove_edges_from(edges)

        # tries to run the parallel version of this function
        if self.parallel:
            Parallel.check_neighbors_gpu(self)
        else:

            distance = self.neighbor_distance

            # divides the environment into blocks
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

    def handle_movement(self):
        """ runs the following functions together for a
            given time amount. Resets the force and
            velocity arrays as well.
        """
        # holds the current value of the time until it surpasses the simulation time step value
        time_holder = 0.0

        # loops over the following movement functions until time is surpassed
        while time_holder < self.time_step_value:
            # recheck for new neighbors
            self.check_neighbors()

            # increases the time based on the desired time step
            time_holder += self.move_time_step

            # calculate the forces acting on each cell
            self.force_to_movement()

            # reset all cell velocities and forces to zero
            for i in range(len(self.cells)):
                self.cells[i].velocity = np.array([0.0, 0.0, 0.0])
                self.cells[i].force = np.array([0.0, 0.0, 0.0])

    def force_to_movement(self):
        """ goes through all of the cells and quantifies any forces arising
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
            if magnitude == 0:
                normal = np.array([0.0, 0.0, 0.0])
            else:
                normal = disp_vector / np.linalg.norm(disp_vector)

            # get the total overlap of the cells used later in calculations
            overlap = cell_1.radius + cell_2.radius - magnitude

            # indicate that an adhesive bond has formed between the cells
            if overlap >= 0:
                self.jkr_graph.add_edge(cell_1, cell_2)

            # gets two values used for JKR
            e_hat = (((1 - cell_1.poisson ** 2) / cell_1.youngs_mod) + (
                        (1 - cell_2.poisson ** 2) / cell_2.youngs_mod)) ** -1
            r_hat = ((1 / cell_1.radius) + (1 / cell_2.radius)) ** -1

            # used to calculate the max adhesive distance after bond has been already formed
            overlap_ = (((math.pi * self.adhesion_const) / e_hat) ** (2 / 3)) * (r_hat ** (1 / 3))

            # get the nondimensionalized overlap, used for later calculations and checks
            # also for the use of a polynomial approximation of the force
            d = overlap / overlap_

            # used to see if the adhesive bond once formed has broken
            overlap_condition = d > -0.360562

            # bond condition use to see if bond exists
            bond_condition = self.jkr_graph.has_edge(cell_1, cell_2)

            # check to see if the cells will have a force interaction
            if overlap_condition and bond_condition:

                # plug the value of d into the nondimensionalized equation for the JKR force
                f = (-0.0204 * d ** 3) + (0.4942 * d ** 2) + (1.0801 * d) - 1.324

                # convert from the nondimensionalization to find the adhesive force
                jkr_force = f * math.pi * self.adhesion_const * r_hat

                # adds the adhesive force as a vector in opposite directions to each cell's force holder
                cell_1.force += jkr_force * normal
                cell_2.force -= jkr_force * normal

            # remove the edge if the it fails to meet the criteria for distance, JKR simulating that
            # the bond is broken
            elif bond_condition:
                self.jkr_graph.remove_edge(cell_1, cell_2)

        # now re-loops over cells to move them and reduce work energy from kinetic energy
        for i in range(len(self.cells)):
            # stokes law for velocity based on force and fluid viscosity
            stokes_friction = 6 * math.pi * self.viscosity * self.cells[i].radius

            # update the velocity of the cell based on the solution
            self.cells[i].velocity = self.cells[i].force / stokes_friction

            # multiplies the time step by the velocity and adds that vector to the cell's location
            # convert from seconds to hours
            movement = self.cells[i].velocity * self.move_time_step

            # create a prior location holder
            location = self.cells[i].location

            # set the possible new location
            new_location = location + movement

            # loops over all directions of space
            for j in range(0, 3):

                # check if new location is in environment space if not simulation a collision with the bounds
                if new_location[j] > self.size[j]:
                    self.cells[i].location[j] = self.size[j]
                elif new_location[j] < 0:
                    self.cells[i].location[j] = 0.0
                else:
                    self.cells[i].location[j] = new_location[j]