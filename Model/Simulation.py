import numpy as np
import networkx as nx
import math
import itertools


from operator import methodcaller
class Simulation:
    """ Initialization called once for each simulation. Class holds all information about each simulation as a whole
    """

    def __init__(self, name, path, parallel, size, resolution, num_states, functions, neighbor_distance,
                 time_step_value, beginning_step, end_step, move_time_step, pluri_div_thresh, pluri_to_diff,
                 diff_div_thresh, boolean_thresh, death_thresh, diff_surround, adhesion_const, viscosity, group,
                 slices, image_quality, background_color, bound_color, color_mode, pluri_color, diff_color,
                 pluri_gata6_high_color, pluri_nanog_high_color, pluri_both_high_color, lonely_cell, contact_inhibit,
                 guye_move, motility_force, dox_step, max_radius, division_force, move_thresh, output_images,
                 output_csvs, guye_radius, guye_force):

        """ path: the path to save the simulation information to
            parallel: true / false which determines whether some tasks are run on the GPU
            size: the size of the space (x, y, z)
            resolution: the spatial resolution of the space
            num_states: the number of states for the finite dynamical system (positive integer).
                Currently 2 because the system is a Boolean network
            functions: the finite dynamical system functions as a string from the template file
            neighbor_distance: how close cells need to be in order to be considered 'neighbors'
            time_step_value: how much actual time (in seconds) is the step worth
            beginning_step: the step the model starts at, used primarily to continue a previous simulation
            end_time: the end time for the simulation
            move_time_step: the time value in which the cells are moved incrementally
            pluri_div_thresh: threshold for pluripotent cells to divide
            pluri_to_diff: threshold for pluripotent cells to differentiate
            diff_div_thresh:  threshold for differentiated cells to divide
            boolean_thresh: threshold for updating the boolean values
            death_thresh: the value at which a cell dies
            diff_surround: the amount of differentiated cells needed to surround
                a pluripotent cell inducing its differentiation
            adhesion_const: JKR work of adhesion
            viscosity: the viscosity of the space the cells are in
            group: how many cells are removed or added at once per time step
            slices: the amount of slices taken in the z direction
            image_quality: the dimensions of the output images in pixels
            background_color: the color of the image background
            bound_color: the colors of the bounding lines of the image
            pluri_gata6_high_color: the color of a gata6 high pluripotent cell
            pluri_nanog_high_color: the color of a nanog high pluripotent cell
            pluri_both_high_color: the color of a both high pluripotent cell
            diff_color: the color of a differentiated cell
            lonely_cell: the number of cells needed for a cell not to be alone
            contact_inhibit: the number of neighbors needed to inhibit the division of a differentiated cell
            guye_move: if pluripotent gata6 high cells search for differentiated cell
            motility_force: the force a cell exerts to move
            dox_step: at what step is doxycycline is added to the simulation, inducing the gata6 pathway
            max_radius: the maximum radius that would be achieved shortly before division
            division_force: the force applied to the daughter cells when a cell divides
            move_thresh: the number of neighbors needed to inhibit motion
            output_images: whether the model will create images
            output_csvs: whether the model will create csvs
            guye_radius: the radius of search for a differentiated cell
        """
        self.name = name
        self.path = path
        self.parallel = parallel
        self.size = size
        self.resolution = resolution
        self.num_states = num_states
        self.functions = functions
        self.neighbor_distance = neighbor_distance
        self.time_step_value = time_step_value
        self.beginning_step = beginning_step
        self.end_step = end_step
        self.move_time_step = move_time_step
        self.pluri_div_thresh = pluri_div_thresh
        self.pluri_to_diff = pluri_to_diff
        self.diff_div_thresh = diff_div_thresh
        self.boolean_thresh = boolean_thresh
        self.death_thresh = death_thresh
        self.diff_surround = diff_surround
        self.adhesion_const = adhesion_const
        self.viscosity = viscosity
        self.group = group
        self.slices = slices
        self.image_quality = image_quality
        self.background_color = background_color
        self.bound_color = bound_color
        self.color_mode = color_mode
        self.pluri_color = pluri_color
        self.diff_color = diff_color
        self.pluri_gata6_high_color = pluri_gata6_high_color
        self.pluri_nanog_high_color = pluri_nanog_high_color
        self.pluri_both_high_color = pluri_both_high_color
        self.lonely_cell = lonely_cell
        self.contact_inhibit = contact_inhibit
        self.guye_move = guye_move
        self.motility_force = motility_force
        self.dox_step = dox_step
        self.max_radius = max_radius
        self.division_force = division_force
        self.move_thresh = move_thresh
        self.output_images = output_images
        self.output_csvs = output_csvs
        self.guye_radius = guye_radius
        self.guye_force = guye_force

        # counts how many times an image is created for making videos
        self.image_counter = 0

        # keeps a running count of the simulation steps
        self.current_step = self.beginning_step

        # array to hold all of the Cell objects
        self.cells = np.array([], dtype=np.object)

        # array to hold all of the Extracellular objects
        self.extracellular = np.array([], dtype=np.object)

        # graph representing cells and neighbors
        self.neighbor_graph = nx.Graph()

        # graph representing the presence of JKR adhesion bonds between cells
        self.jkr_graph = nx.Graph()

        # graph representing connections between pluripotent cells and their differentiated neighbors
        self.diff_graph = nx.Graph()

        # holds the objects until they are added or removed from the simulation
        self.cells_to_remove = np.array([], dtype=np.object)
        self.cells_to_add = np.array([], dtype=np.object)

        # holds all of the differentiated cells
        self.diff_cells = np.array([], dtype=np.object)

        # the minimum radius after division currently 2D
        self.min_radius = self.max_radius / 2 ** 0.5

        # growth rate based on min/max radius and division thresh for pluripotent cells
        self.pluri_growth = (self.max_radius - self.min_radius) / self.pluri_div_thresh

        # growth rate based on min/max radius and division thresh for pluripotent cells
        self.diff_growth = (self.max_radius - self.min_radius) / self.diff_div_thresh

        # Youngs modulus 1000 Pa
        self.youngs_mod = 1000

        # Poisson's ratio
        self.poisson = 0.5


    def info(self):
        """ prints information about the simulation as it
            runs. May include more information later
        """
        print("Step: " + str(self.current_step))
        print("Number of cells: " + str(len(self.cells)))

    def initialize_diffusion(self):
        """ see Extracellular.py for description
        """
        for i in range(len(self.extracellular)):
            self.extracellular[i].initialize_gradient()

    def update_diffusion(self):
        """ see Extracellular.py for description
        """
        for i in range(len(self.extracellular)):
            self.extracellular[i].update_gradient(self)

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
        # update the array containing the differentiated cells
        self.check_neighbors(mode="Differentiated")

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

        # adds it to the differentiated graph
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

        # removes it from the differentiated graph
        self.diff_graph.remove_node(cell)


    def update_cell_queue(self):
        """ Updates the queues for adding and removing cell objects
        """
        print("Adding " + str(len(self.cells_to_add)) + " cells...")
        print("Removing " + str(len(self.cells_to_remove)) + " cells...")

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


    def check_neighbors(self, mode="standard"):
        """ checks all of the distances between cells if it
            is less than a fixed value create a connection
            between two cells.
        """

        # tries to run the parallel version of this function
        if self.parallel:
            import Parallel
            Parallel.check_neighbors_gpu(self)
        else:
            if mode == "Differentiated":
                # removes the edges from the graph. Simply no function from networkx exists to do this
                edges = list(self.neighbor_graph.edges())
                self.diff_graph.remove_edges_from(edges)

                # distance threshold for finding nearest differentiated neighbors
                distance = self.guye_radius
            else:
                # removes the edges from the graph. Simply no function from networkx exists to do this
                edges = list(self.neighbor_graph.edges())
                self.neighbor_graph.remove_edges_from(edges)

                # distance threshold between two cells to designate a neighbor
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
                for i, j, k in itertools.product(range(-1, 2), repeat=3):
                    cells_in_block = blocks[location_x + i][location_y + j][location_z + k]

                    # looks at the cells in a block and decides if they are neighbors
                    for l in range(len(cells_in_block)):
                        if mode == "Differentiated":
                            # for the edge to be formed in the diff_graph only one cell must be differentiated
                            con_1 = cells_in_block[l].state == "Differentiated"
                            con_2 = self.cells[h].state == "Differentiated"
                            if cells_in_block[l] != self.cells[h] and (con_1 or con_2 and not (con_1 and con_2)):
                                if np.linalg.norm(cells_in_block[l].location - self.cells[h].location) <= distance:
                                    self.diff_graph.add_edge(self.cells[h], cells_in_block[l])
                        else:
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
            self.get_forces()

            # turn the forces acting on a cell into movements
            self.force_to_movement()

        # reset active force back to zero as these forces are only updated once per turn
        for i in range(len(self.cells)):
            self.cells[i].active_force = np.array([0.0, 0.0, 0.0])

    def get_forces(self):
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
            e_hat = (((1 - self.poisson ** 2) / self.youngs_mod) + (
                        (1 - self.poisson ** 2) / self.youngs_mod)) ** -1
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
                cell_1.inactive_force += jkr_force * normal
                cell_2.inactive_force -= jkr_force * normal

            # remove the edge if the it fails to meet the criteria for distance, JKR simulating that
            # the bond is broken
            elif bond_condition:
                self.jkr_graph.remove_edge(cell_1, cell_2)


    def force_to_movement(self):
        """ turns the forces acting on cells into
            small movements in the net force direction
        """
        # now re-loops over cells to move them and reduce work energy from kinetic energy
        for i in range(len(self.cells)):

            # stokes law for velocity based on force and fluid viscosity
            stokes_friction = 6 * math.pi * self.viscosity * self.cells[i].radius

            # update the velocity of the cell based on the solution
            self.cells[i].velocity = (self.cells[i].active_force + self.cells[i].inactive_force) / stokes_friction

            # set the possible new location
            new_location = self.cells[i].location + self.cells[i].velocity * self.move_time_step

            # loops over all directions of space
            for j in range(0, 3):
                # check if new location is in environment space if not simulation a collision with the bounds
                if new_location[j] > self.size[j]:
                    self.cells[i].location[j] = self.size[j]
                elif new_location[j] < 0:
                    self.cells[i].location[j] = 0.0
                else:
                    self.cells[i].location[j] = new_location[j]

            # reset velocity and inactive force and not the active force as that remains constant for the entire step
            self.cells[i].velocity = np.array([0.0, 0.0, 0.0])
            self.cells[i].inactive_force = np.array([0.0, 0.0, 0.0])