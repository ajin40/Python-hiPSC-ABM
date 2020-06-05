import numpy as np
import random as r
import igraph
import math
import itertools


class Simulation:
    def __init__(self, name, path, parallel, size, resolution, num_states, functions, neighbor_distance,
                 time_step_value, beginning_step, end_step, move_time_step, pluri_div_thresh, pluri_to_diff,
                 diff_div_thresh, boolean_thresh, death_thresh, diff_surround, adhesion_const, viscosity, group,
                 slices, image_quality, background_color, bound_color, color_mode, pluri_color, diff_color,
                 pluri_gata6_high_color, pluri_nanog_high_color, pluri_both_high_color, lonely_cell, contact_inhibit,
                 guye_move, motility_force, dox_step, max_radius, move_thresh, output_images, output_csvs, guye_radius,
                 guye_force):

        # documentation should explain how all instance variables are used
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
        self.move_thresh = move_thresh
        self.output_images = output_images
        self.output_csvs = output_csvs
        self.guye_radius = guye_radius
        self.guye_force = guye_force

        # these arrays hold all values of the cells, each index corresponds to a cell
        self.cell_locations = np.array([], dtype=float)
        self.cell_radii = np.array([], dtype=float)
        self.cell_motion = np.array([], dtype=bool)
        self.cell_booleans = np.array([], dtype=object)
        self.cell_states = np.array([], dtype=str)
        self.cell_diff_counter = np.array([], dtype=int)
        self.cell_div_counter = np.array([], dtype=int)
        self.cell_death_counter = np.array([], dtype=int)
        self.cell_bool_counter = np.array([], dtype=int)
        self.cell_motility_force = np.array([], dtype=object)
        self.cell_jkr_force = np.array([], dtype=object)
        self.cell_closest_diff = np.array([], dtype=int)

        # holds the current number of cells
        self.number_cells = 0

        # counts how many times an image is created for making videos
        self.image_counter = 0

        # keeps a running count of the simulation steps
        self.current_step = self.beginning_step

        # array to hold all of the Cell objects
        self.cells = np.array([], dtype=np.object)

        # array to hold all of the Extracellular objects
        self.extracellular = np.array([], dtype=np.object)

        # graph representing cells and neighbors
        self.neighbor_graph = igraph.Graph()

        # graph representing the presence of JKR adhesion bonds between cells
        self.jkr_graph = igraph.Graph()

        # holds indices that will used to divide a cell or remove it
        self.cells_to_divide = np.array([], dtype=int)
        self.cells_to_remove = np.array([], dtype=int)

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
        """ Loops over all indices of cells and updates
            certain parameters
        """
        # call the parallel version if you think gpus are cool
        if self.parallel:
            # prevents the need for having the numba library if it's not installed
            import Parallel
            Parallel.update_cells_gpu(self)

        # call the boring non-parallel cpu version
        else:
            # loop over the cells
            for i in range(self.number_cells):
                # increase the cell radius based on the state and whether or not it has reached the max size
                if self.cell_radii[i] < self.max_radius:
                    # pluripotent growth
                    if self.cell_states[i] == "Pluripotent":
                        self.cell_radii[i] += self.pluri_growth
                    # differentiated growth
                    else:
                        self.cell_radii[i] += self.diff_growth

                # checks to see if the non-moving cell should divide
                if not self.cell_motion[i]:
                    # check for contact inhibition of a differentiated cell
                    if self.cell_states[i] == "Differentiated" and self.cell_div_counter[i] >= self.diff_div_thresh:
                        neighbors = self.neighbor_graph.neighbors(i)
                        if len(neighbors) < self.contact_inhibit:
                            self.cells_to_divide = np.append(self.cells_to_divide, i)

                    # division for pluripotent cell
                    elif self.cell_states[i] == "Pluripotent" and self.cell_div_counter[i] >= self.pluri_div_thresh:
                        self.cells_to_divide = np.append(self.cells_to_divide, i)

                    # if not dividing stochastically increase the division counter
                    else:
                        self.cell_div_counter[i] += r.randint(0, 2)

                # activate the following pathway based on if dox has been induced yet
                if self.current_step >= self.dox_step:
                    # coverts position in space into an integer for array location
                    x_step = self.extracellular[0].dx
                    y_step = self.extracellular[0].dy
                    z_step = self.extracellular[0].dz
                    half_index_x = self.cell_locations[i][0] // (x_step / 2)
                    half_index_y = self.cell_locations[i][1] // (y_step / 2)
                    half_index_z = self.cell_locations[i][2] // (z_step / 2)
                    index_x = math.ceil(half_index_x / 2)
                    index_y = math.ceil(half_index_y / 2)
                    index_z = math.ceil(half_index_z / 2)

                    # if a certain spot of the grid is less than the max FGF4 it can hold and the cell is NANOG high
                    # increase the FGF4 by 1
                    if self.extracellular[0].diffuse_values[index_x][index_y][index_z] < \
                            self.extracellular[0].maximum and self.cell_booleans[i][3] == 1:
                        self.extracellular[0].diffuse_values[index_x][index_y][index_z] += 1

                    # if the FGF4 amount for the location is greater than 0, set the fgf4_bool value to be 1 for the
                    # functions
                    if self.extracellular[0].diffuse_values[index_x][index_y][index_z] > 0:
                        fgf4_bool = 1
                    else:
                        fgf4_bool = 0

                    # temporarily hold the FGFR value
                    temp_fgfr = self.cell_booleans[i][0]

                    # only update the booleans when the counter matches the boolean update rate
                    if self.cell_bool_counter[i] % self.boolean_thresh == 0:
                        # gets the functions from the simulation
                        function_list = self.functions

                        # xn is equal to the value corresponding to its function
                        x1 = fgf4_bool
                        x2 = self.cell_booleans[i][0]
                        x3 = self.cell_booleans[i][1]
                        x4 = self.cell_booleans[i][2]
                        x5 = self.cell_booleans[i][3]

                        # evaluate the functions by turning them from strings to math equations
                        new_fgf4 = eval(function_list[0]) % self.num_states
                        new_fgfr = eval(function_list[1]) % self.num_states
                        new_erk = eval(function_list[2]) % self.num_states
                        new_gata6 = eval(function_list[3]) % self.num_states
                        new_nanog = eval(function_list[4]) % self.num_states

                        # updates self.booleans with the new boolean values and returns the new fgf4 value
                        self.cell_booleans[i] = np.array([new_fgfr, new_erk, new_gata6, new_nanog])

                    # otherwise carry the same value for fgf4
                    else:
                        new_fgf4 = fgf4_bool

                    # increase the boolean counter
                    self.cell_bool_counter[i] += 1

                    # if the temporary FGFR value is 0 and the FGF4 value is 1 decrease the amount of FGF4 by 1
                    # this simulates FGFR using FGF4
                    if temp_fgfr == 0 and new_fgf4 == 1 and \
                            self.extracellular[0].diffuse_values[index_x][index_y][index_z] >= 1:
                        self.extracellular[0].diffuse_values[index_x][index_y][index_z] -= 1

                    # if the cell is GATA6 high and pluripotent increase the differentiation counter by 1
                    if self.cell_booleans[i][2] == 1 and self.cell_states[i] == "Pluripotent":
                        self.cell_diff_counter[i] += 1

                        # if the differentiation counter is greater than the threshold, differentiate
                        if self.cell_diff_counter[i] >= self.pluri_to_diff:
                            # change the state to differentiated
                            self.cell_states[i] = "Differentiated"

                            # set GATA6 high and NANOG low
                            self.cell_booleans[i][2] = 1
                            self.cell_booleans[i][3] = 0

                            # allow the cell to move again
                            self.cell_motion[i] = True

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
        """ Will add a cell to the main cell holder
            "self.cells" in addition to adding the instance
            to all graphs
        """
        # adds it to the array holding the cell objects
        self.cells = np.append(self.cells, cell)

        # add it to the following graphs, this is simply done by increasing the graph length by one
        self.neighbor_graph.add_vertex()
        self.jkr_graph.add_vertex()

    def remove_cell(self, cell):
        """ Will remove a cell from the main cell holder
            "self.cells" in addition to removing the instance
            from all graphs
        """
        # find the index of the cell and delete it from self.cells
        index = int(np.argwhere(self.cells == cell))
        self.cells = np.delete(self.cells, index)

        # remove the particular index from the following graphs as these deal in terms of indices not objects
        # this will actively adjust edges as the indices change, so no worries here
        self.neighbor_graph.delete_vertices(index)
        self.jkr_graph.delete_vertices(index)

    def update_cell_queue(self):
        """ Controls how cells are added/removed from
            the simulation.
        """
        # give the user an idea of how many cells are being added/removed during a given step
        print("Adding " + str(len(self.cells_to_add)) + " cells...")
        print("Removing " + str(len(self.cells_to_remove)) + " cells...")

        # loops over all objects to add
        for i in range(len(self.cells_to_add)):
            self.add_cell(self.cells_to_add[i])

            # Cannot add all of the new cell objects, otherwise several cells are likely to be added
            #   in close proximity to each other at later time steps. Such object addition, coupled
            #   with handling collisions, make give rise to sudden changes in overall positions of
            #   cells within the simulation. Instead, collisions are handled after 'group' number
            #   of cell objects are added.
            if self.group != 0:
                if (i + 1) % self.group == 0:
                    self.handle_movement()

        # loops over all objects to remove
        for i in range(len(self.cells_to_remove)):
            self.remove_cell(self.cells_to_remove[i])

            # much like above where many cells are added together, removing all at once may create unrealistic results
            if self.group != 0:
                if (i + 1) % self.group == 0:
                    self.handle_movement()

        # clear the arrays for the next step
        self.cells_to_add = np.array([], dtype=np.object)
        self.cells_to_remove = np.array([], dtype=np.object)

    def check_neighbors(self):
        """ checks all of the distances between cells if it
            is less than a fixed value create a connection
            between two cells.
        """
        # clear all of the edges in the neighbor graph
        self.neighbor_graph.delete_edges(None)

        # distance threshold between two cells to designate a neighbor
        distance = self.neighbor_distance

        # provide an idea of the maximum number of neighbors for a cells
        max_neighbors = 15
        length = len(self.cells) * max_neighbors
        edge_holder = np.zeros((length, 2), dtype=np.int)

        # call the parallel version if desired
        if self.parallel:
            # prevents the need for having the numba library if it's not installed
            import Parallel
            edge_holder = Parallel.check_neighbors_gpu(self, distance, edge_holder, max_neighbors)

        # call the boring non-parallel cpu version
        else:
            # a counter used to know where the next edge will be placed in the edges_holder
            edge_counter = 0

            # create an 3D array that will divide the space up into a collection of bins
            bins_size = self.size // distance + np.array([3, 3, 3])
            bins_size = tuple(bins_size.astype(int))
            bins = np.empty(bins_size, dtype=np.object)

            # each bin will be a numpy array that will hold indices of cells
            for i, j, k in itertools.product(range(bins_size[0]), range(bins_size[1]), range(bins_size[2])):
                bins[i][j][k] = np.array([], dtype=np.int)

            # loops over all cells appending their index value in the corresponding bin
            for pivot_index in range(len(self.cells)):
                # offset the bin location by 1 to help when searching over bins and reduce potential error of cells
                # that may be slightly outside the space
                bin_location = self.cells[pivot_index].location // distance + np.array([1, 1, 1])
                bin_location = bin_location.astype(int)
                x, y, z = bin_location[0], bin_location[1], bin_location[2]

                # adds the cell to the corresponding bin
                bins[x][y][z] = np.append(bins[x][y][z], pivot_index)

                # looks at the bins surrounding a given bin that houses the cell as these are the only potential cells
                # to be neighbors of the cell in question
                for i, j, k in itertools.product(range(-1, 2), repeat=3):
                    # get the array that is holding the indices of a cells in a block
                    indices_in_bin = bins[x + i][y + j][z + k]

                    # looks at the cells in a block and decides if they are neighbors
                    for l in range(len(indices_in_bin)):
                        # get the index of the current cell in question
                        current_index = indices_in_bin[l]

                        # for the specified index in that block get the cell object in self.cells
                        current_cell = self.cells[current_index]

                        # check to see if that cell is within the search radius
                        if np.linalg.norm(current_cell.location - self.cells[pivot_index].location) <= distance and \
                                pivot_index != current_index:
                            # update the edge array and increase the place for the next addition
                            edge_holder[edge_counter][0] = pivot_index
                            edge_holder[edge_counter][1] = current_index
                            edge_counter += 1

        # add the new edges and remove any duplicate edges or loops
        self.neighbor_graph.add_edges(edge_holder)
        self.neighbor_graph.simplify()

    def update_neighbors(self):
        """ Updates each cell's instance variable
            pointing to the cell objects that are its
            neighbors.
        """
        # this is separate from check_neighbors as that is run numerous times without the need for updating
        # all the instance variables holding the neighbor cell objects
        # loops over all cells and gets the neighbors based on the index in the graph
        for i in range(len(self.cells)):
            # clear the neighbors from the previous step
            self.cells[i].neighbors = np.array([], np.object)

            # get the indices of the node's neighbors
            neighbors = self.neighbor_graph.neighbors(i)

            # loops over the neighbors adding the corresponding cell object to the array holding the neighbors
            for j in range(len(neighbors)):
                self.cells[i].neighbors = np.append(self.cells[i].neighbors, self.cells[neighbors[j]])

    def closest_diff(self):
        """ This will find the closest differentiated
            cell within a given search radius and assign
            that cell to an instance variable for each cell
        """
        # this may appear rather similar to the check_neighbors function; however there are stark differences that
        # warrant a separate function rather than a combination
        # get the distance for the search radius
        distance = self.guye_radius

        # create an array of blocks that takes up the entire space and extra to prevent errors
        blocks_size = self.size // distance + np.array([3, 3, 3])
        blocks_size = tuple(blocks_size.astype(int))
        blocks = np.empty(blocks_size, dtype=np.object)

        # gives each block an array that acts as a holder for cells optimized triple for-loop
        for i, j, k in itertools.product(range(blocks_size[0]), range(blocks_size[1]), range(blocks_size[2])):
            blocks[i][j][k] = np.array([], dtype=np.int)

        # loops over all cells appending their index value in the corresponding block
        for h in range(len(self.cells)):
            if self.cells[h].state == "Differentiated":
                # offset the block location by 1 to help when searching over blocks and reduce potential error
                block_location = self.cells[h].location // distance + np.array([1, 1, 1])
                block_location = block_location.astype(int)
                x, y, z = block_location[0], block_location[1], block_location[2]

                # adds the cell to a given block
                blocks[x][y][z] = np.append(blocks[x][y][z], h)

        for h in range(len(self.cells)):
            if self.cells[h].state == "Pluripotent":
                # offset the block location by 1 to help when searching over blocks and reduce potential error
                block_location = self.cells[h].location // distance + np.array([1, 1, 1])
                block_location = block_location.astype(int)
                x, y, z = block_location[0], block_location[1], block_location[2]

                # create an arbitrary distance that any cell close enough will replace
                closest_index = None
                closest_dist = distance * 2

                # looks at the blocks surrounding a given block that houses the cell
                for i, j, k in itertools.product(range(-1, 2), repeat=3):
                    # an array holding the cell indices
                    indices_in_block = blocks[x + i][y + j][z + k]

                    # looks at the cells in a block and decides if they are neighbors
                    for l in range(len(indices_in_block)):
                        # for the specified index in that block get the cell object in self.cells
                        cell_in_block = self.cells[indices_in_block[l]]

                        # check to see if that cell is within the search radius and not the same cell
                        magnitude = np.linalg.norm(cell_in_block.location - self.cells[h].location)
                        if magnitude <= distance and h != indices_in_block[l]:
                            # if it's closer than the last cell, update the closest magnitude and index
                            if magnitude < closest_dist:
                                closest_index = indices_in_block[l]
                                closest_dist = magnitude

                # check to make sure the initial cell doesn't slip through
                if closest_dist <= distance:
                    self.cells[h].closest_diff = self.cells[closest_index]

    def handle_movement(self):
        """ runs the following functions together for a
            given time amount. Resets the force and
            velocity arrays as well.
        """
        # get the total amount of times the cells will be incrementally moved during the step
        steps = int(self.time_step_value / self.move_time_step)

        # run the following functions consecutively for the given amount of steps
        for i in range(steps):
            # calculate the forces acting on each cell
            self.get_forces()

            # turn the forces into movement
            self.apply_forces()

            # recheck neighbors after the cells have moved
            self.check_neighbors()

        for i in range(len(self.cells)):
            self.cells[i].active_force = np.array([0.0, 0.0, 0.0], dtype=float)

    def get_forces(self):
        """ goes through all of the cells and quantifies any forces arising
            from adhesion or repulsion between the cells
        """
        # get a list of the edges from both graphs
        neighbor_edges = self.neighbor_graph.get_edgelist()

        # create arrays used for adding/deleting edges, sadly there is a difference between the way these
        # input information
        add_jkr_edges = np.zeros((len(neighbor_edges), 2), dtype=np.int)
        delete_jkr_edges = np.zeros(len(neighbor_edges), dtype=np.int)

        # make sure that all edges in the jkr graph are also in the neighbor graph as there is no way to eliminate
        # stray edges if they exist
        self.jkr_graph = self.jkr_graph & self.neighbor_graph

        # look at the neighbor edges and see if any will classify as jkr edges
        if self.parallel:
            import Parallel
            add_jkr_edges = Parallel.get_jkr_edges(self, neighbor_edges, add_jkr_edges)

        else:
            # loop over the neighbor edges and check to see if a jkr bond will be created
            for i in range(len(neighbor_edges)):
                # get the indices of the nodes in the edge
                index_1 = neighbor_edges[i][0]
                index_2 = neighbor_edges[i][1]

                # assigns the nodes of each edge to a variable
                cell_1 = self.cells[index_1]
                cell_2 = self.cells[index_2]

                # hold the vector between the centers of the cells and the magnitude of this vector
                disp_vector = cell_1.location - cell_2.location
                magnitude = np.linalg.norm(disp_vector)

                # get the total overlap of the cells used later in calculations
                overlap = cell_1.radius + cell_2.radius - magnitude

                # indicate that an adhesive bond has formed between the cells
                if overlap >= 0:
                    add_jkr_edges[i] = (index_1, index_2)

        # add the new jkr edges
        self.jkr_graph.add_edges(add_jkr_edges)
        self.jkr_graph.simplify()

        # get the values for Youngs modulus and Poisson's ratio
        poisson = self.poisson
        youngs = self.youngs_mod
        adhesion_const = self.adhesion_const

        # get the updated edges of the jkr graph
        jkr_edges = self.jkr_graph.get_edgelist()

        # call the parallel version if desired
        if self.parallel:
            # prevents the need for having the numba library if it's not installed
            import Parallel
            delete_jkr_edges = Parallel.get_forces_gpu(self, jkr_edges, delete_jkr_edges, poisson, youngs,
                                                       adhesion_const)

        # call the boring non-parallel cpu version
        else:
            # loops over the jkr edges
            for i in range(len(jkr_edges)):
                # get the indices of the nodes in the edge
                index_1 = jkr_edges[i][0]
                index_2 = jkr_edges[i][1]

                # assigns the nodes of each edge to a variable
                cell_1 = self.cells[index_1]
                cell_2 = self.cells[index_2]

                # hold the vector between the centers of the cells and the magnitude of this vector
                disp_vector = cell_1.location - cell_2.location
                magnitude = np.linalg.norm(disp_vector)

                if magnitude == 0:
                    normal = np.array([0.0, 0.0, 0.0])
                else:
                    normal = disp_vector / np.linalg.norm(disp_vector)

                # get the total overlap of the cells used later in calculations
                overlap = cell_1.radius + cell_2.radius - magnitude

                # gets two values used for JKR
                e_hat = (((1 - poisson ** 2) / youngs) + ((1 - poisson ** 2) / youngs)) ** -1
                r_hat = ((1 / cell_1.radius) + (1 / cell_2.radius)) ** -1

                # used to calculate the max adhesive distance after bond has been already formed
                overlap_ = (((math.pi * adhesion_const) / e_hat) ** (2 / 3)) * (r_hat ** (1 / 3))

                # get the nondimensionalized overlap, used for later calculations and checks
                # also for the use of a polynomial approximation of the force
                d = overlap / overlap_

                # check to see if the cells will have a force interaction
                if d > -0.360562:

                    # plug the value of d into the nondimensionalized equation for the JKR force
                    f = (-0.0204 * d ** 3) + (0.4942 * d ** 2) + (1.0801 * d) - 1.324

                    # convert from the nondimensionalization to find the adhesive force
                    jkr_force = f * math.pi * self.adhesion_const * r_hat

                    # adds the adhesive force as a vector in opposite directions to each cell's force holder
                    cell_1.inactive_force += jkr_force * normal
                    cell_2.inactive_force -= jkr_force * normal

                # remove the edge if the it fails to meet the criteria for distance, JKR simulating that
                # the bond is broken
                else:
                    delete_jkr_edges[i] = i

        # update the jkr graph after the arrays have been updated by either the parallel or non-parallel function
        self.jkr_graph.delete_edges(delete_jkr_edges)
        self.jkr_graph.simplify()

    def apply_forces(self):
        """ Turns the active motility/division forces
            and inactive JKR forces into movement
        """
        # call the parallel version if desired
        if self.parallel:
            # prevents the need for having the numba library if it's not installed
            import Parallel
            Parallel.apply_forces_gpu(self)

        # call the boring non-parallel cpu version
        else:
            # loops over cells to move them
            for i in range(len(self.cells)):
                # stokes law for velocity based on force and fluid viscosity
                stokes_friction = 6 * math.pi * self.viscosity * self.cells[i].radius

                # update the velocity of the cell based on the solution
                velocity = (self.cells[i].active_force + self.cells[i].inactive_force) / stokes_friction

                # set the possible new location
                new_location = self.cells[i].location + velocity * self.move_time_step

                # loops over all directions of space
                for j in range(0, 3):
                    # check if new location is in environment space if not simulation a collision with the bounds
                    if new_location[j] > self.size[j]:
                        self.cells[i].location[j] = self.size[j]
                    elif new_location[j] < 0:
                        self.cells[i].location[j] = 0.0
                    else:
                        self.cells[i].location[j] = new_location[j]

                # reset the forces back to zero
                self.cells[i].inactive_force = np.array([0.0, 0.0, 0.0])
                self.cells[i].active_force = np.array([0.0, 0.0, 0.0])