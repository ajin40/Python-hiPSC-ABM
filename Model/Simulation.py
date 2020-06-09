import numpy as np
import random as r
import igraph
import math
import itertools


# used to hold all values necessary to the simulation as it moves from one step to the next
class Simulation:
    def __init__(self, name, path, parallel, size, resolution, num_states, functions, neighbor_distance,
                 time_step_value, beginning_step, end_step, move_time_step, pluri_div_thresh, pluri_to_diff,
                 diff_div_thresh, boolean_thresh, death_thresh, diff_surround, adhesion_const, viscosity, group,
                 slices, image_quality, background_color, bound_color, color_mode, pluri_color, diff_color,
                 pluri_gata6_high_color, pluri_nanog_high_color, pluri_both_high_color, lonely_cell, contact_inhibit,
                 guye_move, motility_force, dox_step, max_radius, move_thresh, output_images, output_csvs,
                 guye_distance, guye_force, jkr_distance):

        # the following instance variables should be fixed meaning that they don't change from step to step
        # they are merely used to hold initial parameters from the template file
        # see documentation for better explanations
        self.name = name    # the name of the simulation such as Run_43
        self.path = path    # the directory of where the simulation will output to
        self.parallel = parallel    # whether the model is used parallel GPU processing or not
        self.size = size    # the dimensions of the space the cells reside in
        self.resolution = resolution    # the spatial resolution of the environment, used for diffusion
        self.num_states = num_states    # the number of states for the finite dynamical system
        self.functions = functions    # the finite dynamical system functions
        self.neighbor_distance = neighbor_distance    # the distance threshold for assigning nearby cells as neighbors
        self.time_step_value = time_step_value    # the real-time value of each step
        self.beginning_step = beginning_step    # the step the simulation starts on, used for continuations
        self.end_step = end_step    # the final step of a simulation
        self.move_time_step = move_time_step    # the time step used for each time the JKR contact function is used
        self.pluri_div_thresh = pluri_div_thresh    # the division threshold (in steps) of a pluripotent cell
        self.pluri_to_diff = pluri_to_diff    # the differentiation threshold (in steps) of a pluripotent cell
        self.diff_div_thresh = diff_div_thresh    # the division threshold (in steps) of a differentiated cell
        self.boolean_thresh = boolean_thresh    # the boolean threshold (in steps) for updating the FDS
        self.death_thresh = death_thresh    # the death threshold (in steps) for cell death
        self.diff_surround = diff_surround    # the number of cells needed to help induce differentiation
        self.adhesion_const = adhesion_const    # the adhesion constant used for JKR contact
        self.viscosity = viscosity    # the viscosity of the medium used for stokes friction
        self.group = group    # the number of cells introduced into the space at once
        self.slices = slices    # the number of 2D slices of a 3D space
        self.image_quality = image_quality    # the output image/video dimensions in pixels
        self.background_color = background_color    # the background space color
        self.bound_color = bound_color    # the color of the bounds of the space
        self.color_mode = color_mode    # used to vary which method of coloring used
        self.pluri_color = pluri_color    # color of pluripotent cell
        self.diff_color = diff_color    # color of a differentiated cell
        self.pluri_gata6_high_color = pluri_gata6_high_color    # color of pluripotent gata6 high cell
        self.pluri_nanog_high_color = pluri_nanog_high_color    # color of pluripotent nanog high cell
        self.pluri_both_high_color = pluri_both_high_color    # color of pluripotent gata6/nanog high cell
        self.lonely_cell = lonely_cell    # the threshold of cells need to designate a cell lonely
        self.contact_inhibit = contact_inhibit    # the contact inhibition threshold of differentiated cells
        self.guye_move = guye_move    # whether or not to use the Guye method of motility
        self.motility_force = motility_force    # the active force of a cell moving
        self.dox_step = dox_step    # the step at which dox is added to the simulation
        self.max_radius = max_radius    # the maximum radius of a cell
        self.move_thresh = move_thresh    # the threshold of neighbors for a differentiated cell to move
        self.output_images = output_images    # whether or not to produce images
        self.output_csvs = output_csvs    # whether or not to produce csvs with cell information
        self.guye_distance = guye_distance    # the radius of search for the nearest differentiated neighbor
        self.guye_force = guye_force    # the movement force for pluripotent cell to a differentiated cell
        self.jkr_distance = jkr_distance    # the radius of search for JKR adhesive bonds formed between cells

        # these arrays hold all values of the cells, each index corresponds to a cell
        # you may ask why not create an array that holds a bunch of Cell objects...and my answer to that
        # is in order to parallelize the functions with a GPU you need to convert a collection of variables into
        # an array that can be interpreted by the GPU so Cell classes will have to be constantly mined for their
        # instance variables, put into an array, and then reevaluated from an updated array after processing
        self.cell_locations = np.empty((0, 3), dtype=float)
        self.cell_radii = np.empty((0, 1), dtype=float)
        self.cell_motion = np.empty((0, 1), dtype=bool)
        self.cell_booleans = np.empty((0, 4), dtype=float)
        self.cell_states = np.empty((0, 1), dtype=str)
        self.cell_diff_counter = np.empty((0, 1), dtype=int)
        self.cell_div_counter = np.empty((0, 1), dtype=int)
        self.cell_death_counter = np.empty((0, 1), dtype=int)
        self.cell_bool_counter = np.empty((0, 1), dtype=int)
        self.cell_motility_force = np.empty((0, 3), dtype=float)
        self.cell_jkr_force = np.empty((0, 3), dtype=float)
        self.cell_closest_diff = np.empty((0, 1), dtype=int)

        # holds the current number of cells
        self.number_cells = 0

        # counts how many times an image is created for making videos
        self.image_counter = 0

        # keeps a running count of the simulation steps
        self.current_step = self.beginning_step

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
        """ Prints the step number and number of
            cells. Checks to make sure all arrays
            match in length.
        """
        # prints info about the current step and number of cells
        print("Step: " + str(self.current_step))
        print("Number of cells: " + str(self.number_cells))

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

    def random_vector(self):
        """ Computes a random point on a unit sphere centered at the origin
            Returns - point [x,y,z]
        """
        # gets random angle on the cell
        theta = r.random() * 2 * math.pi

        # gets x,y,z off theta and whether 2D or 3D
        if self.size[2] == 0:
            # 2D
            x = math.cos(theta)
            y = math.sin(theta)
            return np.array([x, y, 0.0])

        else:
            # 3D spherical coordinates
            phi = r.random() * 2 * math.pi
            radius = math.cos(phi)

            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            z = math.sin(phi)
            return np.array([x, y, z])

    def add_cell(self, location, radius, motion, booleans, state, diff_counter, div_counter, death_counter,
                 bool_counter, motility_force, jkr_force, closest_diff):
        """ Adds the parameters to each of the arrays
            that act as holders for all information
        """
        # adds it to the array holding the cell objects
        self.cell_locations = np.append(self.cell_locations, [location], axis=0)
        self.cell_radii = np.append(self.cell_radii, radius)
        self.cell_motion = np.append(self.cell_motion, motion)
        self.cell_booleans = np.append(self.cell_booleans, [booleans], axis=0)
        self.cell_states = np.append(self.cell_states, state)
        self.cell_diff_counter = np.append(self.cell_diff_counter, diff_counter)
        self.cell_div_counter = np.append(self.cell_div_counter, div_counter)
        self.cell_death_counter = np.append(self.cell_death_counter, death_counter)
        self.cell_bool_counter = np.append(self.cell_bool_counter, bool_counter)
        self.cell_motility_force = np.append(self.cell_motility_force, [motility_force], axis=0)
        self.cell_jkr_force = np.append(self.cell_jkr_force, [jkr_force], axis=0)
        self.cell_closest_diff = np.append(self.cell_closest_diff, closest_diff)

        # add it to the following graphs, this is simply done by increasing the graph length by one
        self.neighbor_graph.add_vertex()
        self.jkr_graph.add_vertex()

        # update the number of cells
        self.number_cells += 1

    def remove_cell(self, index):
        """ Will remove a cell from the main cell holder
            "self.cells" in addition to removing the instance
            from all graphs
        """
        # delete the index of each holder array
        self.cell_locations = np.delete(self.cell_locations, index, axis=0)
        self.cell_radii = np.delete(self.cell_radii, index)
        self.cell_motion = np.delete(self.cell_motion, index)
        self.cell_booleans = np.delete(self.cell_booleans, index, axis=0)
        self.cell_states = np.delete(self.cell_states, index)
        self.cell_diff_counter = np.delete(self.cell_diff_counter, index)
        self.cell_div_counter = np.delete(self.cell_div_counter, index)
        self.cell_death_counter = np.delete(self.cell_death_counter, index)
        self.cell_bool_counter = np.delete(self.cell_bool_counter, index)
        self.cell_motility_force = np.delete(self.cell_motility_force, index, axis=0)
        self.cell_jkr_force = np.delete(self.cell_jkr_force, index, axis=0)
        self.cell_closest_diff = np.delete(self.cell_closest_diff, index)

        # remove the particular index from the following graphs as these deal in terms of indices not objects
        # this will actively adjust edges as the indices change, so no worries here
        self.neighbor_graph.delete_vertices(index)
        self.jkr_graph.delete_vertices(index)

        # update the number of cells
        self.number_cells -= 1

    def update_cell_queue(self):
        """ Controls how cells are added/removed from
            the simulation.
        """
        # give the user an idea of how many cells are being added/removed during a given step
        print("Adding " + str(len(self.cells_to_divide)) + " cells...")
        print("Removing " + str(len(self.cells_to_remove)) + " cells...")

        # loops over all objects to add
        for i in range(len(self.cells_to_divide)):
            self.divide(self.cells_to_divide[i])

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
            index = self.cells_to_remove[i]
            self.remove_cell(index)

            # adjusts the indices as the holders will shift when a cell is removed
            for j in range(i + 1, len(self.cells_to_remove)):
                if index < self.cells_to_remove[j]:
                    self.cells_to_remove[j] -= 1

            # much like above where many cells are added together, removing all at once may create unrealistic results
            if self.group != 0:
                if (i + 1) % self.group == 0:
                    self.handle_movement()

        # clear the arrays for the next step
        self.cells_to_divide = np.array([], dtype=int)
        self.cells_to_remove = np.array([], dtype=int)

    def divide(self, index):
        """ Simulates the division of cell by mitosis
            in addition to adding the cell with given
            initial parameters
        """
        # move the cells to a position that is representative of the new locations of daughter cells
        division_position = self.random_vector() * (self.max_radius - self.min_radius)
        self.cell_locations[index] += division_position
        location = self.cell_locations[index] - division_position

        # reduce radius to minimum size, set the division counter to zero, and None as the nearest differentiated cell
        self.cell_radii[index] = radius = self.min_radius
        self.cell_div_counter[index] = div_counter = 0
        self.cell_closest_diff[index] = closest_diff = None

        # keep identical values for motion, booleans, state, differentiation, cell death, and boolean update
        motion = self.cell_motion[index]
        booleans = self.cell_booleans[index]
        state = self.cell_states[index]
        diff_counter = self.cell_diff_counter[index]
        death_counter = self.cell_death_counter[index]
        bool_counter = self.cell_bool_counter[index]

        # set the force vector to zero
        motility_force = np.zeros(3, dtype=float)
        jkr_force = np.zeros(3, dtype=float)

        # add the cell to the array holders and graphs via the instance method
        self.add_cell(location, radius, motion, booleans, state, diff_counter, div_counter, death_counter,
                      bool_counter, motility_force, jkr_force, closest_diff)

    def update_cells(self):
        """ Loops over all indices of cells and updates
            certain parameters
        """
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

    def cell_death(self):
        """ Simulates cell death based on pluripotency
            and number of neighbors
        """
        # loops over all cells
        for i in range(self.number_cells):
            # checks to see if cell is pluripotent
            if self.cell_states[i] == "Pluripotent":
                # looks at the neighbors and counts them, adding to the death counter if not enough neighbors
                neighbors = self.neighbor_graph.neighbors(i)
                if len(neighbors) < self.lonely_cell:
                    self.cell_death_counter[i] += 1
                else:
                    self.cell_death_counter[i] = 0

                # removes cell if it meets the parameters
                if self.cell_death_counter[i] >= self.death_thresh:
                    self.cells_to_remove = np.append(self.cells_to_remove, i)

    def cell_diff_surround(self):
        """ Simulates the phenomenon of differentiated
            cells inducing the differentiation of a
            pluripotent/NANOG high cell
        """
        # loops over all cells
        for i in range(self.number_cells):
            # checks to see if cell is pluripotent and GATA6 low
            if self.cell_states[i] == "Pluripotent" and self.cell_booleans[i][2] == 0:
                # finds neighbors of a cell
                neighbors = self.neighbor_graph.neighbors(i)

                # holds the current number differentiated neighbors
                num_diff_neighbors = 0

                # loops over the neighbors of a cell
                for j in range(len(neighbors)):
                    # checks to see if current neighbor is differentiated if so add it to the counter
                    if self.cell_states[neighbors[j]] == "Differentiated":
                        num_diff_neighbors += 1

                    # if the number of differentiated meets the threshold, increase the counter and break the loop
                    if num_diff_neighbors >= self.diff_surround:
                        self.cell_diff_counter[i] += 1
                        break

    def cell_motility(self):
        """ Gives the cells a motion force, depending on
            set rules for the cell types.
        """
        # loop over all of the cells
        for i in range(self.number_cells):
            # set motion to false if the cell is surrounded by many neighbors
            neighbors = self.neighbor_graph.neighbors(i)

            if len(neighbors) >= self.move_thresh:
                self.cell_motion[i] = False

            # check whether differentiated or pluripotent
            if self.cell_states[i] == "Differentiated":

                # directed movement if the cell has neighbors
                if 0 < len(neighbors) < 6:
                    # create a vector to hold the sum of normal vectors between a cell and its neighbors
                    vector_holder = np.array([0.0, 0.0, 0.0])

                    # loop over the neighbors getting the normal and adding to the holder
                    for j in range(len(neighbors)):
                        vector = self.cell_locations[neighbors[j]] - self.cell_locations[i]
                        vector_holder += vector

                    # get the magnitude of the sum of normals
                    magnitude = np.linalg.norm(vector_holder)

                    # if for some case, its zero set the new normal vector to zero
                    if magnitude == 0:
                        normal = np.array([0.0, 0.0, 0.0])
                    else:
                        normal = vector_holder / magnitude

                    # move in direction opposite to cells
                    self.cell_motility_force[i] += self.motility_force * normal * -1

                # if there aren't any neighbors and still in motion then move randomly
                elif self.cell_motion[i]:
                    self.cell_motility_force[i] += self.random_vector() * self.motility_force

            # for pluripotent cells
            else:
                # apply movement if the cell is "in motion"
                if self.cell_motion[i]:
                    if self.cell_booleans[i][2] == 1:
                        # continue if using Guye et al. movement and if there exists differentiated cells
                        if self.guye_move and self.cell_closest_diff[i] is not None:
                            # get the differentiated neighbors
                            guye_neighbor = self.cell_closest_diff[i]

                            vector = self.cell_locations[guye_neighbor] - self.cell_locations[i]
                            magnitude = np.linalg.norm(vector)

                            # move in the direction of the closest differentiated neighbor
                            normal = vector / magnitude
                            self.cell_motility_force[i] += normal * self.guye_force

                        else:
                            # if not Guye et al. movement, move randomly
                            self.cell_motility_force[i] += self.random_vector() * self.motility_force
                    else:
                        # if not GATA6 high
                        self.cell_motility_force[i] += self.random_vector() * self.motility_force

    def check_neighbors(self):
        """ checks all of the distances between cells if it
            is less than a fixed value create a connection
            between two cells.
        """
        # clear all of the edges in the neighbor graph and get the radius of search length
        self.neighbor_graph.delete_edges(None)
        distance = self.neighbor_distance

        # provide an idea of the maximum number of neighbors for a cells
        max_neighbors = 15
        length = self.number_cells * max_neighbors
        edge_holder = np.zeros((length, 2), dtype=int)

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
            bins_size = self.size // distance + np.array([4, 4, 4])
            bins_size = tuple(bins_size.astype(int))
            bins = np.empty(bins_size, dtype=np.object)

            # each bin will be a numpy array that will hold indices of cells
            for i, j, k in itertools.product(range(bins_size[0]), range(bins_size[1]), range(bins_size[2])):
                bins[i][j][k] = np.array([], dtype=np.int)

            # loops over all cells appending their index value in the corresponding bin
            for pivot_index in range(self.number_cells):
                # offset the bin location by 1 to help when searching over bins and reduce potential error of cells
                # that may be slightly outside the space
                bin_location = self.cell_locations[pivot_index] // distance + np.array([2, 2, 2])
                bin_location = bin_location.astype(int)
                x, y, z = bin_location[0], bin_location[1], bin_location[2]

                # adds the cell to the corresponding bin
                bins[x][y][z] = np.append(bins[x][y][z], pivot_index)

                for i, j, k in itertools.product(range(-1, 2), repeat=3):
                    # get the array that is holding the indices of a cells in a block
                    indices_in_bin = bins[x + i][y + j][z + k]

                    # looks at the cells in a block and decides if they are neighbors
                    for l in range(len(indices_in_bin)):
                        # get the index of the current cell in question
                        current_index = indices_in_bin[l]

                        # check to see if that cell is within the search radius
                        if np.linalg.norm(self.cell_locations[current_index] - self.cell_locations[pivot_index]) \
                                <= distance and pivot_index != current_index:
                            # update the edge array and increase the place for the next addition
                            edge_holder[edge_counter][0] = pivot_index
                            edge_holder[edge_counter][1] = current_index
                            edge_counter += 1

        # add the new edges and remove any duplicate edges or loops
        self.neighbor_graph.add_edges(edge_holder)
        self.neighbor_graph.simplify()

    def nearest_diff(self):
        """ looks at cells within a given radius
            a determines the closest differentiated
            cell to a pluripotent cell.
        """
        # get the radius of search length
        distance = self.guye_distance

        # create an 3D array that will divide the space up into a collection of bins
        bins_size = self.size // distance + np.array([4, 4, 4])
        bins_size = tuple(bins_size.astype(int))
        bins = np.empty(bins_size, dtype=np.object)

        # each bin will be a numpy array that will hold indices of cells
        for i, j, k in itertools.product(range(bins_size[0]), range(bins_size[1]), range(bins_size[2])):
            bins[i][j][k] = np.array([], dtype=np.int)

        # loops over all pluripotent cells appending their index value in the corresponding bin
        for pivot_index in range(self.number_cells):
            if self.cell_states[pivot_index] == "Pluripotent":
                # offset the bin location by 1 to help when searching over bins and reduce potential error of
                # cells that may be slightly outside the space
                bin_location = self.cell_locations[pivot_index] // distance + np.array([2, 2, 2])
                bin_location = bin_location.astype(int)
                x, y, z = bin_location[0], bin_location[1], bin_location[2]

                # adds the cell to the corresponding bin
                bins[x][y][z] = np.append(bins[x][y][z], pivot_index)

        # loops over all differentiated cells looking for the surrounding differentiated cells
        for pivot_index in range(self.number_cells):
            if self.cell_states[pivot_index] == "Differentiated":
                bin_location = self.cell_locations[pivot_index] // distance + np.array([2, 2, 2])
                bin_location = bin_location.astype(int)
                x, y, z = bin_location[0], bin_location[1], bin_location[2]
                # create an arbitrary distance that any cell close enough will replace
                closest_index = None
                closest_dist = distance * 2

                # look at the surrounding bins
                for i, j, k in itertools.product(range(-1, 2), repeat=3):
                    # get the array that is holding the indices of a cells in a block
                    indices_in_bin = bins[x + i][y + j][z + k]

                    # looks at the cells in a block and decides if they are neighbors
                    for l in range(len(indices_in_bin)):
                        # get the index of the current cell in question
                        current_index = indices_in_bin[l]

                        # check to see if that cell is within the search radius and not the same cell
                        m = np.linalg.norm(self.cell_locations[current_index] - self.cell_locations[pivot_index])
                        if m <= distance and current_index != pivot_index:
                            # if it's closer than the last cell, update the closest magnitude and index
                            if m < closest_dist:
                                closest_index = current_index
                                closest_dist = m

                # check to make sure the initial cell doesn't slip through
                if closest_dist <= distance:
                    self.cell_closest_diff[pivot_index] = closest_index

    def jkr_neighbors(self):
        """ finds all pairs of cells that are overlapping
            by 0 or more and adds this to the running
            JKR graph.
        """
        # get the radius of search length
        distance = self.jkr_distance

        # provide an idea of the maximum number of neighbors for a cells
        max_neighbors = 15
        length = self.number_cells * max_neighbors
        edge_holder = np.zeros((length, 2), dtype=int)

        # call the parallel version if desired
        if self.parallel:
            # prevents the need for having the numba library if it's not installed
            import Parallel
            edge_holder = Parallel.jkr_neighbors_gpu(self, distance, edge_holder, max_neighbors)

        # call the boring non-parallel cpu version
        else:
            # a counter used to know where the next edge will be placed in the edges_holder
            edge_counter = 0

            # create an 3D array that will divide the space up into a collection of bins
            bins_size = self.size // distance + np.array([5, 5, 5])
            bins_size = tuple(bins_size.astype(int))
            bins = np.empty(bins_size, dtype=np.object)

            # each bin will be a numpy array that will hold indices of cells
            for i, j, k in itertools.product(range(bins_size[0]), range(bins_size[1]), range(bins_size[2])):
                bins[i][j][k] = np.array([], dtype=np.int)

            # loops over all cells appending their index value in the corresponding bin
            for pivot_index in range(self.number_cells):
                # offset the bin location by 1 to help when searching over bins and reduce potential error of cells
                # that may be slightly outside the space
                bin_location = self.cell_locations[pivot_index] // distance + np.array([2, 2, 2])
                bin_location = bin_location.astype(int)
                x, y, z = bin_location[0], bin_location[1], bin_location[2]

                # adds the cell to the corresponding bin
                bins[x][y][z] = np.append(bins[x][y][z], pivot_index)

                # loop over all nearby bins
                for i, j, k in itertools.product(range(-1, 2), repeat=3):
                    # get the array that is holding the indices of a cells in a block
                    indices_in_bin = bins[x + i][y + j][z + k]

                    # looks at the cells in a block and decides if they are neighbors
                    for l in range(len(indices_in_bin)):
                        # get the index of the current cell in question
                        current_index = indices_in_bin[l]

                        # get the magnitude of the vector between the cells
                        mag = np.linalg.norm(self.cell_locations[current_index] - self.cell_locations[pivot_index])

                        # compute the overlap of the cell space
                        overlap = self.cell_radii[current_index] + self.cell_radii[pivot_index] - mag

                        # if overlap present and not the same cell
                        if overlap >= 0 and pivot_index != current_index:
                            # update the edge array and increase the place for the next addition
                            edge_holder[edge_counter][0] = pivot_index
                            edge_holder[edge_counter][1] = current_index
                            edge_counter += 1

        # add the new edges and remove any duplicate edges or loops
        self.jkr_graph.add_edges(edge_holder)
        self.jkr_graph.simplify()

    def handle_movement(self):
        """ runs the following functions together for a
            given time amount. Resets the force and
            velocity arrays as well.
        """
        # get the total amount of times the cells will be incrementally moved during the step
        steps = int(self.time_step_value / self.move_time_step)

        # run the following functions consecutively for the given amount of steps
        for i in range(steps):
            # update the jkr neighbors
            self.jkr_neighbors()

            # calculate the forces acting on each cell
            self.get_forces()

            # turn the forces into movement
            self.apply_forces()

        # reset all forces back to zero vectors
        self.cell_motility_force = np.zeros((self.number_cells, 3))

    def get_forces(self):
        """ goes through all of the cells and quantifies any forces arising
            from adhesion or repulsion between the cells
        """
        # get the values for Youngs modulus and Poisson's ratio
        poisson = self.poisson
        youngs = self.youngs_mod
        adhesion_const = self.adhesion_const

        # get the updated edges of the jkr graph
        jkr_edges = self.jkr_graph.get_edgelist()
        delete_jkr_edges = np.zeros(len(jkr_edges), dtype=int)

        # call the parallel version if desired
        if self.parallel and len(jkr_edges) > 0:
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

                # hold the vector between the centers of the cells and the magnitude of this vector
                disp_vector = self.cell_locations[index_1] - self.cell_locations[index_2]
                magnitude = np.linalg.norm(disp_vector)

                if magnitude == 0:
                    normal = np.array([0.0, 0.0, 0.0])
                else:
                    normal = disp_vector / np.linalg.norm(disp_vector)

                # get the total overlap of the cells used later in calculations
                overlap = self.cell_radii[index_1] + self.cell_radii[index_2] - magnitude

                # gets two values used for JKR
                e_hat = (((1 - poisson ** 2) / youngs) + ((1 - poisson ** 2) / youngs)) ** -1
                r_hat = ((1 / self.cell_radii[index_1]) + (1 / self.cell_radii[index_2])) ** -1

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
                    self.cell_jkr_force[index_1] += jkr_force * normal
                    self.cell_jkr_force[index_2] -= jkr_force * normal

                # remove the edge if the it fails to meet the criteria for distance, JKR simulating that
                # the bond is broken
                else:
                    delete_jkr_edges[i] = i

        # update the jkr graph after the arrays have been updated by either the parallel or non-parallel function
        self.jkr_graph.delete_edges(delete_jkr_edges)

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
            for i in range(self.number_cells):
                # stokes law for velocity based on force and fluid viscosity
                stokes_friction = 6 * math.pi * self.viscosity * self.cell_radii[i]

                # update the velocity of the cell based on the solution
                velocity = (self.cell_motility_force[i] + self.cell_jkr_force[i]) / stokes_friction

                # set the possible new location
                new_location = self.cell_locations[i] + velocity * self.move_time_step

                # loops over all directions of space
                for j in range(0, 3):
                    # check if new location is in environment space if not simulation a collision with the bounds
                    if new_location[j] > self.size[j]:
                        self.cell_locations[i][j] = self.size[j]
                    elif new_location[j] < 0:
                        self.cell_locations[i][j] = 0.0
                    else:
                        self.cell_locations[i][j] = new_location[j]

                # reset the jkr forces back to zero
                self.cell_jkr_force[i] = np.array([0.0, 0.0, 0.0])
