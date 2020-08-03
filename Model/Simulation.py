"""

Here you can add or remove parameters and functions. The __init__ function of the Simulation class
 allows you to change the instance variables from either from the template file or just directly
 to the class. The template file acts as a user interface of sorts.

"""
import numpy as np
import random as r
import igraph
import math
import time
from numba import jit

import Input
import Functions


# used to hold all values necessary to the simulation as it moves from one step to the next
class Simulation:
    def __init__(self, template_location):
        # open the .txt template file that contains the initial parameters
        with open(template_location) as template_file:
            lines = template_file.readlines()

        # the following lines correspond to lines of template file, so anything implemented to the template file
        # can be added to the class initialization.
        # general parameters
        self.name = lines[8][2:-3]  # the name of the simulation, used to name files in the output directory
        self.output_direct = lines[11][2:-3]   # the output directory where simulation is placed
        self.parallel = eval(lines[14][2:-3]) # whether the model is using parallel GPU processing for certain functions
        self.size = np.array(eval(lines[17][2:-3]))  # the dimensions of the space (in meters) the cells reside in
        self.num_GATA6 = int(lines[20][2:-3])   # the number of GATA6 high cells to begin the simulation
        self.num_NANOG = int(lines[23][2:-3])   # the number of NANOG high cells to being the simulation

        # finite dynamical system
        self.functions = eval(lines[29][2:-3])  # the finite dynamical system functions as strings in an array
        self.num_fds_states = int(lines[32][2:-3])  # the number of states for the finite dynamical system

        # modes
        self.output_csvs = eval(lines[41][2:-3])  # whether or not to produce csvs with cell information
        self.output_images = eval(lines[38][2:-3])   # whether or not to produce images
        self.continuation = eval(lines[44][2:-3])   # continuation of a previous simulation
        self.csv_to_images = eval(lines[47][2:-3])  # turn a collection of csvs to images
        self.images_to_video = eval(lines[50][2:-3])    # turn a collection of images into a video

        # timing
        self.beginning_step = int(lines[57][2:-3])  # the step the simulation starts on, used for certain modes
        self.end_step = int(lines[60][2:-3])   # the last step of a simulation
        self.time_step_value = float(lines[63][2:-3])   # the real-time value of each step
        self.fds_thresh = int(lines[66][2:-3])  # the threshold (in steps) for updating the finite dynamical system
        self.pluri_div_thresh = int(lines[69][2:-3])  # the division threshold (in steps) of a pluripotent cell
        self.pluri_to_diff = int(lines[72][2:-3])  # the differentiation threshold (in steps) of a pluripotent cell
        self.diff_div_thresh = int(lines[75][2:-3])  # the division threshold (in steps) of a differentiated cell
        self.death_thresh = int(lines[78][2:-3])  # the death threshold (in steps) for cell death

        # intercellular
        self.neighbor_distance = float(lines[84][2:-3]) # the distance threshold for assigning nearby cells as neighbors
        self.nearest_distance = float(lines[87][2:-3])  # the radius of search for the nearest cell of desired type
        self.jkr_distance = float(lines[90][2:-3])  # the radius of search for JKR adhesive bonds formed between cells
        self.lonely_cell = int(lines[93][2:-3])  # if the number of neighbors is below this threshold, a cell is lonely
        self.contact_inhibit = int(lines[96][2:-3])  # if the number of neighbors is below this threshold, no inhibition
        self.move_thresh = int(lines[99][2:-3])  # if the number of neighbors is above this threshold, inhibit motion
        self.diff_surround = int(lines[102][2:-3])  # the number of diff cells needed to help induce differentiation

        # extracellular
        self.diffuse = float(lines[108][2:-3])  # the diffusion constant
        self.dx = eval(lines[111][2:-3])[0]  # the diffusion resolution along the x-axis
        self.dy = eval(lines[111][2:-3])[1]  # the diffusion resolution along the y-axis
        self.dz = eval(lines[111][2:-3])[2]  # the diffusion resolution along the z-axis

        # movement/physical
        self.move_time_step = float(lines[117][2:-3])  # the time step used for each time the JKR function
        self.adhesion_const = float(lines[120][2:-3])  # the adhesion constant between cells used for JKR contact
        self.viscosity = float(lines[123][2:-3])   # the viscosity of the medium used for stokes friction
        self.motility_force = float(lines[126][2:-3])   # the active force (in Newtons) of a cell actively moving
        self.max_radius = float(lines[129][2:-3])    # the maximum radius (in meters) of a cell

        # imaging
        self.image_quality = eval(lines[135][2:-3])    # the output image/video dimensions in pixels
        self.fps = float(lines[138][2:-3])   # the frames per second of the video produced
        self.background_color = eval(lines[141][2:-3])    # the background space color
        self.color_mode = eval(lines[144][2:-3])   # used to vary which method of coloring used
        self.pluri_color = eval(lines[147][2:-3])   # color of a pluripotent cell
        self.diff_color = eval(lines[150][2:-3])   # color of a differentiated cell
        self.pluri_gata6_high_color = eval(lines[153][2:-3])    # color of a pluripotent gata6 high cell
        self.pluri_nanog_high_color = eval(lines[156][2:-3])    # color of a pluripotent nanog high cell
        self.pluri_both_high_color = eval(lines[159][2:-3])    # color of a pluripotent gata6/nanog high cell

        # miscellaneous/experimental
        self.dox_step = int(lines[165][2:-3])  # the step at which dox is introduced into the simulation
        self.stochastic = eval(lines[168][2:-3])    # if initial fds variables are stochastic
        self.group = int(lines[171][2:-3])   # the number of cells introduced into or removed from the space at once
        self.guye_move = eval(lines[174][2:-3])    # whether or not to use the Guye method of cell motility
        self.diffuse_radius = float(lines[177][2:-3])   # the radius of search of diffusion points
        self.max_fgf4 = float(lines[180][2:-3])  # the maximum amount of fgf4 at a diffusion point
        self.eunbi_move = eval(lines[183][2:-3])    # use Eunbi's model for movement
        self.fgf4_move = eval(lines[186][2:-3])     # use FGF4 concentrations for NANOG high movements
        self.output_gradient = eval(lines[189][2:-3])   # output an image of the extracellular gradient

        # check that the name and path from the template are valid
        self.path = Input.check_name(self, template_location)

        # holds the current number of cells, step, and time when a step started (used for tracking efficiency)
        self.number_cells = 0
        self.current_step = self.beginning_step
        self.step_start = float()

        # these arrays hold all values of the cells, each index corresponds to a cell.
        self.cell_locations = np.empty((0, 3), dtype=float)    # holds every cell's location vector in the space
        self.cell_radii = np.empty((0, 1), dtype=float)     # holds every cell's radius
        self.cell_motion = np.empty((0, 1), dtype=bool)    # holds every cell's boolean for being in motion or not
        self.cell_fds = np.empty((0, 4), dtype=float)    # holds every cell's values for the fds, currently 4
        self.cell_states = np.empty((0, 1), dtype='<U14')    # holds every cell's state pluripotent or differentiated
        self.cell_diff_counter = np.empty((0, 1), dtype=int)    # holds every cell's differentiation counter
        self.cell_div_counter = np.empty((0, 1), dtype=int)    # holds every cell's division counter
        self.cell_death_counter = np.empty((0, 1), dtype=int)    # holds every cell's death counter
        self.cell_fds_counter = np.empty((0, 1), dtype=int)    # holds every cell's finite dynamical system counter
        self.cell_motility_force = np.empty((0, 3), dtype=float)    # holds every cell's motility force vector
        self.cell_jkr_force = np.empty((0, 3), dtype=float)    # holds every cell's JKR force vector
        self.cell_nearest_gata6 = np.empty((0, 1))  # holds index of nearest gata6 high neighbor
        self.cell_nearest_nanog = np.empty((0, 1))  # holds index of nearest nanog high neighbor
        self.cell_nearest_diff = np.empty((0, 1))  # holds index of nearest differentiated neighbor
        self.cell_highest_fgf4 = np.empty((0, 3))  # holds the location of highest fgf4 point

        # holds the run time for key functions to track efficiency. each step these are outputted to the CSV file.
        self.update_diffusion_time = float()
        self.check_neighbors_time = float()
        self.nearest_diff_time = float()
        self.cell_motility_time = float()
        self.cell_update_time = float()
        self.update_queue_time = float()
        self.handle_movement_time = float()
        self.jkr_neighbors_time = float()
        self.get_forces_time = float()
        self.apply_forces_time = float()

        # neighbor graph is used to locate cells that are in close proximity, while the JKR graph holds adhesion bonds
        # between cells that are either currently overlapping or still maintain an adhesive bond
        self.neighbor_graph, self.jkr_graph = igraph.Graph(), igraph.Graph()

        # squaring the approximation of the differential
        self.dx2, self.dy2, self.dz2 = self.dx ** 2, self.dy ** 2, self.dz ** 2

        # get the time step value for diffusion updates depending on whether 2D or 3D
        if self.size[2] == 0:
            self.dt = (self.dx2 * self.dy2) / (2 * self.diffuse * (self.dx2 + self.dy2))
        else:
            self.dt = (self.dx2 * self.dy2 * self.dz2) / (2 * self.diffuse * (self.dx2 + self.dy2 + self.dz2))

        # the points at which the diffusion values are calculated
        x_steps = (int(self.size[0] / self.dx) + 1)
        y_steps = (int(self.size[1] / self.dy) + 1)
        z_steps = (int(self.size[2] / self.dz) + 1)
        self.fgf4_values = np.zeros((x_steps, y_steps, z_steps))

        # holds all indices of cells that will divide at a current step or be removed at that step
        self.cells_to_divide, self.cells_to_remove = np.empty((0, 1), dtype=int), np.empty((0, 1), dtype=int)

        # min and max radius lengths are used to calculate linear growth of the radius over time in 2D
        self.min_radius = self.max_radius / 2 ** 0.5
        self.pluri_growth = (self.max_radius - self.min_radius) / self.pluri_div_thresh
        self.diff_growth = (self.max_radius - self.min_radius) / self.diff_div_thresh

        # Youngs modulus and Poisson's ratio used for JKR contact
        self.youngs_mod, self.poisson = 1000, 0.5

        # the csv and video objects that will be updated each step
        self.csv_object = object()
        self.video_object = object()

        # given all of the above parameters, run the corresponding mode
        Input.setup_simulation(self)

    def info(self):
        """ Gives an idea of how the simulation is running
            and records the beginning of the step in real time
        """
        # records the real time value of when a step begins
        self.step_start = time.time()

        # prints the current step number and the number of cells
        print("Step: " + str(self.current_step))
        print("Number of cells: " + str(self.number_cells))

    def setup_diffusion_bins(self):
        """ This function will put the diffusion points
            into corresponding bins that will be used to
            find values of diffusion within a radius
        """
        # reduce length of variable name for ease of writing
        steps = self.fgf4_values.shape

        # set up the locations of the diffusion points
        x, y, z = np.meshgrid(np.arange(steps[0]), np.arange(steps[1]), np.arange(steps[2]), indexing='ij')
        x, y, z = x * self.dx, y * self.dy, z * self.dz
        self.diffuse_locations = np.stack((x, y, z), axis=3)

        # set up the appropriate size for the diffusion bins and the help array, 100 is the bin limit
        bins_size = self.size // self.diffuse_radius + np.array([5, 5, 5])
        bins_size_help = tuple(bins_size.astype(int))
        bins_size = np.append(bins_size, [100, 3])
        bins_size = tuple(bins_size.astype(int))
        self.diffuse_bins = np.empty(bins_size, dtype=int)
        self.diffuse_bins_help = np.zeros(bins_size_help, dtype=int)

        # loop over all diffusion points
        for i in range(0, steps[0]):
            for j in range(0, steps[1]):
                for k in range(0, steps[2]):
                    # get the location in the bin array
                    bin_location = self.diffuse_locations[i][j][k] // self.diffuse_radius + np.array([2, 2, 2])
                    x, y, z = int(bin_location[0]), int(bin_location[1]), int(bin_location[2])

                    # get the index of the where the point will be added
                    place = self.diffuse_bins_help[x][y][z]

                    # add the diffusion point to a corresponding bin and increase the place index
                    self.diffuse_bins[x][y][z][place] = np.array([i, j, k])
                    self.diffuse_bins_help[x][y][z] += 1

    def update_diffusion(self):
        """ calls update_diffusion_cpu for all specified
            extracellular gradients, which will update
            the diffusion of the extracellular molecule
        """
        # start time
        self.update_diffusion_time = -1 * time.time()

        # list the gradients that need to be updated and get the number of time steps for the diffusion calculation
        gradients_to_update = [self.fgf4_values]
        time_steps = int(self.time_step_value / self.dt)

        # go through all gradients and update the diffusion of each
        for i in range(len(gradients_to_update)):
            gradients_to_update[i] = update_diffusion_cpu(gradients_to_update[i], time_steps, self.dt, self.dx2,
                                                          self.dy2, self.dz2, self.diffuse, self.size)
        # end time
        self.update_diffusion_time += time.time()

    def highest_fgf4(self):
        """ Search for the highest concentration of
            fgf4 within a fixed radius
        """
        self.cell_highest_fgf4 = highest_fgf4_cpu(self.diffuse_radius, self.diffuse_bins, self.diffuse_bins_help,
                                                  self.diffuse_locations, self.cell_locations, self.number_cells,
                                                  self.cell_highest_fgf4, self.fgf4_values)

    def add_cell(self, location, radius, motion, fds, state, diff_counter, div_counter, death_counter, fds_counter,
                 motility_force, jkr_force, nearest_gata6, nearest_nanog, nearest_diff, highest_fgf4):
        """ Adds each of the new cell's values to
            the array holders, graphs, and total
            number of cells.
        """
        # adds the cell to the arrays holding the cell values, the 2D arrays have to be handled a bit differently as
        # axis=0 has to be provided and the appended array should also be of the same shape with additional brackets
        self.cell_locations = np.append(self.cell_locations, [location], axis=0)
        self.cell_radii = np.append(self.cell_radii, radius)
        self.cell_motion = np.append(self.cell_motion, motion)
        self.cell_fds = np.append(self.cell_fds, [fds], axis=0)
        self.cell_states = np.append(self.cell_states, state)
        self.cell_diff_counter = np.append(self.cell_diff_counter, diff_counter)
        self.cell_div_counter = np.append(self.cell_div_counter, div_counter)
        self.cell_death_counter = np.append(self.cell_death_counter, death_counter)
        self.cell_fds_counter = np.append(self.cell_fds_counter, fds_counter)
        self.cell_motility_force = np.append(self.cell_motility_force, [motility_force], axis=0)
        self.cell_jkr_force = np.append(self.cell_jkr_force, [jkr_force], axis=0)
        self.cell_nearest_gata6 = np.append(self.cell_nearest_gata6, nearest_gata6)
        self.cell_nearest_nanog = np.append(self.cell_nearest_nanog, nearest_nanog)
        self.cell_nearest_diff = np.append(self.cell_nearest_diff, nearest_diff)
        self.cell_highest_fgf4 = np.append(self.cell_highest_fgf4, [highest_fgf4], axis=0)

        # add it to the following graphs, this is done implicitly by increasing the length of the vertex list by
        # one, which the indices directly correspond to the cell holder arrays
        self.neighbor_graph.add_vertex()
        self.jkr_graph.add_vertex()

        # revalue the total number of cells
        self.number_cells += 1

    def remove_cell(self, index):
        """ Given the index of a cell to remove,
            this will remove that from each array,
            graphs, and total number of cells
        """
        # delete the index of each holder array, the 2D arrays require the axis=0 parameter to essentially delete
        # that row of the matrix much like when appending a new row
        self.cell_locations = np.delete(self.cell_locations, index, axis=0)
        self.cell_radii = np.delete(self.cell_radii, index)
        self.cell_motion = np.delete(self.cell_motion, index)
        self.cell_fds = np.delete(self.cell_fds, index, axis=0)
        self.cell_states = np.delete(self.cell_states, index)
        self.cell_diff_counter = np.delete(self.cell_diff_counter, index)
        self.cell_div_counter = np.delete(self.cell_div_counter, index)
        self.cell_death_counter = np.delete(self.cell_death_counter, index)
        self.cell_fds_counter = np.delete(self.cell_fds_counter, index)
        self.cell_motility_force = np.delete(self.cell_motility_force, index, axis=0)
        self.cell_jkr_force = np.delete(self.cell_jkr_force, index, axis=0)
        self.cell_nearest_gata6 = np.delete(self.cell_nearest_gata6, index)
        self.cell_nearest_nanog = np.delete(self.cell_nearest_nanog, index)
        self.cell_nearest_diff = np.delete(self.cell_nearest_diff, index)
        self.cell_highest_fgf4 = np.delete(self.cell_highest_fgf4, index, axis=0)

        # remove the particular index from the following graphs as these deal in terms of indices
        # this will adjust edges as the indices change, so no worries here
        self.neighbor_graph.delete_vertices(index)
        self.jkr_graph.delete_vertices(index)

        # revalue the number of cells
        self.number_cells -= 1

    def update_queue(self):
        """ Introduces and removes cells into the simulation.
            This also provides control into how many cells
            are added/removed at a time.
        """
        # start time
        self.update_queue_time = -1 * time.time()

        # give the user an idea of how many cells are being added/removed during a given step
        print("Adding " + str(len(self.cells_to_divide)) + " cells...")
        print("Removing " + str(len(self.cells_to_remove)) + " cells...")

        # loops over all indices that are set to divide
        for i in range(len(self.cells_to_divide)):
            self.divide(self.cells_to_divide[i])

            # Cannot add all of the new cells, otherwise several cells are likely to be added in
            #   close proximity to each other at later time steps. Such addition, coupled with
            #   handling collisions, make give rise to sudden changes in overall positions of
            #   cells within the simulation. Instead, collisions are handled after 'group' number
            #   of cells are added.

            # if self.group is equal to 0, all will be added in at once
            if self.group != 0:
                if (i + 1) % self.group == 0:
                    # call the handle movement function, which should reduce cell overlap especially with high density
                    Functions.handle_movement(self)

        # loops over all indices that are set to be removed
        for i in range(len(self.cells_to_remove)):
            # record the index
            index = self.cells_to_remove[i]
            self.remove_cell(index)

            # adjusts the indices as deleting part of the array may change the correct indices to remove
            for j in range(i + 1, len(self.cells_to_remove)):
                # if the current cell being deleted falls after the index, shift the indices by 1
                if index < self.cells_to_remove[j]:
                    self.cells_to_remove[j] -= 1

            # if self.group is equal to 0, all will be removed at once
            if self.group != 0:
                if (i + 1) % self.group == 0:
                    # call the handle movement function, which should reduce cell overlap especially with high density
                    Functions.handle_movement(self)

        # clear the arrays for the next step
        self.cells_to_divide = np.empty((0, 1), dtype=int)
        self.cells_to_remove = np.empty((0, 1), dtype=int)

        # end time
        self.update_queue_time += time.time()

    def divide(self, index):
        """ Takes a cell or rather an index in the holder
            arrays and adds a new cell (index). This also
            updates factors such as size and counters.
        """
        # move the cells to positions that are representative of the new locations of daughter cells
        division_position = self.random_vector() * (self.max_radius - self.min_radius)
        self.cell_locations[index] += division_position
        location = self.cell_locations[index] - 2 * division_position

        # reduce radius to minimum size, set the division counter to zero, and None as the nearest differentiated cell
        self.cell_radii[index] = radius = self.min_radius
        self.cell_div_counter[index] = div_counter = 0
        self.cell_death_counter[index] = death_counter = 0
        self.cell_nearest_gata6[index] = nearest_gata6 = np.nan
        self.cell_nearest_nanog[index] = nearest_nanog = np.nan
        self.cell_nearest_diff[index] = nearest_diff = np.nan
        self.cell_highest_fgf4[index] = highest_fgf4 = np.array([np.nan, np.nan, np.nan])

        # keep identical values for motion, booleans, state, differentiation, and boolean update
        motion = self.cell_motion[index]
        booleans = self.cell_fds[index]
        state = self.cell_states[index]
        diff_counter = self.cell_diff_counter[index]
        bool_counter = self.cell_fds_counter[index]

        # set the force vector to zero
        motility_force = np.zeros(3, dtype=float)
        jkr_force = np.zeros(3, dtype=float)

        # add the cell to the simulation
        self.add_cell(location, radius, motion, booleans, state, diff_counter, div_counter, death_counter,
                      bool_counter, motility_force, jkr_force, nearest_gata6, nearest_nanog, nearest_diff, highest_fgf4)

    def cell_update(self):
        """ Loops over all indices of cells and updates
            their values accordingly.
        """
        # start time
        self.cell_update_time = -1 * time.time()

        # loop over the cells
        for i in range(self.number_cells):
            # Cell death
            # checks to see if cell is pluripotent
            if self.cell_states[i] == "Pluripotent":
                # looks at the neighbors and counts them, increasing the death counter if not enough neighbors
                neighbors = self.neighbor_graph.neighbors(i)
                if len(neighbors) < self.lonely_cell:
                    self.cell_death_counter[i] += 1
                # if not reset the death counter back to zero
                else:
                    self.cell_death_counter[i] = 0

                # removes cell if it meets the parameters
                if self.cell_death_counter[i] >= self.death_thresh:
                    self.cells_to_remove = np.append(self.cells_to_remove, i)

            # Differentiated surround
            # checks to see if cell is pluripotent and GATA6 low
            if self.cell_states[i] == "Pluripotent" and self.cell_fds[i][2] == 0:
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
                        self.cell_diff_counter[i] += r.randint(0, 2)
                        break

            # Growth
            # increase the cell radius based on the state and whether or not it has reached the max size
            if self.cell_radii[i] < self.max_radius:
                # pluripotent growth
                if self.cell_states[i] == "Pluripotent":
                    self.cell_radii[i] += self.pluri_growth
                # differentiated growth
                else:
                    self.cell_radii[i] += self.diff_growth

            # Division
            # checks to see if the non-moving cell should divide or increase its division counter
            if not self.cell_motion[i]:
                # if it's a differentiated cell, also check for contact inhibition
                if self.cell_states[i] == "Differentiated" and self.cell_div_counter[i] >= self.diff_div_thresh:
                    neighbors = self.neighbor_graph.neighbors(i)
                    if len(neighbors) < self.contact_inhibit:
                        self.cells_to_divide = np.append(self.cells_to_divide, i)

                # no contact inhibition for pluripotent cells
                elif self.cell_states[i] == "Pluripotent" and self.cell_div_counter[i] >= self.pluri_div_thresh:
                    self.cells_to_divide = np.append(self.cells_to_divide, i)

                # stochastically increase the division counter by either 0, 1, or 2 if nothing else
                else:
                    self.cell_div_counter[i] += r.randint(0, 2)

            # Extracellular interaction and GATA6 pathway
            # take the location of a cell and determine the nearest diffusion point by creating a zone around a
            # diffusion point an any cells in the zone will base their value off of that
            half_index_x = self.cell_locations[i][0] // (self.dx / 2)
            half_index_y = self.cell_locations[i][1] // (self.dy / 2)
            half_index_z = self.cell_locations[i][2] // (self.dz / 2)
            index_x = math.ceil(half_index_x / 2)
            index_y = math.ceil(half_index_y / 2)
            index_z = math.ceil(half_index_z / 2)

            # if the diffusion point value is less than the max FGF4 it can hold and the cell is NANOG high
            # increase the FGF4 value by 1
            if self.cell_fds[i][3] == 1 and self.fgf4_values[index_x][index_y][index_z] < self.max_fgf4:
                self.fgf4_values[index_x][index_y][index_z] += 1

            # activate the following pathway based on if dox has been induced yet
            if self.current_step >= self.dox_step:
                # if the FGF4 amount for the location is greater than 0, set the fgf4_bool value to be 1 for the
                # functions
                if self.fgf4_values[index_x][index_y][index_z] > 0:
                    fgf4_bool = 1
                else:
                    fgf4_bool = 0

                # Finite dynamical system and state change
                # temporarily hold the FGFR value
                temp_fgfr = self.cell_fds[i][0]

                # only update the booleans when the counter matches the boolean update rate
                if self.cell_fds_counter[i] % self.fds_thresh == 0:
                    # xn is equal to the value corresponding to its function
                    x1 = fgf4_bool
                    x2 = self.cell_fds[i][0]
                    x3 = self.cell_fds[i][1]
                    x4 = self.cell_fds[i][2]
                    x5 = self.cell_fds[i][3]

                    # evaluate the functions by turning them from strings to equations
                    new_fgf4 = eval(self.functions[0]) % self.num_fds_states
                    new_fgfr = eval(self.functions[1]) % self.num_fds_states
                    new_erk = eval(self.functions[2]) % self.num_fds_states
                    new_gata6 = eval(self.functions[3]) % self.num_fds_states
                    new_nanog = eval(self.functions[4]) % self.num_fds_states

                    # updates self.booleans with the new boolean values
                    self.cell_fds[i] = np.array([new_fgfr, new_erk, new_gata6, new_nanog])

                # if no fds update, maintain the same fgf4 boolean value
                else:
                    new_fgf4 = fgf4_bool

                # increase the finite dynamical system counter
                self.cell_fds_counter[i] += 1

                # if the temporary FGFR value is 0 and the FGF4 value is 1 decrease the amount of FGF4 by 1
                # this simulates FGFR using FGF4
                if temp_fgfr == 0 and new_fgf4 == 1:
                    if self.fgf4_values[index_x][index_y][index_z] > 1:
                        self.fgf4_values[index_x][index_y][index_z] -= 1

                # if the cell is GATA6 high and pluripotent increase the differentiation counter by 1
                if self.cell_fds[i][2] == 1 and self.cell_states[i] == "Pluripotent":
                    self.cell_diff_counter[i] += r.randint(0, 2)

                    # if the differentiation counter is greater than the threshold, differentiate
                    if self.cell_diff_counter[i] >= self.pluri_to_diff:
                        # change the state to differentiated
                        self.cell_states[i] = "Differentiated"

                        # make sure NANOG is low or rather 0
                        self.cell_fds[i][3] = 0

                        # allow the cell to actively move again
                        self.cell_motion[i] = True
        # end time
        self.cell_update_time += time.time()

    def cell_motility(self):
        """ Gives the cells a motive force, depending on
            set rules for the cell types.
        """
        # start time
        self.cell_motility_time = -1 * time.time()

        # loop over all of the cells
        for i in range(self.number_cells):
            # set motion to false if the cell is surrounded by many neighbors
            neighbors = self.neighbor_graph.neighbors(i)
            if len(neighbors) >= self.move_thresh:
                self.cell_motion[i] = False

            # check whether differentiated or pluripotent
            if self.cell_states[i] == "Differentiated":
                # this will move differentiated cells together
                if 0 < len(neighbors):
                    # create a vector to hold the sum of normal vectors between a cell and its neighbors
                    vector_holder = np.array([0.0, 0.0, 0.0])

                    # loop over the neighbors getting the normal and adding to the holder
                    for j in range(len(neighbors)):
                        if self.cell_states[neighbors[j]] == "Differentiated":
                            vector = self.cell_locations[neighbors[j]] - self.cell_locations[i]
                            vector_holder += vector

                    # get the normal vector
                    normal = normal_vector(vector_holder)

                    # move in direction of the differentiated cells
                    self.cell_motility_force[i] += self.motility_force * normal * 0.5

                if 0 < len(neighbors):
                    # create a vector to hold the sum of normal vectors between a cell and its neighbors
                    vector_holder = np.array([0.0, 0.0, 0.0])

                    # loop over the neighbors getting the normal and adding to the holder
                    for j in range(len(neighbors)):
                        if self.cell_states[neighbors[j]] == "Pluripotent":
                            vector = self.cell_locations[neighbors[j]] - self.cell_locations[i]
                            vector_holder += vector

                    # get the normal vector
                    normal = normal_vector(vector_holder)

                    # move in direction opposite to pluripotent cells
                    self.cell_motility_force[i] += self.motility_force * normal * -1 * 0.5

                # if there aren't any neighbors and still in motion then move randomly
                if self.cell_motion[i]:
                    # move based on Eunbi's model
                    if self.eunbi_move:
                        # if there is a nearby nanog cell, move away from it
                        if not np.isnan(self.cell_nearest_nanog[i]):
                            nearest_index = int(self.cell_nearest_nanog[i])
                            normal = normal_vector(self.cell_locations[i] - self.cell_locations[nearest_index])
                            self.cell_motility_force[i] += normal * self.motility_force * -1

                        # move randomly instead
                        else:
                            self.cell_motility_force[i] += self.random_vector() * self.motility_force

                    # move randomly instead
                    else:
                        self.cell_motility_force[i] += self.random_vector() * self.motility_force

            # for pluripotent cells
            else:
                # apply movement if the cell is "in motion"
                if self.cell_motion[i]:
                    # GATA6 high cell
                    if self.cell_fds[i][2] == 1:
                        # continue if using Guye et al. movement and if there exists differentiated cells
                        if self.guye_move and not np.isnan(self.cell_nearest_diff[i]):
                            # get the differentiated neighbors
                            guye_neighbor = self.cell_nearest_diff[i]

                            # get the normal vector
                            normal = normal_vector(self.cell_locations[guye_neighbor] - self.cell_locations[i])

                            # calculate the motility force
                            self.cell_motility_force[i] += normal * self.motility_force

                    # NANOG high cell
                    elif self.cell_fds[i][3] == 1:
                        # move based on fgf4 concentrations
                        if self.fgf4_move:
                            # makes sure not the numpy nan type, proceed if actual value
                            if (np.isnan(self.cell_highest_fgf4[i]) == np.zeros(3, dtype=bool)).all():
                                # get the location of the diffusion point and move toward it
                                x = int(self.cell_highest_fgf4[i][0])
                                y = int(self.cell_highest_fgf4[i][1])
                                z = int(self.cell_highest_fgf4[i][2])
                                normal = normal_vector(self.cell_locations[i] - self.diffuse_locations[x][y][z])
                                self.cell_motility_force[i] += normal * self.motility_force

                        # move based on Eunbi's model
                        elif self.eunbi_move:
                            # if there is a gata6 high cell nearby, move away from it
                            if not np.isnan(self.cell_nearest_gata6[i]):
                                nearest_index = int(self.cell_nearest_gata6[i])
                                normal = normal_vector(self.cell_locations[nearest_index] - self.cell_locations[i])
                                self.cell_motility_force[i] += normal * self.motility_force * -1

                            # if there is a nanog high cell nearby, move to it
                            elif not np.isnan(self.cell_nearest_nanog[i]):
                                nearest_index = int(self.cell_nearest_nanog[i])
                                normal = normal_vector(self.cell_locations[nearest_index] - self.cell_locations[i])
                                self.cell_motility_force[i] += normal * self.motility_force

                            # if nothing else, move randomly
                            else:
                                self.cell_motility_force[i] += self.random_vector() * self.motility_force
                        # if nothing else, move randomly
                        else:
                            self.cell_motility_force[i] += self.random_vector() * self.motility_force
                    # if nothing else, move randomly
                    else:
                        self.cell_motility_force[i] += self.random_vector() * self.motility_force
        # end time
        self.cell_motility_time += time.time()

    def nearest(self):
        """ looks at cells within a given radius
            a determines the closest cells of
            all types.
        """
        # start time
        self.nearest_diff_time = -1 * time.time()

        # divides the space into bins and gives a holder of fixed size for each bin, the addition of 5 offsets
        # the space to prevent any errors, and 100 is the max cells for a bin which can be changed given errors
        bins_size = self.size // self.nearest_distance + np.array([5, 5, 5])
        bins_size_help = tuple(bins_size.astype(int))
        bins_size = np.append(bins_size, 100)
        bins_size = tuple(bins_size.astype(int))

        # an empty array used to represent the bins the cells are put into
        bins = np.empty(bins_size, dtype=int)

        # an array used to accelerate the function by eliminating the lookup for number of cells in a bin
        bins_help = np.zeros(bins_size_help, dtype=int)

        # find the nearest cell of each type with the external method, no gpu function yet
        new_gata6, new_nanog, new_diff = nearest_cpu(self.number_cells, self.nearest_distance, bins, bins_help,
                                                     self.cell_locations, self.cell_nearest_gata6,
                                                     self.cell_nearest_nanog, self.cell_nearest_diff, self.cell_states,
                                                     self.cell_fds)

        # revalue the array holding the indices of nearest cells of given type
        self.cell_nearest_gata6 = new_gata6
        self.cell_nearest_nanog = new_nanog
        self.cell_nearest_diff = new_diff

        # end time
        self.nearest_diff_time += time.time()

    def random_vector(self):
        """ Computes a random point on a unit sphere centered at the origin
            Returns - point [x,y,z]
        """
        # a random angle on the cell
        theta = r.random() * 2 * math.pi

        # determine if simulation is 2D or 3D
        if self.size[2] == 0:
            # 2D vector
            x, y, z = math.cos(theta), math.sin(theta), 0.0

        else:
            # 3D vector
            phi = r.random() * 2 * math.pi
            radius = math.cos(phi)
            x, y, z = radius * math.cos(theta), radius * math.sin(theta), math.sin(phi)

        return np.array([x, y, z])


def normal_vector(vector):
    """ Return the normal vector
    """
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return np.array([0, 0, 0])
    else:
        return vector / magnitude


@jit(nopython=True)
def highest_fgf4_cpu(diffuse_radius, diffuse_bins, diffuse_bins_help, diffuse_locations, cell_locations, number_cells,
                     highest_fgf4, fgf4_values):
    """ This is the Numba optimized version of
        the highest_fgf4 function.
    """
    for pivot_index in range(number_cells):
        # offset bins by 2 to avoid missing cells
        block_location = cell_locations[pivot_index] // diffuse_radius + np.array([2, 2, 2])
        x, y, z = int(block_location[0]), int(block_location[1]), int(block_location[2])

        # create an initial value to check for the highest fgf4 point in a radius
        highest_index_x = np.nan
        highest_index_y = np.nan
        highest_index_z = np.nan
        highest_value = 0

        # loop over the bins that surround the current bin
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of cells in a bin
                    bin_count = diffuse_bins_help[x + i][y + j][z + k]

                    # go through the bin determining if a cell is a neighbor
                    for l in range(bin_count):
                        # get the index of the current cell in question
                        x_ = int(diffuse_bins[x + i][y + j][z + k][l][0])
                        y_ = int(diffuse_bins[x + i][y + j][z + k][l][1])
                        z_ = int(diffuse_bins[x + i][y + j][z + k][l][2])

                        # check to see if that cell is within the search radius and not the same cell
                        m = np.linalg.norm(diffuse_locations[x_][y_][z_] - cell_locations[pivot_index])
                        if m < diffuse_radius:
                            if fgf4_values[x_ + 1][y_ + 1][z_] > highest_value:
                                highest_index_x = x_
                                highest_index_y = y_
                                highest_index_z = z_
                                highest_value = fgf4_values[x_ + 1][y_ + 1][z_]

        # update the highest fgf4 diffusion point
        highest_fgf4[pivot_index][0] = highest_index_x
        highest_fgf4[pivot_index][1] = highest_index_y
        highest_fgf4[pivot_index][2] = highest_index_z

    # return the array back
    return highest_fgf4


@jit(nopython=True)
def nearest_cpu(number_cells, distance, bins, bins_help, cell_locations, nearest_gata6, nearest_nanog, nearest_diff,
                cell_states, cell_fds):
    """ This is the Numba optimized
        version of the nearest function.
    """
    for i in range(number_cells):
        # offset bins by 2 to avoid missing cells
        block_location = cell_locations[i] // distance + np.array([2, 2, 2])
        x, y, z = int(block_location[0]), int(block_location[1]), int(block_location[2])

        # use the help array to place the cells in corresponding bins
        place = bins_help[x][y][z]

        # gives the cell's array location
        bins[x][y][z][place] = i

        # updates the total amount cells in a bin
        bins_help[x][y][z] += 1

    # loops over all cells, with the current cell being the pivot of the search method
    for focus in range(number_cells):
        # offset bins by 2 to avoid missing cells
        block_location = cell_locations[focus] // distance + np.array([2, 2, 2])
        x, y, z = int(block_location[0]), int(block_location[1]), int(block_location[2])

        # initialize these variables with essentially nothing values and the distance as an initial comparison
        nearest_gata6_index, nearest_nanog_index, nearest_diff_index = np.nan, np.nan, np.nan
        nearest_gata6_dist, nearest_nanog_dist, nearest_diff_dist = distance * 2, distance * 2, distance * 2

        # loop over the bins that surround the current bin
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of cells in a bin
                    bin_count = bins_help[x + i][y + j][z + k]

                    # go through the bin determining if a cell is a neighbor
                    for l in range(bin_count):
                        # get the index of the current cell in question
                        current = int(bins[x + i][y + j][z + k][l])

                        # check to see if that cell is within the search radius and not the same cell
                        magnitude = np.linalg.norm(cell_locations[current] - cell_locations[focus])
                        if magnitude <= distance and focus != current:
                            # update the nearest gata6 high cell
                            if cell_fds[current][2] == 1:
                                # if it's closer than the last cell, update the closest magnitude and index
                                if magnitude < nearest_gata6_dist:
                                    nearest_gata6_index = current
                                    nearest_gata6_dist = magnitude

                            # update the nearest nanog high cell
                            elif cell_fds[current][3] == 1:
                                # if it's closer than the last cell, update the closest magnitude and index
                                if magnitude < nearest_nanog_dist:
                                    nearest_nanog_index = current
                                    nearest_nanog_dist = magnitude

                            # update the nearest differentiated cell
                            elif cell_states[current] == "Differentiated":
                                # if it's closer than the last cell, update the closest magnitude and index
                                if magnitude < nearest_diff_dist:
                                    nearest_diff_index = current
                                    nearest_diff_dist = magnitude

        # update the nearest cell of desired type index
        nearest_gata6[focus] = nearest_gata6_index
        nearest_nanog[focus] = nearest_nanog_index
        nearest_diff[focus] = nearest_diff_index

    # return the updated edges
    return nearest_gata6, nearest_nanog, nearest_diff


@jit(nopython=True)
def update_diffusion_cpu(gradient, time_steps, dt, dx2, dy2, dz2, diffuse, size):
    """ This is the Numba optimized version of
        the update_diffusion function that runs
        solely on the cpu.
    """
    # finite differences to solve the 2D Laplacian
    if size[2] == 0:
        for i in range(time_steps):
            # perform the first part of the calculation
            x = (gradient[2:, 1:-1] - 2 * gradient[1:-1, 1:-1] + gradient[:-2, 1:-1]) / dx2
            y = (gradient[1:-1, 2:] - 2 * gradient[1:-1, 1:-1] + gradient[1:-1, :-2]) / dy2

            # update the gradient array
            gradient[1:-1, 1:-1] = gradient[1:-1, 1:-1] + diffuse * dt * (x + y)

    # finite differences to solve the 3D Laplacian
    else:
        for i in range(time_steps):
            # perform the first part of the calculation
            x = (gradient[2:, 1:-1, 1:-1] - 2 * gradient[1:-1, 1:-1, 1:-1] + gradient[:-2, 1:-1, 1:-1]) / dx2
            y = (gradient[1:-1, 2:, 1:-1] - 2 * gradient[1:-1, 1:-1, 1:-1] + gradient[1:-1, :-2, 1:-1]) / dy2
            z = (gradient[1:-1, 1:-1, 2:] - 2 * gradient[1:-1, 1:-1, 1:-1] + gradient[1:-1, 1:-1, :-2]) / dz2

            # update the gradient array
            gradient[1:-1, 1:-1, 1:-1] = gradient[1:-1, 1:-1, 1:-1] + diffuse * dt * (x + y + z)

    # return the gradient back to the simulation
    return gradient
