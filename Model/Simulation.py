import numpy as np
import random as r
import igraph
import math
import time
from numba import jit, prange

import Parallel


# used to hold all values necessary to the simulation as it moves from one step to the next
class Simulation:
    def __init__(self, path, name, parallel, size, resolution, num_fds_states, functions, neighbor_distance,
                 nearest_distance, jkr_distance, lonely_cell, contact_inhibit, move_thresh, time_step_value,
                 beginning_step, end_step, move_time_step, dox_step, pluri_div_thresh, pluri_to_diff, diff_div_thresh,
                 fds_thresh, death_thresh, diff_surround, adhesion_const, viscosity, group, output_csvs, output_images,
                 image_quality, fps, background_color, bound_color, color_mode, pluri_color, diff_color,
                 pluri_gata6_high_color, pluri_nanog_high_color, pluri_both_high_color, guye_move, motility_force,
                 max_radius, fgf4_negative_move, random_move):

        # the following instance variables should remain fixed, meaning that they don't change from step to step.
        # they are merely used to hold initial parameters from the template file that will needed to be consistent
        # throughout the simulation
        self.path = path    # the directory of where the simulation will output to
        self.name = name    # the name of the simulation, used to name files in the output directory
        self.parallel = parallel    # whether the model is using parallel GPU processing for certain functions
        self.size = size    # the dimensions of the space (in meters) the cells reside in
        self.resolution = resolution    # the diffusion resolution of the space
        self.num_fds_states = num_fds_states    # the number of states for the finite dynamical system
        self.functions = functions    # the finite dynamical system functions as strings in an array
        self.neighbor_distance = neighbor_distance    # the distance threshold for assigning nearby cells as neighbors
        self.nearest_distance = nearest_distance    # the radius of search for the nearest cell of desired type
        self.jkr_distance = jkr_distance    # the radius of search for JKR adhesive bonds formed between cells
        self.lonely_cell = lonely_cell    # if the number of neighbors is below this threshold, a cell is lonely
        self.contact_inhibit = contact_inhibit    # if the number of neighbors is below this threshold, no inhibition
        self.move_thresh = move_thresh    # if the number of neighbors is above this threshold, motion is inhibited
        self.time_step_value = time_step_value    # the real-time value of each step
        self.beginning_step = beginning_step    # the step the simulation starts on, used for certain modes
        self.end_step = end_step    # the last step of a simulation
        self.move_time_step = move_time_step    # the time step used for each time the JKR contact function is used
        self.dox_step = dox_step    # the step at which dox is introduced into the simulation
        self.pluri_div_thresh = pluri_div_thresh    # the division threshold (in steps) of a pluripotent cell
        self.pluri_to_diff = pluri_to_diff    # the differentiation threshold (in steps) of a pluripotent cell
        self.diff_div_thresh = diff_div_thresh    # the division threshold (in steps) of a differentiated cell
        self.fds_thresh = fds_thresh    # the threshold (in steps) for updating the finite dynamical system
        self.death_thresh = death_thresh    # the death threshold (in steps) for cell death
        self.diff_surround = diff_surround    # the number of differentiated cells needed to help induce differentiation
        self.adhesion_const = adhesion_const    # the adhesion constant between cells used for JKR contact
        self.viscosity = viscosity    # the viscosity of the medium used for stokes friction
        self.group = group    # the number of cells introduced into or removed from the space at once
        self.output_csvs = output_csvs    # whether or not to produce csvs with cell information
        self.output_images = output_images    # whether or not to produce images
        self.image_quality = image_quality    # the output image/video dimensions in pixels
        self.fps = fps    # the frames per second of the video produced
        self.background_color = background_color    # the background space color
        self.bound_color = bound_color    # the color of the bounds of the space
        self.color_mode = color_mode    # used to vary which method of coloring used
        self.pluri_color = pluri_color    # color of a pluripotent cell
        self.diff_color = diff_color    # color of a differentiated cell
        self.pluri_gata6_high_color = pluri_gata6_high_color    # color of a pluripotent gata6 high cell
        self.pluri_nanog_high_color = pluri_nanog_high_color    # color of a pluripotent nanog high cell
        self.pluri_both_high_color = pluri_both_high_color    # color of a pluripotent gata6/nanog high cell
        self.guye_move = guye_move    # whether or not to use the Guye method of cell motility
        self.motility_force = motility_force    # the active force (in Newtons) of a cell actively moving
        self.max_radius = max_radius    # the maximum radius (in meters) of a cell
        self.fgf4_negative_move = fgf4_negative_move
        self.random_move = random_move

        # these arrays hold all values of the cells, each index corresponds to a cell.
        # you may ask why not create an array that holds a bunch of Cell objects...and my answer to that
        # is in order to parallelize the functions with a GPU you need to convert a collection of variables into
        # an array that can be interpreted by the GPU so Cell classes will have to be constantly mined for their
        # instance variables, put into an array, and then reevaluated from an updated array after processing
        self.cell_locations = np.empty((0, 3), dtype=float)    # holds every cell's location vector in the space
        self.cell_radii = np.empty((0, 1), dtype=float)     # holds every cell's radius
        self.cell_motion = np.empty((0, 1), dtype=bool)    # holds every cell's boolean for being in motion or not
        self.cell_fds = np.empty((0, 4), dtype=float)    # holds every cell's values for the fds, currently 4
        self.cell_states = np.empty((0, 1), dtype=str)    # holds every cell's state pluripotent or differentiated
        self.cell_diff_counter = np.empty((0, 1), dtype=int)    # holds every cell's differentiation counter
        self.cell_div_counter = np.empty((0, 1), dtype=int)    # holds every cell's division counter
        self.cell_death_counter = np.empty((0, 1), dtype=int)    # holds every cell's death counter
        self.cell_fds_counter = np.empty((0, 1), dtype=int)    # holds every cell's finite dynamical system counter
        self.cell_motility_force = np.empty((0, 3), dtype=float)    # holds every cell's motility force vector
        self.cell_jkr_force = np.empty((0, 3), dtype=float)    # holds every cell's JKR force vector
        self.cell_nearest_gata6 = np.empty((0, 1), dtype=None)    # holds index of nearest gata6 high neighbor
        self.cell_nearest_nanog = np.empty((0, 1), dtype=None)    # holds index of nearest nanog high neighbor
        self.cell_nearest_diff = np.empty((0, 1), dtype=None)    # holds index of nearest differentiated neighbor

        # holds the run time for key functions as a way tracking efficiency. each step these are outputted to the data
        # CSV file. this just initializes the variables as floats
        self.update_diffusion_time = float()
        self.check_neighbors_time = float()
        self.nearest_diff_time = float()
        self.cell_death_time = float()
        self.cell_diff_surround_time = float()
        self.cell_motility_time = float()
        self.cell_update_time = float()
        self.update_queue_time = float()
        self.handle_movement_time = float()

        # holds the current number of cells, step, and time when a step started (used for tracking efficiency)
        self.number_cells = 0
        self.current_step = self.beginning_step
        self.step_start = float()

        # holds all extracellular objects...this may be edited...later
        self.extracellular = np.empty((0, 1), dtype=object)

        # neighbor graph is used to locate cells that are in close proximity, while the JKR graph holds adhesion bonds
        # between cells that are either currently overlapping or still maintain an adhesive bond
        self.neighbor_graph = igraph.Graph()
        self.jkr_graph = igraph.Graph()

        # holds all indices of cells that will divide at a current step or be removed at that step
        self.cells_to_divide = np.empty((0, 1), dtype=int)
        self.cells_to_remove = np.empty((0, 1), dtype=int)

        # min and max radius lengths are used to calculate linear growth of the radius over time
        self.min_radius = self.max_radius / 2 ** 0.5
        self.pluri_growth = (self.max_radius - self.min_radius) / self.pluri_div_thresh
        self.diff_growth = (self.max_radius - self.min_radius) / self.diff_div_thresh

        # Youngs modulus and Poisson's ratio used for JKR contact
        self.youngs_mod = 1000
        self.poisson = 0.5

        # the csv and video objects that will be updated each step
        self.csv_object = object()
        self.video_object = object()

    def info(self):
        """ Records the beginning time of the step then
            prints the current step number and the
            total number of cells at the start of a step
        """
        # records the real time value of when a step begins
        self.step_start = time.time()

        # prints the current step number and the number of cells
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
        # start time
        self.update_diffusion_time = -1 * time.time()

        # go through all of the Extracellular objects updating their diffusion
        for i in range(len(self.extracellular)):
            self.extracellular[i].update_gradient(self)

        # end time
        self.update_diffusion_time += time.time()

    def add_cell(self, location, radius, motion, fds, state, diff_counter, div_counter, death_counter, fds_counter,
                 motility_force, jkr_force, nearest_gata6, nearest_nanog, nearest_diff):
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

        # add it to the following graphs, this is done implicitly by increasing the length of the vertex list by
        # one, which the indices directly correspond to the cell holder arrays
        self.neighbor_graph.add_vertex()
        self.jkr_graph.add_vertex()

        # revalue the total number of cells
        self.number_cells += 1

    def remove_cell(self, index):
        """ Given the index of a cell to remove,
            this will remove that from each array,
            graph, and total number of cells
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
                    self.handle_movement()

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
                    self.handle_movement()

        # clear the arrays for the next step
        self.cells_to_divide = np.empty((0, 1), dtype=int)
        self.cells_to_remove = np.empty((0, 1), dtype=int)

        # end time
        self.update_queue_time += time.time()

    def divide(self, index):
        """ Takes a cell or rather an index in the
            holder arrays and creates a new cell
            (index). This also updates factors such
            as size and counters.
        """
        # move the cells to positions that are representative of the new locations of daughter cells
        division_position = self.random_vector() * (self.max_radius - self.min_radius)
        self.cell_locations[index] += division_position
        location = self.cell_locations[index] - division_position

        # reduce radius to minimum size, set the division counter to zero, and None as the nearest differentiated cell
        self.cell_radii[index] = radius = self.min_radius
        self.cell_div_counter[index] = div_counter = 0
        self.cell_nearest_gata6[index] = nearest_gata6 = np.nan
        self.cell_nearest_nanog[index] = nearest_nanog = np.nan
        self.cell_nearest_diff[index] = nearest_diff = np.nan

        # keep identical values for motion, booleans, state, differentiation, cell death, and boolean update
        motion = self.cell_motion[index]
        booleans = self.cell_fds[index]
        state = self.cell_states[index]
        diff_counter = self.cell_diff_counter[index]
        death_counter = self.cell_death_counter[index]
        bool_counter = self.cell_fds_counter[index]

        # set the force vector to zero
        motility_force = np.zeros(3, dtype=float)
        jkr_force = np.zeros(3, dtype=float)

        # add the cell to the simulation
        self.add_cell(location, radius, motion, booleans, state, diff_counter, div_counter, death_counter,
                      bool_counter, motility_force, jkr_force, nearest_gata6, nearest_nanog, nearest_diff)

    def cell_update(self):
        """ Loops over all indices of cells and updates
            their values accordingly.
        """
        # start time
        self.cell_update_time = -1 * time.time()

        # loop over the cells
        for i in range(self.number_cells):
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

            # activate the following pathway based on if dox has been induced yet
            if self.current_step >= self.dox_step:
                # Extracellular interaction
                # take the location of a cell and determine the nearest diffusion point by creating a zone around a
                # diffusion point an any cells in the zone will base their value off of that
                x_step = self.extracellular[0].dx
                y_step = self.extracellular[0].dy
                z_step = self.extracellular[0].dz
                half_index_x = self.cell_locations[i][0] // (x_step / 2)
                half_index_y = self.cell_locations[i][1] // (y_step / 2)
                half_index_z = self.cell_locations[i][2] // (z_step / 2)
                index_x = math.ceil(half_index_x / 2)
                index_y = math.ceil(half_index_y / 2)
                index_z = math.ceil(half_index_z / 2)

                # if the diffusion point value is less than the max FGF4 it can hold and the cell is NANOG high
                # increase the FGF4 value by 1
                if self.extracellular[0].diffuse_values[index_x][index_y][index_z] < \
                        self.extracellular[0].maximum and self.cell_fds[i][3] == 1:
                    self.extracellular[0].diffuse_values[index_x][index_y][index_z] += 1

                # if the FGF4 amount for the location is greater than 0, set the fgf4_bool value to be 1 for the
                # functions
                if self.extracellular[0].diffuse_values[index_x][index_y][index_z] > 0:
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
                if temp_fgfr == 0 and new_fgf4 == 1 and \
                        self.extracellular[0].diffuse_values[index_x][index_y][index_z] > 0:
                    self.extracellular[0].diffuse_values[index_x][index_y][index_z] -= 1

                # if the cell is GATA6 high and pluripotent increase the differentiation counter by 1
                if self.cell_fds[i][2] == 1 and self.cell_states[i] == "Pluripotent":
                    self.cell_diff_counter[i] += 1

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

    def cell_death(self):
        """ Cellular death based on pluripotency
            and number of neighbors
        """
        # start time
        self.cell_death_time = -1 * time.time()

        # loops over all cells
        for i in range(self.number_cells):
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

        # end time
        self.cell_death_time += time.time()

    def cell_diff_surround(self):
        """ Simulates the phenomenon of differentiated
            cells inducing the differentiation of a
            pluripotent/NANOG high cell
        """
        # start time
        self.cell_diff_surround_time = -1 * time.time()

        # loops over all cells
        for i in range(self.number_cells):
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
                        self.cell_diff_counter[i] += 1
                        break

        # end time
        self.cell_diff_surround_time += time.time()

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

            else:
                # if pluripotent
                if self.cell_states[i] == "Pluripotent":
                    # if nanog high
                    if self.cell_fds[i][3] == 1:
                        # if there is a gata6 high cell nearby, move away from it
                        if not np.isnan(self.cell_nearest_gata6[i]):
                            nearest_index = self.cell_nearest_gata6[i]
                            normal = normal_vector(self.cell_locations[i], self.cell_locations[nearest_index])
                            self.cell_motility_force = normal * self.motility_force * -1

                        # if there is a nanog high cell nearby, move to it
                        elif not np.isnan(self.cell_nearest_nanog[i]):
                            nearest_index = self.cell_nearest_nanog[i]
                            normal = normal_vector(self.cell_locations[i], self.cell_locations[nearest_index])
                            self.cell_motility_force = normal * self.motility_force

                        # if nothing else, move randomly
                        else:
                            self.cell_motility_force = self.random_vector() * self.motility_force

                    # if gata6 high
                    elif self.cell_fds[i][2] == 1:
                        # if fgf4_negative movement
                        if self.fgf4_negative_move:
                            # if there is a nearby nanog cell, move away from it
                            if not np.isnan(self.cell_nearest_nanog[i]):
                                nearest_index = self.cell_nearest_nanog[i]
                                normal = normal_vector(self.cell_locations[i], self.cell_locations[nearest_index])
                                self.cell_motility_force = normal * self.motility_force * -1
                            # move randomly instead
                            else:
                                self.cell_motility_force = self.random_vector() * self.motility_force

                        # if guye movement
                        elif self.guye_move:
                            # if a nearby differentiated cell, move to it
                            if not np.isnan(self.cell_nearest_diff[i]):
                                nearest_index = self.cell_nearest_diff[i]
                                normal = normal_vector(self.cell_locations[i], self.cell_locations[nearest_index])
                                self.cell_motility_force = normal * self.motility_force
                            # move randomly instead
                            else:
                                self.cell_motility_force = self.random_vector() * self.motility_force

                        # if random movement
                        elif self.random_move:
                            self.cell_motility_force = self.random_vector() * self.motility_force

                # differentiated
                else:
                    # if there is a nearby nanog cell, move away from it
                    if not np.isnan(self.cell_nearest_nanog[i]):
                        nearest_index = self.cell_nearest_nanog[i]
                        normal = normal_vector(self.cell_locations[i], self.cell_locations[nearest_index])
                        self.cell_motility_force = normal * self.motility_force * -1
                    # move randomly instead
                    else:
                        self.cell_motility_force = self.random_vector() * self.motility_force

        # end time
        self.cell_motility_time += time.time()

    def check_neighbors(self):
        """ checks all of the distances between cells if it
            is less than a fixed value create a connection
            between two cells.
        """
        # start time
        self.check_neighbors_time = -1 * time.time()

        # clear all of the edges in the neighbor graph and get the radius of search length
        self.neighbor_graph.delete_edges(None)
        distance = self.neighbor_distance

        # provide an idea of the maximum number of neighbors for a cells
        max_neighbors = 15
        length = self.number_cells * max_neighbors
        edge_holder = np.zeros((length, 2), dtype=int)

        # divides the space into bins and gives a holder of fixed size for each bin
        bins_size = self.size // distance + np.array([5, 5, 5])
        bins_size_help = tuple(bins_size.astype(int))
        bins_size = np.append(bins_size, 100)
        bins_size = tuple(bins_size.astype(int))

        # assigns values of -1 to denote a lack of cells
        bins = np.empty(bins_size, dtype=int)

        # an array used to accelerate the cuda function by telling the function how many cells are in a given bin
        bins_help = np.zeros(bins_size_help, dtype=int)

        # call the gpu version
        if self.parallel:
            edge_holder = Parallel.check_neighbors_gpu(self.number_cells, distance, max_neighbors, edge_holder, bins,
                                                       bins_help, self.cell_locations)
        # call the cpu version
        else:
            edge_holder = check_neighbors_cpu(self.number_cells, distance, edge_holder, bins, bins_help,
                                              self.cell_locations)

        # add the new edges and remove any duplicate edges or loops
        self.neighbor_graph.add_edges(edge_holder)
        self.neighbor_graph.simplify()

        # end time
        self.check_neighbors_time += time.time()

    def nearest(self):
        """ looks at cells within a given radius
            a determines the closest differentiated
            cell to a pluripotent cell.
        """
        # start time
        self.nearest_diff_time = -1 * time.time()

        # get the radius of search length
        distance = self.nearest_distance

        # divides the space into bins and gives a holder of fixed size for each bin
        bins_size = self.size // distance + np.array([5, 5, 5])
        bins_size_help = tuple(bins_size.astype(int))
        bins_size = np.append(bins_size, 100)
        bins_size = tuple(bins_size.astype(int))

        # assigns values of -1 to denote a lack of cells
        bins = np.empty(bins_size, dtype=int)

        # an array used to accelerate the cuda function by telling the function how many cells are in a given bin
        bins_help = np.zeros(bins_size_help, dtype=int)

        # find the nearest gata6 high cell
        new_gata6, new_nanog, new_diff = nearest_cpu(self.number_cells, distance, bins, bins_help,
                                                     self.cell_locations, self.cell_nearest_gata6,
                                                     self.cell_nearest_nanog, self.cell_nearest_diff, self.cell_states,
                                                     self.cell_fds)

        # revalue the array holding the indices of nearest cells of given type
        self.cell_nearest_gata6 = new_gata6
        self.cell_nearest_nanog = new_nanog
        self.cell_nearest_diff = new_diff

        # end time
        self.nearest_diff_time += time.time()

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

        # divides the space into bins and gives a holder of fixed size for each bin
        bins_size = self.size // distance + np.array([5, 5, 5])
        bins_size_help = tuple(bins_size.astype(int))
        bins_size = np.append(bins_size, 100)
        bins_size = tuple(bins_size.astype(int))

        # assigns values of -1 to denote a lack of cells
        bins = np.empty(bins_size, dtype=int)

        # an array used to accelerate the cuda function by telling the function how many cells are in a given bin
        bins_help = np.zeros(bins_size_help, dtype=int)

        # call the gpu version
        if self.parallel:
            edge_holder = Parallel.jkr_neighbors_gpu(self.number_cells, distance, max_neighbors, edge_holder, bins,
                                                     bins_help, self.cell_locations, self.cell_radii)
        # call the cpu version
        else:
            edge_holder = jkr_neighbors_cpu(self.number_cells, distance, edge_holder, bins, bins_help,
                                            self.cell_locations, self.cell_radii)

        # add the new edges and remove any duplicate edges or loops
        self.jkr_graph.add_edges(edge_holder)
        self.jkr_graph.simplify()

    def handle_movement(self):
        """ runs the following functions together for a
            given time amount. Resets the force and
            velocity arrays as well.
        """
        # start time
        self.handle_movement_time = -1 * time.time()

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

        # end time
        self.handle_movement_time += time.time()

    def get_forces(self):
        """ goes through all of the cells and quantifies any forces arising
            from adhesion or repulsion between the cells
        """
        # get the updated edges of the jkr graph
        jkr_edges = np.array(self.jkr_graph.get_edgelist())
        delete_edges = np.zeros(len(jkr_edges), dtype=int)

        # call the gpu version
        if self.parallel and len(jkr_edges) > 0:
            forces, delete_edges = Parallel.get_forces_gpu(jkr_edges, delete_edges, self.poisson, self.youngs_mod,
                                                           self.adhesion_const, self.cell_locations, self.cell_radii,
                                                           self.cell_jkr_force)
        # call the cpu version
        else:
            forces, delete_edges = get_forces_cpu(jkr_edges, delete_edges, self.poisson, self.youngs_mod,
                                                  self.adhesion_const, self.cell_locations, self.cell_radii,
                                                  self.cell_jkr_force)

        # update the jkr graph to remove an edges that have be broken and update the cell jkr forces
        self.jkr_graph.delete_edges(delete_edges)
        self.cell_jkr_force = forces

    def apply_forces(self):
        """ Turns the active motility/division forces
            and inactive JKR forces into movement
        """
        # call the gpu version
        if self.parallel:
            # prevents the need for having the numba library if it's not installed
            new_locations = Parallel.apply_forces_gpu(self.number_cells, self.cell_jkr_force, self.cell_motility_force,
                                                      self.cell_locations, self.cell_radii, self.viscosity, self.size,
                                                      self.move_time_step)
        # call the cpu version
        else:
            new_locations = apply_forces_cpu(self.number_cells, self.cell_jkr_force, self.cell_motility_force,
                                             self.cell_locations, self.cell_radii, self.viscosity, self.size,
                                             self.move_time_step)

        # update the locations and reset the jkr forces back to zero
        self.cell_locations = new_locations
        self.cell_jkr_force = np.zeros((self.number_cells, 3))

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

def normal_vector(location_1, location_2):
    """ Get the normal vector between
        two points
    """
    vector = location_2 - location_1
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return np.array([0, 0, 0])
    else:
        return vector / magnitude


@jit(nopython=False)
def nearest_cpu(number_cells, distance, bins, bins_help, cell_locations, nearest_gata6, nearest_nanog, nearest_diff,
                cell_states, cell_fds):
    """ This is the Numba optimized
        version of the nearest function.
    """
    for i in range(number_cells):
        # offset bins by 1 to avoid missing cells
        block_location = cell_locations[i] // distance + np.array([2, 2, 2])
        x, y, z = int(block_location[0]), int(block_location[1]), int(block_location[2])

        # tries to place the cell in the holder for the bin. if the holder's value is other than -1 it will move
        # to the next spot to see if it's empty
        place = bins_help[x][y][z]

        # gives the cell's array location
        bins[x][y][z][place] = i

        # updates the total amount cells in a bin
        bins_help[x][y][z] += 1

    # loops over all cells, with the current cell being the pivot of the search method
    for pivot_index in range(number_cells):
        # offset bins by 2 to avoid missing cells
        block_location = cell_locations[pivot_index] // distance + np.array([2, 2, 2])
        x, y, z = int(block_location[0]), int(block_location[1]), int(block_location[2])

        # initialize this variables with essentially nothing values
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
                        current_index = int(bins[x + i][y + j][z + k][l])

                        # check to see if that cell is within the search radius and not the same cell
                        m = np.linalg.norm(cell_locations[current_index] - cell_locations[pivot_index])
                        if m <= distance and current_index != pivot_index:
                            # update the nearest gata6 high cell
                            if cell_fds[current_index][2] == 1:
                                # if it's closer than the last cell, update the closest magnitude and index
                                if m < nearest_gata6_dist:
                                    nearest_gata6_index = current_index
                                    nearest_gata6_dist = m

                            # update the nearest nanog high cell
                            elif cell_fds[current_index][3] == 1:
                                # if it's closer than the last cell, update the closest magnitude and index
                                if m < nearest_nanog_dist:
                                    nearest_nanog_index = current_index
                                    nearest_nanog_dist = m

                            # update the nearest differentiated cell
                            elif cell_states[current_index] == "Differentiated":
                                # if it's closer than the last cell, update the closest magnitude and index
                                if m < nearest_diff_dist:
                                    nearest_diff_index = current_index
                                    nearest_diff_dist = m

        # update the nearest cell of desired type index
        nearest_gata6[pivot_index] = nearest_gata6_index
        nearest_nanog[pivot_index] = nearest_nanog_index
        nearest_diff[pivot_index] = nearest_diff_index

    # return the updated edges
    return nearest_gata6, nearest_nanog, nearest_diff


@jit(nopython=True)
def check_neighbors_cpu(number_cells, distance, edge_holder, bins, bins_help, cell_locations):
    """ This is the Numba optimized version of
        the check_neighbors function that runs
        solely on the cpu.
    """
    # holds the total amount of edges as the function runs, used for indexing
    edge_counter = 0

    # loops over all cells, with the current cell being the pivot of the search method
    for pivot_index in range(number_cells):
        # offset bins by 2 to avoid missing cells
        block_location = cell_locations[pivot_index] // distance + np.array([2, 2, 2])
        x, y, z = int(block_location[0]), int(block_location[1]), int(block_location[2])

        # gets the index where the cell should be placed
        place = bins_help[x][y][z]

        # adds the cell index to the bins array
        bins[x][y][z][place] = pivot_index

        # increase the count of cell in the bin by 1
        bins_help[x][y][z] += 1

        # loop over the bins that surround the current bin
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of cells in a bin
                    bin_count = bins_help[x + i][y + j][z + k]

                    # go through the bin determining if a cell is a neighbor
                    for l in range(bin_count):
                        # get the index of the current cell in question
                        current_index = int(bins[x + i][y + j][z + k][l])

                        # check to see if that cell is within the search radius
                        vector = cell_locations[current_index] - cell_locations[pivot_index]
                        if np.linalg.norm(vector) <= distance and pivot_index != current_index:
                            # update the edge array and increase the index for the next addition
                            edge_holder[edge_counter][0] = pivot_index
                            edge_holder[edge_counter][1] = current_index
                            edge_counter += 1

    # return the updated edges
    return edge_holder


@jit(nopython=True)
def jkr_neighbors_cpu(number_cells, distance, edge_holder, bins, bins_help, cell_locations, cell_radii):
    """ This is the Numba optimized version of
        the jkr_neighbors function that runs
        solely on the cpu.
    """
    # holds the total amount of edges as the function runs, used for indexing
    edge_counter = 0

    # loops over all cells, with the current cell being the pivot of the search method
    for pivot_index in range(number_cells):
        # offset bins by 2 to avoid missing cells
        block_location = cell_locations[pivot_index] // distance + np.array([2, 2, 2])
        x, y, z = int(block_location[0]), int(block_location[1]), int(block_location[2])

        # gets the index where the cell should be placed
        place = bins_help[x][y][z]

        # adds the cell index to the bins array
        bins[x][y][z][place] = pivot_index

        # increase the count of cell in the bin by 1
        bins_help[x][y][z] += 1

        # loop over the bins that surround the current bin
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of cells in a bin
                    bin_count = bins_help[x + i][y + j][z + k]

                    # go through the bin determining if a cell is a neighbor
                    for l in range(bin_count):
                        # get the index of the current cell in question
                        current_index = int(bins[x + i][y + j][z + k][l])

                        # get the magnitude of the distance between the cells
                        vector = cell_locations[current_index] - cell_locations[pivot_index]
                        mag = np.linalg.norm(vector)

                        # calculate the cell overlap
                        overlap = cell_radii[current_index] + cell_radii[pivot_index] - mag

                        # if there is overlap and not the same cell add the edge
                        if overlap >= 0 and pivot_index != current_index:
                            # update the edge array and increase the index for the next addition
                            edge_holder[edge_counter][0] = pivot_index
                            edge_holder[edge_counter][1] = current_index
                            edge_counter += 1

    # return the updated edges
    return edge_holder


@jit(nopython=True, parallel=True)
def get_forces_cpu(jkr_edges, delete_jkr_edges, poisson, youngs_mod, adhesion_const, cell_locations,
                   cell_radii, cell_jkr_force):
    """ This is the Numba optimized version of
        the get_forces function that runs
        solely on the cpu.
    """
    # loops over the jkr edges
    for i in prange(len(jkr_edges)):
        # get the indices of the nodes in the edge
        index_1 = jkr_edges[i][0]
        index_2 = jkr_edges[i][1]

        # hold the vector between the centers of the cells and the magnitude of this vector
        disp_vector = cell_locations[index_1] - cell_locations[index_2]
        magnitude = np.linalg.norm(disp_vector)
        normal = np.array([0.0, 0.0, 0.0])

        # the parallel jit prefers reductions so it's better to initialize the value and revalue it
        if magnitude != 0:
            normal += disp_vector / magnitude

        # get the total overlap of the cells used later in calculations
        overlap = cell_radii[index_1] + cell_radii[index_2] - magnitude

        # gets two values used for JKR
        e_hat = (((1 - poisson ** 2) / youngs_mod) + ((1 - poisson ** 2) / youngs_mod)) ** -1
        r_hat = ((1 / cell_radii[index_1]) + (1 / cell_radii[index_2])) ** -1

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
            jkr_force = f * math.pi * adhesion_const * r_hat

            # adds the adhesive force as a vector in opposite directions to each cell's force holder
            cell_jkr_force[index_1] += jkr_force * normal
            cell_jkr_force[index_2] -= jkr_force * normal

        # remove the edge if the it fails to meet the criteria for distance, JKR simulating that
        # the bond is broken
        else:
            delete_jkr_edges[i] = i

    # return the updated jkr forces and the edges to be deleted
    return cell_jkr_force, delete_jkr_edges


@jit(nopython=True, parallel=True)
def apply_forces_cpu(number_cells, cell_jkr_force, cell_motility_force, cell_locations, cell_radii, viscosity, size,
                     move_time_step):
    """ This is the Numba optimized version of
        the apply_forces function that runs
        solely on the cpu.
    """
    # loops over all cells using the explicit parallel loop from Numba
    for i in prange(number_cells):
        # stokes law for velocity based on force and fluid viscosity
        stokes_friction = 6 * math.pi * viscosity * cell_radii[i]

        # update the velocity of the cell based on the solution
        velocity = (cell_motility_force[i] + cell_jkr_force[i]) / stokes_friction

        # set the possible new location
        new_location = cell_locations[i] + velocity * move_time_step

        # loops over all directions of space
        for j in range(0, 3):
            # check if new location is in environment space if not simulation a collision with the bounds
            if new_location[j] > size[j]:
                cell_locations[i][j] = size[j]
            elif new_location[j] < 0:
                cell_locations[i][j] = 0.0
            else:
                cell_locations[i][j] = new_location[j]

    # return the updated cell locations
    return cell_locations
