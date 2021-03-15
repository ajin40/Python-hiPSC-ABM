import numpy as np
import random as r
import math
from numba import cuda

from backend import *


class Functions:

    def info(self):
        """ Records the beginning of the step in real time and
            prints the current step/number of cells.
        """
        # records when the step begins, used for measuring efficiency
        self.step_start = time.perf_counter()  # time.perf_counter() is more accurate than time.time()

        # prints the current step number and the number of cells
        print("Step: " + str(self.current_step))
        print("Number of cells: " + str(self.number_cells))

    @record_time
    def cell_death(self):
        """ Marks the cell for removal if it meets
            the criteria for cell death.
        """
        for index in range(self.number_cells):
            # checks to see if cell is pluripotent
            if self.states[index] == 0:

                # gets the number of neighbors for a cell, increasing the death counter if not enough neighbors
                if len(self.neighbor_graph.neighbors(index)) < self.lonely_thresh:
                    self.death_counters[index] += 1

                # if not, reset the death counter back to zero
                else:
                    self.death_counters[index] = 0

                # add cell to removal array if it meets the parameters
                if self.death_counters[index] >= self.death_thresh:
                    self.cells_to_remove = np.append(self.cells_to_remove, index)

    @record_time
    def cell_diff_surround(self):
        """ Simulates differentiated cells inducing the
            differentiation of a pluripotent cell.
        """
        for index in range(self.number_cells):
            # checks to see if cell is pluripotent and GATA6 low/medium
            if self.states[index] == 0 and self.GATA6[index] < self.NANOG[index]:

                # get the list of neighbors for the cell
                neighbors = self.neighbor_graph.neighbors(index)

                # loop over neighbors, counting the ones that are differentiated
                diff_neighbors = 0
                for neighbor_index in neighbors:
                    # checks to see if current neighbor is differentiated and if so add to the counter
                    if self.states[neighbor_index] == 1:
                        diff_neighbors += 1

                    # if the number of differentiated meets the threshold, set the cell as gata6 high and nanog low
                    if diff_neighbors >= 6:
                        self.GATA6[index] = self.field - 1
                        self.NANOG[index] = 0
                        break

    @record_time
    def cell_division(self):
        """ Increases the cell division counter and if the
            cell meets criteria mark it for division.
        """
        for index in range(self.number_cells):
            # stochastically increase the division counter by either 0 or 1
            self.div_counters[index] += r.randint(0, 1)

            # pluripotent cell
            if self.states[index] == 0:
                # check the division counter against the threshold, add to array if dividing
                if self.div_counters[index] >= self.pluri_div_thresh:
                    self.cells_to_divide = np.append(self.cells_to_divide, index)

            # differentiated cell
            else:
                # check the division counter against the threshold, add to array if dividing
                if self.div_counters[index] >= self.diff_div_thresh:

                    # check for contact inhibition since differentiated
                    if len(self.neighbor_graph.neighbors(index)) < 6:
                        self.cells_to_divide = np.append(self.cells_to_divide, index)

    @record_time
    def cell_growth(self):
        """ Simulates the growth of a cell currently linear,
            radius-based growth.
        """
        for index in range(self.number_cells):
            # increase the cell radius based on the state and whether or not it has reached the max size
            if self.radii[index] < self.max_radius:
                # pluripotent growth
                if self.states[index] == 0:
                    radius = self.pluri_growth * self.div_counters[index] + self.min_radius

                # differentiated growth
                else:
                    radius = self.diff_growth * self.div_counters[index] + self.min_radius

                # update the radius for the index
                self.radii[index] = radius

    @record_time
    def cell_stochastic_update(self):
        """ Stochastically updates the value for GATA6 and NANOG
            based on set probabilities.
        """
        for index in range(self.number_cells):
            # if falling under threshold, raise the GATA6 value to the highest
            if r.random() < self.GATA6_prob:
                if self.GATA6[index] != self.field - 1:
                    self.GATA6[index] += 1

            # # if falling under threshold, raise the NANOG value to the highest
            # if r.random() < self.NANOG_prob:
            #     if self.NANOG[index] != self.field - 1:
            #         self.NANOG[index] += 1

    @record_time
    def cell_pathway(self):
        """ Updates finite dynamical system variables and
            extracellular conditions.
        """
        for index in range(self.number_cells):
            # add FGF4 to the gradient based on the cell's value of NANOG
            if self.NANOG[index] > 0:
                # get the amount
                amount = self.NANOG[index]

                # add it to the normal FGF4 gradient and the alternative FGF4 gradient
                self.adjust_morphogens("fgf4_values", index, amount, "nearest")
                # self.adjust_morphogens("fgf4_alt", index, amount, "distance")

            # activate the following pathway based on if doxycycline  has been induced yet (after 24 hours/48 steps)
            if self.current_step >= self.dox_step:
                # get an FGF4 value for the FDS based on the concentration of FGF4
                fgf4_value = self.get_concentration("fgf4_values", index)

                # if FDS is boolean
                if self.field == 2:
                    # base thresholds on the maximum concentrations
                    if fgf4_value < 0.5 * self.max_concentration:
                        fgf4_fds = 0   # FGF4 low
                    else:
                        fgf4_fds = 1    # FGF4 high

                # otherwise assume ternary for now
                else:
                    # base thresholds on the maximum concentrations
                    if fgf4_value < 1/3 * self.max_concentration:
                        fgf4_fds = 0    # FGF4 low
                    elif fgf4_value < 2/3 * self.max_concentration:
                        fgf4_fds = 1    # FGF4 medium
                    else:
                        fgf4_fds = 2    # FGF4 high

                # temporarily hold the FGFR value
                temp_fgfr = self.FGFR[index]

                # if updating the FDS values this step
                if self.fds_counters[index] % self.fds_thresh == 0:
                    # get the current FDS values of the cell
                    x1 = fgf4_fds
                    x2 = self.FGFR[index]
                    x3 = self.ERK[index]
                    x4 = self.GATA6[index]
                    x5 = self.NANOG[index]

                    # if the FDS is boolean
                    if self.field == 2:
                        # update boolean values based on FDS functions
                        new_fgfr = (x1 * x4) % 2
                        new_erk = x2 % 2
                        new_gata6 = (1 + x5 + x5 * x4) % 2
                        new_nanog = ((x3 + 1) * (x4 + 1)) % 2

                    # otherwise assume ternary
                    else:
                        # update ternary values based on FDS functions
                        new_fgfr = (x1 * x4 * ((2*x1 + 1) * (2*x4 + 1) + x1 * x4)) % 3
                        new_erk = x2 % 3
                        new_gata6 = ((x4**2) * (x5 + 1) + (x5**2) * (x4 + 1) + 2*x5 + 1) % 3
                        new_nanog = (x5**2 + x5 * (x5 + 1) * (x3 * (2*x4**2 + 2*x3 + 1) + x4*(2*x3**2 + 2*x4 + 1)) +
                                     (2*x3**2 + 1) * (2*x4**2 + 1)) % 3

                    # if the amount of FGFR has increased, subtract that much FGF4 from the gradient
                    fgfr_change = new_fgfr - temp_fgfr
                    if fgfr_change > 0:
                        self.adjust_morphogens("fgf4_values", index, -1 * fgfr_change, "nearest")

                    # update the FDS values of the cell
                    self.FGFR[index] = new_fgfr
                    self.ERK[index] = new_erk
                    self.GATA6[index] = new_gata6
                    self.NANOG[index] = new_nanog

                # increase the finite dynamical system counter
                self.fds_counters[index] += 1

    @record_time
    def cell_differentiate(self):
        """ Based on GATA6 and NANOG values, stochastically increase
            differentiation counter and/or differentiate.
        """
        for index in range(self.number_cells):
            # if the cell is GATA6 high and pluripotent
            if self.GATA6[index] > self.NANOG[index] and self.states[index] == 0:

                # increase the differentiation counter by 0 or 1
                self.diff_counters[index] += r.randint(0, 1)

                # if the differentiation counter is greater than or equal to the threshold, differentiate
                if self.diff_counters[index] >= self.pluri_to_diff:
                    # change the state to differentiated
                    self.states[index] = 1

                    # make sure NANOG is low
                    self.NANOG[index] = 0

                    # allow the cell to actively move again
                    self.motion[index] = True

    @record_time
    def cell_motility(self):
        """ Gives the cells a motive force depending on
            set rules for the cell types.
        """
        # this is the motility force of the cells
        motility_force = 0.000000002

        # loop over all of the cells
        for index in range(self.number_cells):
            # get the neighbors of the cell
            neighbors = self.neighbor_graph.neighbors(index)

            # if not surrounded 6 or more cells, calculate motility forces
            if len(neighbors) < 6:
                # if the cell is differentiated
                if self.states[index] == 1:
                    # create a vector to hold the sum of normal vectors between a cell and its neighbors
                    vector_holder = np.array([0.0, 0.0, 0.0])

                    # loop over the neighbors
                    count = 0
                    for i in range(len(neighbors)):
                        # if neighbor is nanog high, add vector to the cell to the holder
                        if self.NANOG[neighbors[i]] > self.GATA6[neighbors[i]]:
                            count += 1
                            vector = self.locations[neighbors[i]] - self.locations[index]
                            vector_holder += vector

                    # if there is at least one nanog high cell move away from it
                    if count > 0:
                        # get the normalized vector
                        normal = backend.normal_vector(vector_holder)

                        # move in direction opposite to nanog high cells
                        random = self.random_vector()
                        self.motility_forces[index] += (normal * -0.8 + random * 0.2) * motility_force

                    # if no nanog high cells around, move randomly
                    else:
                        self.motility_forces[index] += self.random_vector() * motility_force

                # otherwise the cell is pluripotent
                else:
                    # if the cell state is GATA6 high
                    if self.GATA6[index] > self.NANOG[index]:
                        # if guye movement, move toward differentiated cells
                        if self.guye_move:
                            # create a vector to hold the sum of normal vectors between a cell and its neighbors
                            vector_holder = np.array([0.0, 0.0, 0.0])

                            # loop over the neighbors
                            count = 0
                            for i in range(len(neighbors)):
                                # if neighbor is differentiated, add vector to the cell to the holder
                                if self.states[neighbors[i]] == 1:
                                    count += 1
                                    vector = self.locations[neighbors[i]] - self.locations[index]
                                    vector_holder += vector

                            # if there is at least differentiated cell move toward it
                            if count > 0:
                                # get the normalized vector
                                normal = backend.normal_vector(vector_holder)

                                # move in direction to differentiated cells
                                random = self.random_vector()
                                self.motility_forces[index] += (normal * 0.8 + random * 0.2) * motility_force

                            # if no differentiated cells around, move randomly
                            else:
                                self.motility_forces[index] += self.random_vector() * motility_force

                        # otherwise move away from nanog high cells
                        else:
                            # create a vector to hold the sum of normal vectors between a cell and its neighbors
                            vector_holder = np.array([0.0, 0.0, 0.0])

                            # loop over the neighbors
                            count = 0
                            for i in range(len(neighbors)):
                                # if neighbor is nanog high, add vector to the cell to the holder
                                if self.NANOG[neighbors[i]] > self.GATA6[neighbors[i]]:
                                    count += 1
                                    vector = self.locations[neighbors[i]] - self.locations[index]
                                    vector_holder += vector

                            # if there is at least one nanog high cell move away from it
                            if count > 0:
                                # get the normalized vector
                                normal = backend.normal_vector(vector_holder)

                                # move in direction opposite to nanog high cells
                                random = self.random_vector()
                                self.motility_forces[index] += (normal * -0.8 + random * 0.2) * motility_force

                            # if no nanog high cells around, move randomly
                            else:
                                self.motility_forces[index] += self.random_vector() * motility_force

                    # if the cell is nanog high and gata6 low
                    elif self.GATA6[index] < self.NANOG[index]:
                        # create a vector to hold the sum of normal vectors between a cell and its neighbors
                        vector_holder = np.array([0.0, 0.0, 0.0])

                        # loop over the neighbors
                        count = 0
                        for i in range(len(neighbors)):
                            # if neighbor is nanog high, add vector to the cell to the holder
                            if self.NANOG[neighbors[i]] > self.GATA6[neighbors[i]]:
                                count += 1
                                vector = self.locations[neighbors[i]] - self.locations[index]
                                vector_holder += vector

                        # if there is at least one nanog high cell move toward it
                        if count > 0:
                            # get the normalized vector
                            normal = backend.normal_vector(vector_holder)

                            # move in direction to nanog high cells
                            random = self.random_vector()
                            self.motility_forces[index] += (normal * 0.8 + random * 0.2) * motility_force

                        # if no nanog high cells around, move randomly
                        else:
                            self.motility_forces[index] += self.random_vector() * motility_force

                    # if same value, move randomly
                    else:
                        self.motility_forces[index] += self.random_vector() * motility_force

    @record_time
    def eunbi_motility(self):
        """ Gives the cells a motive force depending on
            set rules for the cell types where these rules
            are closer to Eunbi's model.
        """
        # this is the motility force of the cells
        motility_force = 0.000000008

        # loop over all of the cells
        for index in range(self.number_cells):
            # see if the cell is moving or not
            if self.motion[index]:
                # get the neighbors of the cell if the cell is actively moving
                neighbors = self.neighbor_graph.neighbors(index)

                # if cell is not surrounded by 6 or more other cells, calculate motility forces
                if len(neighbors) < 6:
                    # if differentiated
                    if self.states[index] == 1:
                        # if there is a nanog high cell nearby, move away from it
                        if self.nearest_nanog[index] != -1:
                            nearest_index = self.nearest_nanog[index]
                            vector = self.locations[nearest_index] - self.locations[index]
                            normal = backend.normal_vector(vector)
                            random = self.random_vector()
                            self.motility_forces[index] += (normal * -0.8 + random * 0.2) * motility_force

                        # if no nearby nanog high cells, move randomly
                        else:
                            self.motility_forces[index] += self.random_vector() * motility_force

                    # if the cell is gata6 high and nanog low
                    elif self.GATA6[index] > self.NANOG[index]:
                        # if there is a differentiated cell nearby, move toward it
                        if self.nearest_diff[index] != -1:
                            nearest_index = self.nearest_diff[index]
                            vector = self.locations[nearest_index] - self.locations[index]
                            normal = backend.normal_vector(vector)
                            random = self.random_vector()
                            self.motility_forces[index] += (normal * 0.8 + random * 0.2) * motility_force

                        # if no nearby differentiated cells, move randomly
                        else:
                            self.motility_forces[index] += self.random_vector() * motility_force

                    # if the cell is nanog high and gata6 low
                    elif self.NANOG[index] > self.GATA6[index]:
                        # if there is a nanog high cell nearby, move toward it
                        if self.nearest_nanog[index] != -1:
                            nearest_index = self.nearest_nanog[index]
                            vector = self.locations[nearest_index] - self.locations[index]
                            normal = backend.normal_vector(vector)
                            random = self.random_vector()
                            self.motility_forces[index] += (normal * 0.8 + random * 0.2) * motility_force

                        # if there is a gata6 high cell nearby, move away from it
                        elif self.nearest_gata6[index] != -1:
                            nearest_index = self.nearest_gata6[index]
                            vector = self.locations[nearest_index] - self.locations[index]
                            normal = backend.normal_vector(vector)
                            random = self.random_vector()
                            self.motility_forces[index] += (normal * -0.8 + random * 0.2) * motility_force

                        else:
                            self.motility_forces[index] += self.random_vector() * motility_force

                    # if both gata6/nanog high or both low, move randomly
                    else:
                        self.motility_forces[index] += self.random_vector() * motility_force

                else:
                    self.motion[index] = False

    @record_time
    def get_neighbors(self, distance=0.00002):
        """ For all cells, determines which cells fall within a fixed
            radius to denote a neighbor then stores this information
            in a graph (uses a bin/bucket sorting method).
        """
        # if a static variable has not been created to hold the maximum number of neighbors for a cell, create one
        if not hasattr(Functions.get_neighbors, "max_neighbors"):
            # begin with a low number of neighbors that can be revalued if the max neighbors exceeds this value
            Functions.get_neighbors.max_neighbors = 5

        # if a static variable has not been created to hold the maximum number of cells in a bin, create one
        if not hasattr(Functions.get_neighbors, "max_cells"):
            # begin with a low number of cells that can be revalued if the max number of cells exceeds this value
            Functions.get_neighbors.max_cells = 5

        # clear all of the edges in the neighbor graph
        self.neighbor_graph.delete_edges(None)

        # calls the function that generates an array of bins that generalize the cell locations in addition to a
        # creating a helper array that assists the search method in counting cells for a particular bin
        bins, bins_help, bin_locations, max_cells = self.assign_bins(distance, Functions.get_neighbors.max_cells)

        # update the value of the max number of cells in a bin
        Functions.get_neighbors.max_cells = max_cells

        # this will run once if all edges are included in edge_holder, breaking the loop. if not, this will
        # run a second time with an updated value for the number of predicted neighbors such that all edges are included
        while True:
            # create array used to hold edges, array to say if edge exists, and array to count the edges per cell
            length = self.number_cells * Functions.get_neighbors.max_neighbors
            edge_holder = np.zeros((length, 2), dtype=int)
            if_edge = np.zeros(length, dtype=bool)
            edge_count = np.zeros(self.number_cells, dtype=int)

            # call the nvidia gpu version
            if self.parallel:
                # send the following as arrays to the gpu
                bin_locations_cuda = cuda.to_device(bin_locations)
                locations_cuda = cuda.to_device(self.locations)
                bins_cuda = cuda.to_device(bins)
                bins_help_cuda = cuda.to_device(bins_help)
                distance_cuda = cuda.to_device(distance)
                edge_holder_cuda = cuda.to_device(edge_holder)
                if_edge_cuda = cuda.to_device(if_edge)
                edge_count_cuda = cuda.to_device(edge_count)
                max_neighbors_cuda = cuda.to_device(Functions.get_neighbors.max_neighbors)

                # allocate threads and blocks for gpu memory "threads per block" and "blocks per grid"
                tpb = 72
                bpg = math.ceil(self.number_cells / tpb)

                # call the cuda kernel with new gpu arrays
                get_neighbors_gpu[bpg, tpb](bin_locations_cuda, locations_cuda, bins_cuda, bins_help_cuda, distance_cuda,
                                            edge_holder_cuda, if_edge_cuda, edge_count_cuda, max_neighbors_cuda)

                # return the only the following array(s) back from the gpu
                edge_holder = edge_holder_cuda.copy_to_host()
                if_edge = if_edge_cuda.copy_to_host()
                edge_count = edge_count_cuda.copy_to_host()

            # call the jit cpu version
            else:
                edge_holder, if_edge, edge_count = get_neighbors_cpu(self.number_cells, bin_locations, self.locations,
                                                                     bins, bins_help, distance, edge_holder, if_edge,
                                                                     edge_count, Functions.get_neighbors.max_neighbors)

            # either break the loop if all neighbors were accounted for or revalue the maximum number of neighbors
            # based on the output of the function call and double it for future calls
            max_neighbors = np.amax(edge_count)
            if Functions.get_neighbors.max_neighbors >= max_neighbors:
                break
            else:
                Functions.get_neighbors.max_neighbors = max_neighbors * 2

        # reduce the edges to only edges that actually exist
        edge_holder = edge_holder[if_edge]

        # add the edges to the neighbor graph
        self.neighbor_graph.add_edges(edge_holder)

    @record_time
    def nearest(self, distance=0.00002):
        """ Determines the nearest GATA6 high, NANOG high, and
            differentiated cell within a fixed radius for each
            cell.
        """
        # if a static variable has not been created to hold the maximum number of cells in a bin, create one
        if not hasattr(Functions.nearest, "max_cells"):
            # begin with a low number of cells that can be revalued if the max number of cells exceeds this value
            Functions.nearest.max_cells = 5

        # calls the function that generates an array of bins that generalize the cell locations in addition to a
        # creating a helper array that assists the search method in counting cells for a particular bin
        bins, bins_help, bin_locations, max_cells = self.assign_bins(distance, Functions.nearest.max_cells)

        # update the value of the max number of cells in a bin
        Functions.nearest.max_cells = max_cells

        # turn the following array into True/False instead of strings
        if_diff = self.states == 1

        # call the nvidia gpu version
        if self.parallel:
            # send the following as arrays to the gpu
            bin_locations_cuda = cuda.to_device(bin_locations)
            locations_cuda = cuda.to_device(self.locations)
            bins_cuda = cuda.to_device(bins)
            bins_help_cuda = cuda.to_device(bins_help)
            distance_cuda = cuda.to_device(distance)
            if_diff_cuda = cuda.to_device(if_diff)
            gata6_cuda = cuda.to_device(self.GATA6)
            nanog_cuda = cuda.to_device(self.NANOG)
            nearest_gata6_cuda = cuda.to_device(self.nearest_gata6)
            nearest_nanog_cuda = cuda.to_device(self.nearest_nanog)
            nearest_diff_cuda = cuda.to_device(self.nearest_diff)

            # allocate threads and blocks for gpu memory "threads per block" and "blocks per grid"
            tpb = 72
            bpg = math.ceil(self.number_cells / tpb)

            # call the cuda kernel with new gpu arrays
            backend.nearest_gpu[bpg, tpb](bin_locations_cuda, locations_cuda, bins_cuda, bins_help_cuda, distance_cuda,
                                          if_diff_cuda, gata6_cuda, nanog_cuda, nearest_gata6_cuda, nearest_nanog_cuda,
                                          nearest_diff_cuda)

            # return the only the following array(s) back from the gpu
            gata6 = nearest_gata6_cuda.copy_to_host()
            nanog = nearest_nanog_cuda.copy_to_host()
            diff = nearest_diff_cuda.copy_to_host()

        # call the cpu version
        else:
            gata6, nanog, diff = backend.nearest_cpu(self.number_cells, bin_locations, self.locations, bins, bins_help,
                                                     distance, if_diff, self.GATA6, self.NANOG, self.nearest_gata6,
                                                     self.nearest_nanog, self.nearest_diff)

        # revalue the array holding the indices of nearest cells of given type
        self.nearest_gata6 = gata6
        self.nearest_nanog = nanog
        self.nearest_diff = diff

    @record_time
    def apply_forces(self, one_step=False, motility=True):
        """ Calls multiple methods used to move the cells to a
            state of physical equilibrium between repulsive,
            adhesive, and motility forces.
        """
        # the viscosity of the medium in Ns/m, used for stokes friction
        viscosity = 10000

        # this method can be called by update_queue() to better represent asynchronous division
        if one_step:
            total_steps, last_dt = 1, self.move_dt    # move once based on move_dt

        # otherwise run normally
        else:
            # calculate the number of steps and the last step time if it doesn't divide nicely
            steps, last_dt = divmod(self.step_dt, self.move_dt)
            total_steps = int(steps) + 1    # add extra step for the last dt, if divides nicely last_dt will equal zero

        # if motility parameter is False, an array of zeros will be used for the motility forces
        if motility:
            motility_forces = self.motility_forces
        else:
            motility_forces = np.zeros_like(self.motility_forces)

        # go through all move steps, calculating the physical interactions and applying the forces
        for step in range(total_steps):
            # update graph for pairs of contacting cells
            self.jkr_neighbors()

            # calculate the JKR forces based on the graph
            self.jkr_forces_func()

            # apply both the JKR forces and the motility forces
            # if on the last step use, that dt
            if step == total_steps - 1:
                move_dt = last_dt
            else:
                move_dt = self.move_dt

            # send the following as arrays to the gpu
            if self.parallel:
                # turn the following into arrays that can be interpreted by the gpu
                jkr_forces_cuda = cuda.to_device(self.jkr_forces)
                motility_forces_cuda = cuda.to_device(motility_forces)
                locations_cuda = cuda.to_device(self.locations)
                radii_cuda = cuda.to_device(self.radii)
                viscosity_cuda = cuda.to_device(viscosity)
                size_cuda = cuda.to_device(self.size)
                move_dt_cuda = cuda.to_device(move_dt)

                # allocate threads and blocks for gpu memory "threads per block" and "blocks per grid"
                tpb = 72
                bpg = math.ceil(self.number_cells / tpb)

                # call the cuda kernel with new gpu arrays
                backend.apply_forces_gpu[bpg, tpb](jkr_forces_cuda, motility_forces_cuda, locations_cuda, radii_cuda,
                                                   viscosity_cuda, size_cuda, move_dt_cuda)

                # return the only the following array(s) back from the gpu
                new_locations = locations_cuda.copy_to_host()

            # call the cpu version
            else:
                new_locations = backend.apply_forces_cpu(self.number_cells, self.jkr_forces, motility_forces,
                                                         self.locations, self.radii, viscosity, self.size,
                                                         move_dt)

            # update the locations and reset the JKR forces back to zero
            self.locations = new_locations
            self.jkr_forces[:, :] = 0

        # reset motility forces back to zero
        self.motility_forces[:, :] = 0

    def jkr_neighbors(self):
        """ For all cells, determines which cells will have physical
            interactions with other cells and puts this information
            into a graph.
        """
        # radius of search (meters) in which neighbors will have physical interactions, double the max cell radius
        jkr_distance = 2 * self.max_radius

        # if a static variable has not been created to hold the maximum number of neighbors for a cell, create one
        if not hasattr(Functions.jkr_neighbors, "max_neighbors"):
            # begin with a low number of neighbors that can be revalued if the max neighbors exceeds this value
            Functions.jkr_neighbors.max_neighbors = 5

        # if a static variable has not been created to hold the maximum number of cells in a bin, create one
        if not hasattr(Functions.jkr_neighbors, "max_cells"):
            # begin with a low number of cells that can be revalued if the max number of cells exceeds this value
            Functions.jkr_neighbors.max_cells = 5

        # this will run once if all edges are included in edge_holder, breaking the loop. if not, this will
        # run a second time with an updated value for the number of predicted neighbors such that all edges are included
        bins, bins_help, bin_locations, max_cells = self.assign_bins(jkr_distance, Functions.jkr_neighbors.max_cells)

        # update the value of the max number of cells in a bin
        Functions.jkr_neighbors.max_cells = max_cells

        # this will run once and if all edges are included in edge_holder, the loop will break. if not this will
        # run a second time with an updated value for number of predicted neighbors such that all edges are included
        while True:
            # create array used to hold edges, array to say where edges are, and array to count the edges per cell
            length = self.number_cells * Functions.jkr_neighbors.max_neighbors
            edge_holder = np.zeros((length, 2), dtype=int)
            if_edge = np.zeros(length, dtype=bool)
            edge_count = np.zeros(self.number_cells, dtype=int)

            # send the following as arrays to the gpu
            if self.parallel:
                # turn the following into arrays that can be interpreted by the gpu
                bin_locations_cuda = cuda.to_device(bin_locations)
                locations_cuda = cuda.to_device(self.locations)
                radii_cuda = cuda.to_device(self.radii)
                bins_cuda = cuda.to_device(bins)
                bins_help_cuda = cuda.to_device(bins_help)
                edge_holder_cuda = cuda.to_device(edge_holder)
                if_edge_cuda = cuda.to_device(if_edge)
                edge_count_cuda = cuda.to_device(edge_count)
                max_neighbors_cuda = cuda.to_device(Functions.jkr_neighbors.max_neighbors)

                # allocate threads and blocks for gpu memory "threads per block" and "blocks per grid"
                tpb = 72
                bpg = math.ceil(self.number_cells / tpb)

                # call the cuda kernel with new gpu arrays
                backend.jkr_neighbors_gpu[bpg, tpb](bin_locations_cuda, locations_cuda, radii_cuda, bins_cuda,
                                                    bins_help_cuda, edge_holder_cuda, if_edge_cuda, edge_count_cuda,
                                                    max_neighbors_cuda)

                # return the only the following array(s) back from the gpu
                edge_holder = edge_holder_cuda.copy_to_host()
                if_edge = if_edge_cuda.copy_to_host()
                edge_count = edge_count_cuda.copy_to_host()

            # call the jit cpu version
            else:
                edge_holder, if_edge, edge_count = backend.jkr_neighbors_cpu(self.number_cells, bin_locations,
                                                                             self.locations, self.radii, bins,
                                                                             bins_help, edge_holder, if_edge,
                                                                             edge_count,
                                                                             Functions.jkr_neighbors.max_neighbors)

            # either break the loop if all neighbors were accounted for or revalue the maximum number of neighbors
            # based on the output of the function call and double it
            max_neighbors = np.amax(edge_count)
            if Functions.jkr_neighbors.max_neighbors >= max_neighbors:
                break
            else:
                Functions.jkr_neighbors.max_neighbors = max_neighbors * 2

        # reduce the edges to only nonzero edges
        edge_holder = edge_holder[if_edge]

        # add the edges and simplify the graph as this graph is never cleared due to its use for holding adhesive JKR
        # bonds from step to step
        self.jkr_graph.add_edges(edge_holder)
        self.jkr_graph.simplify()

    def jkr_forces_func(self):
        """ Goes through all of "JKR" edges and quantifies any
            resulting adhesive or repulsion forces between
            pairs of cells.
        """
        # contact mechanics parameters that rarely change
        adhesion_const = 0.000107    # the adhesion constant in kg/s from P Pathmanathan et al.
        poisson = 0.5    # Poisson's ratio for the cells, 0.5 means incompressible
        youngs = 1000    # Young's modulus for the cells in Pa

        # get the edges as a numpy array, count them, and create an array used to delete edges from the JKR graph
        jkr_edges = np.array(self.jkr_graph.get_edgelist())
        number_edges = len(jkr_edges)
        delete_edges = np.zeros(number_edges, dtype=bool)

        # only continue if edges exist, if no edges compiled functions will raise errors
        if number_edges > 0:
            # send the following as arrays to the gpu
            if self.parallel:
                # turn the following into arrays that can be interpreted by the gpu
                jkr_edges_cuda = cuda.to_device(jkr_edges)
                delete_edges_cuda = cuda.to_device(delete_edges)
                locations_cuda = cuda.to_device(self.locations)
                radii_cuda = cuda.to_device(self.radii)
                forces_cuda = cuda.to_device(self.jkr_forces)
                poisson_cuda = cuda.to_device(poisson)
                youngs_cuda = cuda.to_device(youngs)
                adhesion_const_cuda = cuda.to_device(adhesion_const)

                # allocate threads and blocks for gpu memory "threads per block" and "blocks per grid"
                tpb = 72
                bpg = math.ceil(number_edges / tpb)

                # call the cuda kernel with new gpu arrays
                backend.jkr_forces_gpu[bpg, tpb](jkr_edges_cuda, delete_edges_cuda, locations_cuda, radii_cuda,
                                                 forces_cuda, poisson_cuda, youngs_cuda, adhesion_const_cuda)

                # return the only the following array(s) back from the gpu
                forces = forces_cuda.copy_to_host()
                delete_edges = delete_edges_cuda.copy_to_host()

            # call the cpu version
            else:
                forces, delete_edges = backend.jkr_forces_cpu(number_edges, jkr_edges, delete_edges, self.locations,
                                                              self.radii, self.jkr_forces, poisson, youngs,
                                                              adhesion_const)

            # update the jkr edges to remove any edges that have be broken and update the JKR forces array
            delete_edges_indices = np.arange(number_edges)[delete_edges]
            self.jkr_graph.delete_edges(delete_edges_indices)
            self.jkr_forces = forces

    @record_time
    def update_diffusion(self, gradient_name, diffuse_const=None, diffuse_dt=None):
        """ Approximates the diffusion of the morphogen for the
            extracellular gradient specified.
        """
        # if no parameter specified for diffusion constant use the one in self object
        if diffuse_const is None:
            diffuse_const = self.diffuse_const

        # if no parameter specified for diffusion time step use the one in self object
        if diffuse_dt is None:
            diffuse_dt = self.diffuse_dt

        # the self holds all gradients are 3D arrays for simplicity, get the gradient as a 2D array
        gradient = self.__dict__[gradient_name][:, :, 0]

        # set max and min concentration values
        gradient[gradient > self.max_concentration] = self.max_concentration
        gradient[gradient < 0] = 0

        # pad the sides of the array with zeros for holding ghost points
        base = np.pad(gradient, 1)

        # calculate the number of steps and the last step time if it doesn't divide nicely
        steps, last_dt = divmod(self.step_dt, self.diffuse_dt)
        steps = int(steps) + 1   # make sure steps is an int, add extra step for the last dt if it's less

        # call the JIT diffusion function
        gradient = backend.update_diffusion_jit(base, steps, diffuse_dt, last_dt, diffuse_const, self.spat_res2)

        # degrade the morphogen concentrations
        gradient *= 1 - self.degradation

        # update the self gradient array
        self.__dict__[gradient_name][:, :, 0] = gradient

    @record_time
    def update_queue(self):
        """ Adds and removes cells to and from the self
            either all at once or in "groups".
        """
        # get the number of cells being added and removed
        num_added = len(self.cells_to_divide)
        num_removed = len(self.cells_to_remove)

        # print how many cells are being added/removed during a given step
        print("Adding " + str(num_added) + " cells...")
        print("Removing " + str(num_removed) + " cells...")

        # -------------------- Division --------------------
        # extend each of the arrays by how many cells being added
        for name in self.cell_array_names:
            # copy the indices of the cell array for the dividing cells
            copies = self.__dict__[name][self.cells_to_divide]

            # if the instance variable is 1-dimensional
            if self.__dict__[name].ndim == 1:
                # add the copies to the end of the array
                self.__dict__[name] = np.concatenate((self.__dict__[name], copies))

            # if the instance variable is 2-dimensional
            else:
                # add the copies to the end of the array
                self.__dict__[name] = np.concatenate((self.__dict__[name], copies), axis=0)

        # go through each of the dividing cells
        for i in range(num_added):
            # get the indices of the mother cell and the daughter cell
            mother_index = self.cells_to_divide[i]
            daughter_index = self.number_cells

            # move the cells to new positions
            division_position = self.random_vector() * (self.max_radius - self.min_radius)
            self.locations[mother_index] += division_position
            self.locations[daughter_index] -= division_position

            # reduce both radii to minimum size (representative of a divided cell) and set the division counters to zero
            self.radii[mother_index] = self.radii[daughter_index] = self.min_radius
            self.div_counters[mother_index] = self.div_counters[daughter_index] = 0

            # go through each graph adding the number of dividing cells
            for graph_name in self.graph_names:
                self.__dict__[graph_name].add_vertex()

            # update the number of cells in the self
            self.number_cells += 1

            # if not adding all of the cells at once
            if self.group != 0:
                # Cannot add all of the new cells, otherwise several cells are likely to be added in
                #   close proximity to each other at later time steps. Such addition, coupled with
                #   handling collisions, make give rise to sudden changes in overall positions of
                #   cells within the self. Instead, collisions are handled after 'group' number
                #   of cells are added. - Daniel Cruz

                # if the current number added is divisible by the group number
                if (i + 1) % self.group == 0:
                    # run the following once to better simulate asynchronous division
                    self.apply_forces(one_step=True, motility=False)

        # -------------------- Death --------------------
        # get the indices of the cells leaving the self
        indices = self.cells_to_remove

        # go through the cell arrays remove the indices
        for name in self.cell_array_names:
            # if the array is 1-dimensional
            if self.__dict__[name].ndim == 1:
                self.__dict__[name] = np.delete(self.__dict__[name], indices)
            # if the array is 2-dimensional
            else:
                self.__dict__[name] = np.delete(self.__dict__[name], indices, axis=0)

        # automatically update the graphs and change the number of cells
        for graph_name in self.graph_names:
            self.__dict__[graph_name].delete_vertices(indices)
        self.number_cells -= num_removed

        # clear the arrays for the next step
        self.cells_to_divide = np.array([], dtype=int)
        self.cells_to_remove = np.array([], dtype=int)

    def assign_bins(self, distance, max_cells):
        """ Generalizes cell locations to a bin within lattice imposed on
            the cell space, used for a parallel fixed-radius neighbor search.
        """
        # If there is enough space for all cells that should be in a bin, break out of the loop. If there isn't
        # update the amount of needed space and put all the cells in bins. This will run once if the prediction
        # of max neighbors suffices, twice if it isn't right the first time.
        while True:
            # calculate the size of the array used to represent the bins and the bins helper array, include extra bins
            # for cells that may fall outside of the space
            bins_help_size = np.ceil(self.size / distance).astype(int) + 3
            bins_size = np.append(bins_help_size, max_cells)

            # create the arrays for "bins" and "bins_help"
            bins_help = np.zeros(bins_help_size, dtype=int)  # holds the number of cells currently in a bin
            bins = np.empty(bins_size, dtype=int)  # holds the indices of cells in a bin

            # generalize the cell locations to bin indices and offset by 1 to prevent missing cells that fall out of the
            # self space
            bin_locations = np.floor_divide(self.locations, distance).astype(int)
            bin_locations += 1

            # use jit function to speed up placement of cells
            bins, bins_help = assign_bins_jit(self.number_cells, bin_locations, bins, bins_help)

            # either break the loop if all cells were accounted for or revalue the maximum number of cells based on
            # the output of the function call and double it future calls
            new_max_cells = np.amax(bins_help)
            if max_cells >= new_max_cells:
                break
            else:
                max_cells = new_max_cells * 2  # double to prevent continual updating

        return bins, bins_help, bin_locations, max_cells

    def get_concentration(self, gradient_name, index):
        """ Get the concentration of a gradient for a cell's
            location. Currently this uses the nearest method.
        """
        # get the gradient array from the Simulation instance
        gradient = self.__dict__[gradient_name]

        # find the nearest diffusion point
        half_indices = np.floor(2 * self.locations[index] / self.spat_res)
        indices = np.ceil(half_indices / 2).astype(int)
        x, y, z = indices[0], indices[1], indices[2]

        # return the value of the gradient at the diffusion point
        return gradient[x][y][z]

    def adjust_morphogens(self, gradient_name, index, amount, mode):
        """ Adjust the concentration of the gradient based on
            the amount, location of cell, and mode.
        """
        # get the gradient array from the Simulation instance
        gradient = self.__dict__[gradient_name]

        # use the nearest method similar to the get_concentration()
        if mode == "nearest":
            # find the nearest diffusion point
            half_indices = np.floor(2 * self.locations[index] / self.spat_res)
            indices = np.ceil(half_indices / 2).astype(int)
            x, y, z = indices[0], indices[1], indices[2]

            # add the specified amount to the nearest diffusion point
            gradient[x][y][z] += amount

        # use the distance dependent method for adding concentrations, not optimized yet...
        elif mode == "distance":
            # divide the location for a cell by the spatial resolution then take the floor function of it
            indices = np.floor(self.locations[index] / self.spat_res).astype(int)
            x, y, z = indices[0], indices[1], indices[2]

            # get the four nearest points to the cell in 2D and make array for holding distances
            diffusion_points = np.array([[x, y, 0], [x + 1, y, 0], [x, y + 1, 0], [x + 1, y + 1, 0]], dtype=int)
            distances = -1 * np.ones(4, dtype=float)

            # hold the sum of the reciprocals of the distances
            total = 0

            # get the gradient size and handle each of the four nearest points
            gradient_size = self.gradient_size
            for i in range(4):
                # check that the diffusion point is not outside the space
                if diffusion_points[i][0] < gradient_size[0] and diffusion_points[i][1] < gradient_size[1]:
                    # if ok, calculate magnitude of the distance from the cell to it
                    point_location = diffusion_points[i] * self.spat_res
                    mag = np.linalg.norm(self.locations[index] - point_location)
                    if mag <= self.max_radius:
                        # save the distance and if the cell is not on top of the point add the reciprocal
                        distances[i] = mag
                        if mag != 0:
                            total += 1 / mag

            # add morphogen to each diffusion point that falls within the cell radius
            for i in range(4):
                x, y, z = diffusion_points[i][0], diffusion_points[i][1], 0
                # if on top of diffusion point add all of the concentration
                if distances[i] == 0:
                    gradient[x][y][z] += amount
                # if in radius add proportional amount
                elif distances[i] != -1:
                    gradient[x][y][z] += amount / (distances[i] * total)
                else:
                    pass

        # if some other mode
        else:
            raise Exception("Unknown mode for the adjust_morphogens() method")

    def add_cells(self, number, cell_type=None):
        """ Add cells to the Simulation object and potentially add a cell type
            with bounds for defining alternative initial parameters.

                number (int): the number of cells being added to the Simulation object
                cell_type (str): the name of a cell type that can be used by cell_array() to only apply
                    initial parameters to these cells, instead of the entire array.
        """
        # add specified number of cells to each graph
        for graph_name in self.graph_names:
            self.__dict__[graph_name].add_vertices(number)

        # update the running number of cells and determine bounds for slice if cell_type is used
        begin = self.number_cells
        self.number_cells += number

        # if a cell type name is passed, hold the slice bounds for that particular cell type
        if cell_type is not None:
            self.cell_types[cell_type] = (begin, self.number_cells)

    def cell_array(self, array_name, cell_type=None, dtype=float, vector=None, func=None, override=None):
        """ Create a cell array in the Simulation object used to hold values
            for all cells and optionally specify initial parameters.

                array_name (str): the name of the variable made for the cell array in the Simulation object
                cell_type (str): see add_cells()
                dtype (object): the data type of the array, defaults to float
                vector (int): the length of the vector for each cell in the array
                func (object): a function called for each index of the array to specify initial parameters
                override (array): use the array passed instead of generating a new array
        """
        # if using existing array for cell array
        if override is not None:
            # make sure array have correct length, otherwise raise error
            if override.shape[0] != self.number_cells:
                raise Exception("Length of override array does not match number of cells in simulation!")

            # use the array and add to list of cell array names
            else:
                self.__dict__[array_name] = override
                self.cell_array_names.append(array_name)

        # otherwise make sure a default cell array exists for initial parameters
        else:
            # if no cell array in Simulation object, make one
            if not hasattr(self, array_name):
                # add the array name to a list for automatic addition/removal when cells divide/die
                self.cell_array_names.append(array_name)

                # get the dimensions of the array
                if vector is None:
                    size = self.number_cells  # 1-dimensional array
                else:
                    size = (self.number_cells, vector)  # 2-dimensional array (1-dimensional of vectors)

                # if using python string data type, use object data type instead
                if dtype == str or dtype == object:
                    # create cell array in Simulation object with NoneType as default value
                    self.__dict__[array_name] = np.empty(size, dtype=object)

                else:
                    # create cell array in Simulation object, with zeros as default values
                    self.__dict__[array_name] = np.zeros(size, dtype=dtype)

        # if no cell type parameter passed
        if cell_type is None:
            # if function is passed, apply initial parameter
            if func is not None:
                for i in range(self.number_cells):
                    self.__dict__[array_name][i] = func()

        # otherwise a cell type is passed
        else:
            # get the bounds of the slice
            begin = self.cell_types[cell_type][0]
            end = self.cell_types[cell_type][1]

            # if function is passed, apply initial parameter to slice
            if func is not None:
                for i in range(begin, end):
                    self.__dict__[array_name][i] = func()

    def random_vector(self):
        """ Computes a random vector on the unit sphere centered
            at the origin.
        """
        # random angle on the cell
        theta = r.random() * 2 * math.pi

        # 2D vector: [x, y, 0]
        if self.size[2] == 0:
            return np.array([math.cos(theta), math.sin(theta), 0])

        # 3D vector: [x, y, z]
        else:
            phi = r.random() * 2 * math.pi
            radius = math.cos(phi)
            return np.array([radius * math.cos(theta), radius * math.sin(theta), math.sin(phi)])