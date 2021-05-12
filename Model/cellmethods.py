import numpy as np
import random as r
import math
from numba import cuda

from backend import *


class CellMethods:
    """ The methods in this class are meant to be inherited by the CellSimulation
        class so that CellSimulation objects can call these methods.
    """
    @record_time
    def cell_death(self):
        """ Remove any cells that meet the criteria for cell death.
        """
        # create boolean array to mark cells to be removed
        agents_to_remove = np.zeros(self.number_agents, dtype=bool)

        # determine which cells are being removed
        for index in range(self.number_agents):
            # checks to see if cell is pluripotent
            if self.states[index] == 0:
                # gets the number of neighbors for a cell, increasing the death counter if not enough neighbors
                if len(self.neighbor_graph.neighbors(index)) < self.lonely_thresh:
                    self.death_counters[index] += 1

                # if not, reset the death counter back to zero
                else:
                    self.death_counters[index] = 0

                # mark the cell for removal if it meets the parameters
                if self.death_counters[index] >= self.death_thresh:
                    agents_to_remove[index] = 1

        # get indices of cells to remove with Boolean mask
        indices = np.arange(self.number_agents)[agents_to_remove]

        # get the number of cells being removed, subtract from count, and print out to terminal
        num_removed = len(indices)
        self.number_agents -= num_removed
        print("Removing " + str(num_removed) + " agents...")

        # go through the cell arrays and remove the indices
        for name in self.agent_array_names:
            # if the array is 1-dimensional
            if self.__dict__[name].ndim == 1:
                self.__dict__[name] = np.delete(self.__dict__[name], indices)

            # if the array is 2-dimensional
            else:
                self.__dict__[name] = np.delete(self.__dict__[name], indices, axis=0)

        # remove the indices from each graph
        for graph_name in self.graph_names:
            self.__dict__[graph_name].delete_vertices(indices)

    @record_time
    def cell_division(self):
        """ Increases the cell division counter and if the cell meets criteria
            mark it for division.
        """
        # create boolean array to mark dividing cells
        agents_to_divide = np.zeros(self.number_agents, dtype=bool)

        # determine which cells are dividing
        for index in range(self.number_agents):
            # stochastically increase the division counter by either 0 or 1
            self.div_counters[index] += r.randint(0, 1)

            # pluripotent cell
            if self.states[index] == 0:
                # check the division counter against the threshold, add to array if dividing
                if self.div_counters[index] >= self.pluri_div_thresh:
                    agents_to_divide[index] = 1

            # differentiated cell
            else:
                # check the division counter against the threshold, add to array if dividing
                if self.div_counters[index] >= self.diff_div_thresh:
                    # check for contact inhibition since differentiated
                    if len(self.neighbor_graph.neighbors(index)) < 6:
                        agents_to_divide[index] = 1

        # get indices of the dividing cells with Boolean mask
        indices = np.arange(self.number_agents)[agents_to_divide]

        # get the number of cells being added and print out to terminal
        num_added = len(indices)
        print("Adding " + str(num_added) + " agents...")

        # go through the cell arrays and add indices
        for name in self.agent_array_names:
            # copy the indices of the cell array data for the dividing cells
            copies = self.__dict__[name][indices]

            # if the instance variable is 1-dimensional
            if self.__dict__[name].ndim == 1:
                # add the copies to the end of the array
                self.__dict__[name] = np.concatenate((self.__dict__[name], copies))

            # if the instance variable is 2-dimensional
            else:
                # add the copies to the end of the array
                self.__dict__[name] = np.concatenate((self.__dict__[name], copies), axis=0)

        # go through each of the dividing cells, updating values for the mother and daughter cell
        for i in range(num_added):
            # get the indices of mother cell and daughter cell
            mother_index = indices[i]
            daughter_index = self.number_agents

            # move the cells to new positions
            division_position = self.random_vector() * (self.max_radius - self.min_radius)
            self.locations[mother_index] += division_position
            self.locations[daughter_index] -= division_position

            # set division counters to zero
            self.div_counters[mother_index] = 0
            self.div_counters[daughter_index] = 0

            # go through each graph, adding one new vertex at a time
            for graph_name in self.graph_names:
                self.__dict__[graph_name].add_vertex()

            # update the number of cells, adding one new agent at a time
            self.number_agents += 1

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
                    self.apply_forces(group_add=True)

    @record_time
    def cell_diff_surround(self):
        """ Simulates differentiated cells inducing the differentiation of a
            pluripotent cell.
        """
        for index in range(self.number_agents):
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
    def cell_growth(self):
        """ Simulates the growth of a cell currently linear, radius-based growth.
        """
        for index in range(self.number_agents):
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
        """ Stochastically updates the value for GATA6 and NANOG based on set
            probabilities.
        """
        for index in range(self.number_agents):
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
        """ Updates finite dynamical system variables.
        """
        for index in range(self.number_agents):
            # activate the following pathway based on if doxycycline has been induced yet (after 24 hours/48 steps)
            if self.current_step >= self.dox_step:
                # get the list of neighbors for the cell and the number of neighbors
                neighbors = self.neighbor_graph.neighbors(index)
                num_neighbors = len(neighbors)

                # get perceived FGF4 for self
                if num_neighbors != 0:
                    perceived_FGF4 = (1 + r.gauss(0, 1)) * (self.FGF4[index] / num_neighbors)
                else:
                    perceived_FGF4 = (1 + r.gauss(0, 1)) * self.FGF4[index]

                # go through neighbors to get perceived FGF4 morphogen
                for i in range(num_neighbors):
                    # include gaussian noise
                    perceived_FGF4 += (1 + r.gauss(0, 1)) * (self.FGF4[neighbors[i]] / num_neighbors)

                # floor perceived FGF4 to nearest int, make sure it's max is field - 1 and >= 0
                perceived_FGF4 = int(perceived_FGF4)
                if perceived_FGF4 > self.field - 1:
                    perceived_FGF4 = self.field - 1
                elif perceived_FGF4 < 0:
                    perceived_FGF4 = 0

                # if updating the DDS values this step
                if self.fds_counters[index] % self.fds_thresh == 0:
                    # get the current DDS values of the cell
                    x1 = perceived_FGF4
                    x2 = self.FGFR[index]
                    x3 = self.ERK[index]
                    x4 = self.GATA6[index]
                    x5 = self.NANOG[index]

                    # if the DDS is boolean (BN_9)
                    if self.field == 2:
                        self.FGF4[index] = x5
                        self.FGFR[index] = (1 + x5 + (x4 * x5)) % 2
                        self.ERK[index] = (x1 * x2) % 2
                        self.GATA6[index] = (x3 + x4 + (x3 * x4) + (x3 * x5) + (x4 * x5) + (x3 * x4 * x5)) % 2
                        self.NANOG[index] = (x5 + (x3 * x5) + (x4 * x5) + (x3 * x4 * x5)) % 2

                    # otherwise assume ternary
                    else:
                        self.FGF4[index] = x5
                        self.FGFR[index] = (x1 * x4 * ((2*x1 + 1) * (2*x4 + 1) + x1 * x4)) % 3
                        self.ERK[index] = x2 % 3
                        self.GATA6[index] = ((x4**2) * (x5 + 1) + (x5**2) * (x4 + 1) + 2*x5 + 1) % 3
                        self.NANOG[index] = (x5**2 + x5 * (x5 + 1) * (x3 * (2*x4**2 + 2*x3 + 1) +
                                             x4*(2*x3**2 + 2*x4 + 1)) + (2*x3**2 + 1) * (2*x4**2 + 1)) % 3

                # increase the discrete dynamical system counter
                self.fds_counters[index] += 1

    @record_time
    def cell_differentiate(self):
        """ Based on GATA6 and NANOG values, stochastically increase differentiation
            counter and/or differentiate.
        """
        for index in range(self.number_agents):
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
        """ Gives the cells a motive force depending on set rules for
            the cell types.
        """
        # this is the motility force of the cells
        motility_force = 0.000000002

        # loop over all of the cells
        for index in range(self.number_agents):
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
                            vector_holder += self.locations[neighbors[i]] - self.locations[index]

                    # if there is at least one nanog high cell move away from it
                    if count > 0:
                        # get the normalized vector
                        normal = normal_vector(vector_holder)

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
                                    vector_holder += self.locations[neighbors[i]] - self.locations[index]

                            # if there is at least differentiated cell move toward it
                            if count > 0:
                                # get the normalized vector
                                normal = normal_vector(vector_holder)

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
                                    vector_holder += self.locations[neighbors[i]] - self.locations[index]

                            # if there is at least one nanog high cell move away from it
                            if count > 0:
                                # get the normalized vector
                                normal = normal_vector(vector_holder)

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
                                vector_holder += self.locations[neighbors[i]] - self.locations[index]

                        # if there is at least one nanog high cell move toward it
                        if count > 0:
                            # get the normalized vector
                            normal = normal_vector(vector_holder)

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
    def apply_forces(self, group_add=False):
        """ Calls multiple methods used to move the cells to a state of physical
            equilibrium between repulsive, adhesive, and motility forces.
        """
        # the viscosity of the medium in Ns/m, used for stokes friction
        viscosity = 10000

        # if this method is being used by cell_division(), move the cells slightly without motility forces
        if group_add:
            total_steps = 1
            last_dt = self.move_dt
            motility_forces = np.zeros_like(self.motility_forces)

        # otherwise apply motility forces and iteratively move the cells
        else:
            # calculate the number of steps and the last step time if it doesn't divide nicely
            steps, last_dt = divmod(self.step_dt, self.move_dt)
            total_steps = int(steps) + 1    # add extra step for the last dt, if divides nicely last_dt will equal zero
            motility_forces = self.motility_forces

        # go through all move steps, calculating the physical interactions and applying the forces
        for step in range(total_steps):
            # update graph for pairs of contacting cells
            self.get_neighbors("jkr_graph", 2 * self.max_radius, clear=False)

            # calculate the JKR forces based on the graph
            self.calculate_jkr()

            # apply both the JKR forces and the motility forces
            # if on the last step use, that dt
            if step == total_steps - 1:
                move_dt = last_dt
            else:
                move_dt = self.move_dt

            # call the nvidia gpu version
            if self.parallel:
                # allow the following arrays to be sent/returned by the CUDA kernel
                locations = cuda.to_device(self.locations)

                # allocate threads and blocks for gpu memory "threads per block" and "blocks per grid"
                tpb = 72
                bpg = math.ceil(self.number_agents / tpb)

                # call the cuda kernel with new gpu arrays
                apply_forces_gpu[bpg, tpb](cuda.to_device(self.jkr_forces), cuda.to_device(motility_forces), locations,
                                           cuda.to_device(self.radii), cuda.to_device(viscosity),
                                           cuda.to_device(self.size), cuda.to_device(move_dt))

                # return the only the following array(s) back from the gpu
                new_locations = locations.copy_to_host()

            # call the cpu version
            else:
                new_locations = apply_forces_cpu(self.number_agents, self.jkr_forces, motility_forces, self.locations,
                                                 self.radii, viscosity, self.size, move_dt)

            # update the locations and reset the JKR forces back to zero
            self.locations = new_locations
            self.jkr_forces[:, :] = 0

        # reset motility forces back to zero
        self.motility_forces[:, :] = 0

    def calculate_jkr(self):
        """ Goes through all of "JKR" edges and quantifies any resulting
            adhesive or repulsion forces between pairs of cells.
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
                delete_edges = cuda.to_device(delete_edges)
                forces = cuda.to_device(self.jkr_forces)

                # allocate threads and blocks for gpu memory "threads per block" and "blocks per grid"
                tpb = 72
                bpg = math.ceil(number_edges / tpb)

                # call the cuda kernel with new gpu arrays
                jkr_forces_gpu[bpg, tpb](cuda.to_device(jkr_edges), delete_edges, cuda.to_device(self.locations),
                                         cuda.to_device(self.radii), forces, cuda.to_device(poisson),
                                         cuda.to_device(youngs), cuda.to_device(adhesion_const))

                # return the only the following array(s) back from the gpu
                forces = forces.copy_to_host()
                delete_edges = delete_edges.copy_to_host()

            # call the cpu version
            else:
                forces, delete_edges = jkr_forces_cpu(number_edges, jkr_edges, delete_edges, self.locations, self.radii,
                                                      self.jkr_forces, poisson, youngs, adhesion_const)

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
        gradient = update_diffusion_jit(base, steps, diffuse_dt, last_dt, diffuse_const, self.spat_res2)

        # degrade the morphogen concentrations
        gradient *= 1 - self.degradation

        # update the self gradient array
        self.__dict__[gradient_name][:, :, 0] = gradient

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

    def adjust_morphogens(self, gradient_name, index, amount):
        """ Adjust the concentration of the gradient based on
            the amount and the location of the cell.
        """
        # get the gradient array from the Simulation instance
        gradient = self.__dict__[gradient_name]

        # divide the location for a cell by the spatial resolution then take the floor function of it
        indices = np.floor(self.locations[index] / self.spat_res).astype(int)
        x, y, z = indices[0], indices[1], indices[2]

        # get the four nearest points to the cell in 2D and make array for holding distances
        points = np.array([[x, y, 0], [x + 1, y, 0], [x, y + 1, 0], [x + 1, y + 1, 0]], dtype=int)
        if_nearby = np.zeros(4, dtype=bool)

        # go through potential nearby diffusion points
        for i in range(4):
            # get point and make sure it's in bounds
            point = points[i]
            if point[0] < self.gradient_size[0] and point[1] < self.gradient_size[1]:
                # get location of point
                point_location = point * self.spat_res

                # see if point is in diffuse radius, if so update if_nearby index to True
                if np.linalg.norm(self.locations[index] - point_location) < self.spat_res:
                    if_nearby[i] = 1

        # get the number of points within diffuse radius
        total_nearby = np.sum(if_nearby)
        point_amount = amount / total_nearby

        # go back through points adding morphogen
        for i in range(4):
            if if_nearby[i]:
                x, y, z = points[i][0], points[i][1], 0
                gradient[x][y][z] += point_amount
