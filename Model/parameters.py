import numpy as np
import random as r
import igraph
import math

from backend import *


class Simulation(Base):
    """ This object holds all of the important information about the simulation as it
        runs. Variables can be specified either directly or through the template files.
    """
    def __init__(self, paths, name):
        Base.__init__(self, paths, name)   # initialize the Base object instance variables
        """
        The following instance variables can be updated through template files located in the "templates"
        directory under the "Model" directory. The values must be specified in the .txt files as follows.

            (outputs.txt)
            1   How many frames per second of the output video that collects all step images? Ex. 6
            2   | 6 |
            3

        Note: extraneous spaces before or after the pipes will not affect the interpretation of the
        parameter. Use template_param(path-to-file, line number, data type) to read a specific line
        of a template file and interpret the value as the desired data type.

            self.fps = template_param(path, 2, float)
        """
        # ------------- general template file ------------------------------
        general_path = paths.templates + "general.txt"    # path to general.txt template file
        self.parallel = template_param(general_path, 5, bool)
        self.end_step = template_param(general_path, 8, int)
        self.num_nanog = template_param(general_path, 11, int)
        self.num_gata6 = template_param(general_path, 14, int)
        self.size = np.array(template_param(general_path, 17, tuple))
        self.order_66 = template_param(general_path, 20, str)
        # self.order_66 = commandline_param("-o", bool)

        # ------------- outputs template file ------------------------------
        outputs_path = paths.templates + "outputs.txt"    # path to outputs.txt template file
        self.output_values = template_param(outputs_path, 5, bool)
        self.output_tda = template_param(outputs_path, 9, bool)
        self.output_gradients = template_param(outputs_path, 12, bool)
        self.output_images = template_param(outputs_path, 15, bool)
        self.image_quality = template_param(outputs_path, 19, int)
        self.video_quality = template_param(outputs_path, 23, int)
        self.fps = template_param(outputs_path, 26, float)
        self.color_mode = template_param(outputs_path, 30, bool)

        # ------------- experimental template file -------------------------
        experimental_path = paths.templates + "experimental.txt"    # path to experimental.txt template file
        self.group = template_param(experimental_path, 5, int)
        self.dox_step = template_param(experimental_path, 9, int)
        self.guye_move = template_param(experimental_path, 13, bool)
        self.lonely_thresh = template_param(experimental_path, 17, int)

        # define any other instance variables that are not part of the template files

        # the temporal resolution for the simulation
        self.step_dt = 1800  # dt of each simulation step (1800 sec)
        self.move_dt = 180  # dt for incremental movement (180 sec)
        # self.diffuse_dt = 0.24  # dt for stable diffusion model (0.24 sec)
        self.diffuse_dt = 6.24  # dt for stable diffusion model (6 sec)

        # the field for the finite dynamical system
        self.field = 2

        # probability of randomly increasing FDS value to high
        self.GATA6_prob = 0.01
        self.NANOG_prob = 0.01

        # the rates (in steps) of division, differentiation, death, and finite dynamical system updating
        self.pluri_div_thresh = 36
        self.diff_div_thresh = 72
        self.pluri_to_diff = 72
        self.death_thresh = 144
        self.fds_thresh = 1

        # min and max radius lengths are used to calculate linear growth of the radius over time
        self.max_radius = 0.000005    # 5 um
        self.min_radius = self.max_radius / 2 ** 0.5    # half the area for max radius cell in 2D
        self.pluri_growth = (self.max_radius - self.min_radius) / self.pluri_div_thresh
        self.diff_growth = (self.max_radius - self.min_radius) / self.diff_div_thresh

        # create graph with instance variable jkr_graph for holding adhesions between contacting cells
        self.jkr_graph = igraph.Graph()
        self.graph_names.append("jkr_graph")

        # the spatial resolution of the space, the diffusion constant for the molecule gradients, the radius of
        # search for diffusion points, and the max concentration at a diffusion point
        self.spat_res = 0.00000707106
        self.spat_res2 = self.spat_res ** 2
        # self.diffuse_const = 0.00000000005    # 50 um^2/s
        self.diffuse_const = 0.000000000002  # 2 um^2/s
        self.max_concentration = 2

        # calculate the size of the array for the diffusion points and create gradient array(s)
        self.gradient_size = np.ceil(self.size / self.spat_res).astype(int) + 1
        self.fgf4_values = np.zeros(self.gradient_size, dtype=float)
        self.gradient_names = ["fgf4_values"]  # add names for automatic CSV output of gradients
        self.degradation = 0.1    # this will degrade the morphogen by this much at each step

        # self.fgf4_alt = np.zeros(self.gradient_size, dtype=float)    # for testing morphogen release methods
        # self.gradient_names = ["fgf4_values", "fgf4_alt"]    # add names for automatic CSV output of gradients

    def steps(self):
        """ Specify any Simulation instance methods called before/during/after
            the simulation, see example below.

            Example:
                self.before_steps()

                for self.current_step in range(self.beginning_step, self.end_step + 1):
                    self.during_steps()

                self.after_steps()
        """
        # Iterate over all steps specified in the Simulation object
        for self.current_step in range(self.beginning_step, self.end_step + 1):
            # Records model run time for the step and prints the current step/number of cells.
            self.info()

            # Finds the neighbors of each cell that are within a fixed radius and store this info in a graph.
            self.get_neighbors(distance=0.00001)    # double max cell radius

            # Updates cells by adjusting trackers for differentiation, division, growth, etc. based on intracellular,
            # intercellular, and extracellular conditions through a series of separate methods.
            # functions.cell_death(self)
            self.cell_diff_surround()
            self.cell_division()
            self.cell_growth()
            # self.cell_stochastic_update()
            self.cell_pathway()
            self.cell_differentiate()

            # Simulates diffusion the specified extracellular gradient via the forward time centered space method.
            self.update_diffusion("fgf4_values")
            # self.update_diffusion("fgf4_alt")    # for testing morphogen release methods

            # Adds/removes cells to/from the simulation either all together or in desired groups of cells. If done in
            # groups, the apply_forces() function will be used to better represent asynchronous division and death.
            self.update_queue()

            # Finds the nearest NANOG high, GATA6 high, and differentiated cells within a fixed radius. This provides
            # information that can be used for approximating cell motility.
            # self.nearest(distance=0.000015)    # triple max cell radius

            # Calculates the direction/magnitude of a cell's movement depending on a variety of factors such as state
            # and presence of neighbors.
            self.cell_motility()
            # self.eunbi_motility()

            # Through the series of methods, attempt to move the cells to a state of physical equilibrium between
            # adhesive and repulsive forces acting on the cells, while applying active motility forces.
            self.apply_forces()

            # Saves multiple forms of information about the simulation at the current step, including an image of the
            # space, CSVs with values of the cells, a temporary pickle of the Simulation object, and performance stats.
            # See the outputs.txt template file for turning off certain outputs.
            self.step_image()
            self.step_values(arrays=["locations", "FGFR", "ERK", "GATA6", "NANOG", "states", "diff_counters",
                                     "div_counters"])
            self.step_gradients()
            self.step_tda(in_pixels=True)
            self.temp()
            self.data()

        # Ends the simulation by creating a video from all of the step images
        self.create_video()

    def agent_initials(self):
        """ Add cells into the simulation and specify any values the cells should have.
            The cell arrays will default to float64, 1-dim arrays of zeros. Use the
            parameters to adjust the data type, 2-dim size, and initial conditions. The
            "cell_type" keyword is used to apply initial conditions to the group of cells
            marked with the same cell type in add_cells().
        """
        # Add the specified number of NANOG/GATA6 high cells and create cell type GATA6_high.
        self.add_agents(self.num_nanog)
        self.add_agents(self.num_gata6, agent_type="GATA6_high")

        # Create the following cell arrays with initial conditions.
        self.agent_array("locations", override=np.random.rand(self.number_agents, 3) * self.size)
        self.agent_array("radii")
        self.agent_array("motion", dtype=bool, func=lambda: True)
        self.agent_array("FGFR", dtype=int, func=lambda: r.randrange(0, self.field))
        self.agent_array("ERK", dtype=int, func=lambda: r.randrange(0, self.field))
        self.agent_array("GATA6", dtype=int)
        self.agent_array("NANOG", dtype=int, func=lambda: r.randrange(0, self.field))
        self.agent_array("states", dtype=int)
        self.agent_array("death_counters", dtype=int, func=lambda: r.randrange(0, self.death_thresh))
        self.agent_array("diff_counters", dtype=int, func=lambda: r.randrange(0, self.pluri_to_diff))
        self.agent_array("div_counters", dtype=int, func=lambda: r.randrange(0, self.pluri_div_thresh))
        self.agent_array("fds_counters", dtype=int, func=lambda: r.randrange(0, self.fds_thresh))
        self.agent_array("motility_forces", vector=3)
        self.agent_array("jkr_forces", vector=3)
        # self.agent_array("nearest_nanog", dtype=int, func=lambda: -1)
        # self.agent_array("nearest_gata6", dtype=int, func=lambda: -1)
        # self.agent_array( "nearest_diff", dtype=int, func=lambda: -1)

        # Update the number of cells marked with the "GATA6_high" cell type with alternative initial conditions.
        self.agent_array("GATA6", agent_type="GATA6_high", func=lambda: r.randrange(1, self.field))
        self.agent_array("NANOG", agent_type="GATA6_high", func=lambda: 0)

    @record_time
    def cell_death(self):
        """ Marks the cell for removal if it meets the criteria for cell death.
        """
        for index in range(self.number_agents):
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
                    self.agents_to_remove = np.append(self.agents_to_remove, index)

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
    def cell_division(self):
        """ Increases the cell division counter and if the cell meets criteria
            mark it for division.
        """
        for index in range(self.number_agents):
            # stochastically increase the division counter by either 0 or 1
            self.div_counters[index] += r.randint(0, 1)

            # pluripotent cell
            if self.states[index] == 0:
                # check the division counter against the threshold, add to array if dividing
                if self.div_counters[index] >= self.pluri_div_thresh:
                    self.agents_to_divide = np.append(self.agents_to_divide, index)

            # differentiated cell
            else:
                # check the division counter against the threshold, add to array if dividing
                if self.div_counters[index] >= self.diff_div_thresh:

                    # check for contact inhibition since differentiated
                    if len(self.neighbor_graph.neighbors(index)) < 6:
                        self.agents_to_divide = np.append(self.agents_to_divide, index)

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
        """ Updates finite dynamical system variables and extracellular conditions.
        """
        for index in range(self.number_agents):
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
                        fgf4_fds = 0  # FGF4 low
                    else:
                        fgf4_fds = 1  # FGF4 high

                # otherwise assume ternary for now
                else:
                    # base thresholds on the maximum concentrations
                    if fgf4_value < 1 / 3 * self.max_concentration:
                        fgf4_fds = 0  # FGF4 low
                    elif fgf4_value < 2 / 3 * self.max_concentration:
                        fgf4_fds = 1  # FGF4 medium
                    else:
                        fgf4_fds = 2  # FGF4 high

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
                        new_fgfr = (x1 * x4 * ((2 * x1 + 1) * (2 * x4 + 1) + x1 * x4)) % 3
                        new_erk = x2 % 3
                        new_gata6 = ((x4 ** 2) * (x5 + 1) + (x5 ** 2) * (x4 + 1) + 2 * x5 + 1) % 3
                        new_nanog = (x5 ** 2 + x5 * (x5 + 1) * (
                                    x3 * (2 * x4 ** 2 + 2 * x3 + 1) + x4 * (2 * x3 ** 2 + 2 * x4 + 1)) +
                                     (2 * x3 ** 2 + 1) * (2 * x4 ** 2 + 1)) % 3

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
    def eunbi_motility(self):
        """ Gives the cells a motive force depending on set rules for the
            cell types where these rules are closer to Eunbi's model.
        """
        # this is the motility force of the cells
        motility_force = 0.000000008

        # loop over all of the cells
        for index in range(self.number_agents):
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
                            normal = normal_vector(self.locations[nearest_index] - self.locations[index])
                            random = self.random_vector()
                            self.motility_forces[index] += (normal * -0.8 + random * 0.2) * motility_force

                        # if no nearby nanog high cells, move randomly
                        else:
                            self.motility_forces[index] += self.random_vector() * motility_force

                    # if the cell is gata6 high and nanog low
                    elif self.GATA6[index] > self.NANOG[index]:
                        # if there is a differentiated cell nearby, move toward it
                        if self.nearest_diff[index] != -1:
                            nearest_index = self.nearest_nanog[index]
                            normal = normal_vector(self.locations[nearest_index] - self.locations[index])
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
                            normal = normal_vector(self.locations[nearest_index] - self.locations[index])
                            random = self.random_vector()
                            self.motility_forces[index] += (normal * 0.8 + random * 0.2) * motility_force

                        # if there is a gata6 high cell nearby, move away from it
                        elif self.nearest_gata6[index] != -1:
                            nearest_index = self.nearest_nanog[index]
                            normal = normal_vector(self.locations[nearest_index] - self.locations[index])
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
    def nearest(self, distance=0.00002):
        """ Determines the nearest GATA6 high, NANOG high, and differentiated
            cell within a fixed radius for each cell.
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
            bin_locations = cuda.to_device(bin_locations)
            locations = cuda.to_device(self.locations)
            bins = cuda.to_device(bins)
            bins_help = cuda.to_device(bins_help)
            distance = cuda.to_device(distance)
            if_diff = cuda.to_device(if_diff)
            gata6 = cuda.to_device(self.GATA6)
            nanog = cuda.to_device(self.NANOG)
            nearest_gata6 = cuda.to_device(self.nearest_gata6)
            nearest_nanog = cuda.to_device(self.nearest_nanog)
            nearest_diff = cuda.to_device(self.nearest_diff)

            # allocate threads and blocks for gpu memory "threads per block" and "blocks per grid"
            tpb = 72
            bpg = math.ceil(self.number_agents / tpb)

            # call the cuda kernel with new gpu arrays
            nearest_gpu[bpg, tpb](bin_locations, locations, bins, bins_help, distance, if_diff, gata6, nanog,
                                  nearest_gata6, nearest_nanog, nearest_diff)

            # return the only the following array(s) back from the gpu
            gata6 = nearest_gata6.copy_to_host()
            nanog = nearest_nanog.copy_to_host()
            diff = nearest_diff.copy_to_host()

        # call the cpu version
        else:
            gata6, nanog, diff = nearest_cpu(self.number_agents, bin_locations, self.locations, bins, bins_help,
                                             distance, if_diff, self.GATA6, self.NANOG, self.nearest_gata6,
                                             self.nearest_nanog, self.nearest_diff)

        # revalue the array holding the indices of nearest cells of given type
        self.nearest_gata6 = gata6
        self.nearest_nanog = nanog
        self.nearest_diff = diff

    @record_time
    def apply_forces(self, one_step=False, motility=True):
        """ Calls multiple methods used to move the cells to a state of physical
            equilibrium between repulsive, adhesive, and motility forces.
        """
        # the viscosity of the medium in Ns/m, used for stokes friction
        viscosity = 10000

        # this method can be called by update_queue() to better represent asynchronous division
        if one_step:
            total_steps, last_dt = 1, self.move_dt  # move once based on move_dt

        # otherwise run normally
        else:
            # calculate the number of steps and the last step time if it doesn't divide nicely
            steps, last_dt = divmod(self.step_dt, self.move_dt)
            total_steps = int(steps) + 1  # add extra step for the last dt, if divides nicely last_dt will equal zero

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
            self.calculate_jkr()

            # apply both the JKR forces and the motility forces
            # if on the last step use, that dt
            if step == total_steps - 1:
                move_dt = last_dt
            else:
                move_dt = self.move_dt

            # send the following as arrays to the gpu
            if self.parallel:
                # turn the following into arrays that can be interpreted by the gpu
                jkr_forces = cuda.to_device(self.jkr_forces)
                motility_forces = cuda.to_device(motility_forces)
                locations = cuda.to_device(self.locations)
                radii = cuda.to_device(self.radii)
                viscosity = cuda.to_device(viscosity)
                size = cuda.to_device(self.size)
                move_dt = cuda.to_device(move_dt)

                # allocate threads and blocks for gpu memory "threads per block" and "blocks per grid"
                tpb = 72
                bpg = math.ceil(self.number_agents / tpb)

                # call the cuda kernel with new gpu arrays
                apply_forces_gpu[bpg, tpb](jkr_forces, motility_forces, locations, radii, viscosity, size, move_dt)

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

    def jkr_neighbors(self):
        """ For all cells, determines which cells will have physical interactions
            with other cells and puts this information into a graph.
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
            length = self.number_agents * Functions.jkr_neighbors.max_neighbors
            edge_holder = np.zeros((length, 2), dtype=int)
            if_edge = np.zeros(length, dtype=bool)
            edge_count = np.zeros(self.number_agents, dtype=int)

            # send the following as arrays to the gpu
            if self.parallel:
                # turn the following into arrays that can be interpreted by the gpu
                bin_locations = cuda.to_device(bin_locations)
                locations = cuda.to_device(self.locations)
                radii = cuda.to_device(self.radii)
                bins = cuda.to_device(bins)
                bins_help = cuda.to_device(bins_help)
                edge_holder = cuda.to_device(edge_holder)
                if_edge = cuda.to_device(if_edge)
                edge_count = cuda.to_device(edge_count)
                max_neighbors = cuda.to_device(Functions.jkr_neighbors.max_neighbors)

                # allocate threads and blocks for gpu memory "threads per block" and "blocks per grid"
                tpb = 72
                bpg = math.ceil(self.number_agents / tpb)

                # call the cuda kernel with new gpu arrays
                jkr_neighbors_gpu[bpg, tpb](bin_locations, locations, radii, bins, bins_help, edge_holder, if_edge,
                                            edge_count, max_neighbors)

                # return the only the following array(s) back from the gpu
                edge_holder = edge_holder.copy_to_host()
                if_edge = if_edge.copy_to_host()
                edge_count = edge_count.copy_to_host()

            # call the jit cpu version
            else:
                edge_holder, if_edge, edge_count = jkr_neighbors_cpu(self.number_agents, bin_locations, self.locations,
                                                                     self.radii, bins, bins_help, edge_holder, if_edge,
                                                                     edge_count, Functions.jkr_neighbors.max_neighbors)

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

    def calculate_jkr(self):
        """ Goes through all of "JKR" edges and quantifies any resulting
            adhesive or repulsion forces between pairs of cells.
        """
        # contact mechanics parameters that rarely change
        adhesion_const = 0.000107  # the adhesion constant in kg/s from P Pathmanathan et al.
        poisson = 0.5  # Poisson's ratio for the cells, 0.5 means incompressible
        youngs = 1000  # Young's modulus for the cells in Pa

        # get the edges as a numpy array, count them, and create an array used to delete edges from the JKR graph
        jkr_edges = np.array(self.jkr_graph.get_edgelist())
        number_edges = len(jkr_edges)
        delete_edges = np.zeros(number_edges, dtype=bool)

        # only continue if edges exist, if no edges compiled functions will raise errors
        if number_edges > 0:
            # send the following as arrays to the gpu
            if self.parallel:
                # turn the following into arrays that can be interpreted by the gpu
                jkr_edges = cuda.to_device(jkr_edges)
                delete_edges = cuda.to_device(delete_edges)
                locations = cuda.to_device(self.locations)
                radii = cuda.to_device(self.radii)
                forces = cuda.to_device(self.jkr_forces)
                poisson = cuda.to_device(poisson)
                youngs = cuda.to_device(youngs)
                adhesion_const = cuda.to_device(adhesion_const)

                # allocate threads and blocks for gpu memory "threads per block" and "blocks per grid"
                tpb = 72
                bpg = math.ceil(number_edges / tpb)

                # call the cuda kernel with new gpu arrays
                jkr_forces_gpu[bpg, tpb](jkr_edges, delete_edges, locations, radii, forces, poisson, youngs,
                                         adhesion_const)

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
        steps = int(steps) + 1  # make sure steps is an int, add extra step for the last dt if it's less

        # call the JIT diffusion function
        gradient = update_diffusion_jit(base, steps, diffuse_dt, last_dt, diffuse_const, self.spat_res2)

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
        num_added = len(self.agents_to_divide)
        num_removed = len(self.agents_to_remove)

        # print how many cells are being added/removed during a given step
        print("Adding " + str(num_added) + " cells...")
        print("Removing " + str(num_removed) + " cells...")

        # -------------------- Division --------------------
        # extend each of the arrays by how many cells being added
        for name in self.agent_array_names:
            # copy the indices of the cell array for the dividing cells
            copies = self.__dict__[name][self.agents_to_divide]

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
            mother_index = self.agents_to_divide[i]
            daughter_index = self.number_agents

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
                    self.apply_forces(one_step=True, motility=False)

        # -------------------- Death --------------------
        # get the indices of the cells leaving the self
        indices = self.agents_to_remove

        # go through the cell arrays remove the indices
        for name in self.agent_array_names:
            # if the array is 1-dimensional
            if self.__dict__[name].ndim == 1:
                self.__dict__[name] = np.delete(self.__dict__[name], indices)
            # if the array is 2-dimensional
            else:
                self.__dict__[name] = np.delete(self.__dict__[name], indices, axis=0)

        # automatically update the graphs and change the number of cells
        for graph_name in self.graph_names:
            self.__dict__[graph_name].delete_vertices(indices)
        self.number_agents -= num_removed

        # clear the arrays for the next step
        self.agents_to_divide = np.array([], dtype=int)
        self.agents_to_remove = np.array([], dtype=int)

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

    @record_time
    def step_image(self, background=(0, 0, 0), origin_bottom=True, fgf4_gradient=False):
        """ Creates an image of the simulation space. Note the imaging library
            OpenCV uses BGR instead of RGB.

            background -> (tuple) The color of the background image as BGR
            origin_bottom -> (bool) Location of origin True -> bottom/left, False -> top/left
            fgf4_gradient -> (bool) If outputting image of FGF4 gradient alongside step image
        """
        # only continue if outputting images
        if self.output_images:
            # get path and make sure directory exists
            directory_path = check_direct(self.paths.images)

            # get the size of the array used for imaging in addition to the scaling factor
            x_size = self.image_quality
            scale = x_size / self.size[0]
            y_size = math.ceil(scale * self.size[1])

            # create the cell space background image and apply background color
            image = np.zeros((y_size, x_size, 3), dtype=np.uint8)
            image[:, :] = background

            # if outputting gradient image, create it
            if fgf4_gradient:
                # normalize the concentration values and multiple by 255
                grad_image = 255 * self.fgf4_values[:, :, 0] / self.max_concentration
                grad_image = grad_image.astype(np.uint8)  # use unsigned int8

                # recolor the grayscale image into a colormap and resize to match the cell space image
                grad_image = cv2.applyColorMap(grad_image, cv2.COLORMAP_OCEAN)
                grad_image = cv2.resize(grad_image, (y_size, x_size), interpolation=cv2.INTER_NEAREST)

                # transpose the array to match the point location of OpenCV: (x, y) with origin top left
                grad_image = cv2.transpose(grad_image)

            # go through all of the cells
            for index in range(self.number_agents):
                # get xy coordinates and the axis lengths
                x, y = int(scale * self.locations[index][0]), int(scale * self.locations[index][1])
                major = int(scale * self.radii[index])
                minor = int(scale * self.radii[index])

                # color the cells according to the mode
                if self.color_mode:
                    if self.states[index] == 1:
                        color = (0, 0, 230)  # red
                    elif self.GATA6[index] >= self.NANOG[index] and self.GATA6[index] != 0:
                        color = (255, 255, 255)  # white
                    else:
                        color = (32, 252, 22)  # green

                # False yields coloring based on the finite dynamical system
                else:
                    if self.states[index] == 1:
                        color = (0, 0, 230)  # red
                    elif self.GATA6[index] > self.NANOG[index]:
                        color = (255, 255, 255)  # white
                    elif self.GATA6[index] == self.NANOG[index] == self.field - 1:
                        color = (30, 255, 255)  # yellow
                    elif self.GATA6[index] == self.NANOG[index] == 0:
                        color = (255, 50, 50)  # blue
                    else:
                        color = (32, 252, 22)  # green

                # draw the cell and a black outline to distinguish overlapping cells
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, color, -1)
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, (0, 0, 0), 1)

                # draw a black outline of the cell on the gradient image
                if fgf4_gradient:
                    grad_image = cv2.ellipse(grad_image, (x, y), (major, minor), 0, 0, 360, (255, 255, 255), 1)

            # if including gradient image, combine the to images side by side with gradient image on the right
            if fgf4_gradient:
                image = np.concatenate((image, grad_image), axis=1)

            # if the origin should be bottom-left flip it, otherwise it will be top-left
            if origin_bottom:
                image = cv2.flip(image, 0)

            # save the image as a PNG
            image_compression = 4  # image compression of png (0: no compression, ..., 9: max compression)
            file_name = f"{self.name}_image_{self.current_step}.png"
            cv2.imwrite(directory_path + file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, image_compression])

    @record_time
    def step_gradients(self):
        """ Saves each of the 2D gradients as a CSV at each step of the
            simulation.
        """
        # only continue if outputting gradient CSVs
        if self.output_gradients:
            # get path and make sure directory exists
            directory_path = check_direct(self.paths.gradients)

            # get the separator and save the following gradient outputs each to separate directories
            separator = self.paths.separator

            # go through all gradient arrays
            for gradient_name in self.gradient_names:
                # get directory to specific gradient
                grad_direct = check_direct(directory_path + separator + gradient_name + separator)

                # get file name, use f-string
                file_name = f"{self.name}_{gradient_name}_{self.current_step}.csv"

                # convert gradient from 3D to 2D array and save it as CSV
                gradient = self.__dict__[gradient_name][:, :, 0]
                np.savetxt(grad_direct + file_name, gradient, delimiter=",")

    @record_time
    def step_tda(self, in_pixels=False):
        """ Create CSV files for Topological Data Analysis (TDA) of different cell
            types. Each type will have its own subdirectory.

            in_pixels -> (bool) If the locations should be in pixels instead of meters
        """
        # only continue if outputting TDA files
        if self.output_tda:
            # get path and make sure directory exists
            directory_path = check_direct(self.paths.tda)

            # get the indices as an array of True/False of gata6 high cells and the non gata6 high cells
            red_indices = self.GATA6 > self.NANOG
            green_indices = np.invert(red_indices)

            # if TDA locations should be based on pixel location
            if in_pixels:
                scale = self.image_quality / self.size[0]
            else:
                scale = 1  # use meters

            # get the locations of the cells
            red_locations = self.locations[red_indices, 0:2] * scale
            green_locations = self.locations[green_indices, 0:2] * scale
            all_locations = self.locations[:, 0:2] * scale

            # get the separator and save the following TDA outputs each to separate directories
            separator = self.paths.separator

            # save all cell locations to a CSV
            all_path = check_direct(directory_path + separator + "all" + separator)
            file_name = f"{self.name}_tda_all_{self.current_step}.csv"
            np.savetxt(all_path + file_name, all_locations, delimiter=",")

            # save only GATA6 high cell locations to CSV
            red_path = check_direct(directory_path + separator + "red" + separator)
            file_name = f"{self.name}_tda_red_{self.current_step}.csv"
            np.savetxt(red_path + file_name, red_locations, delimiter=",")

            # save only non-GATA6 high, pluripotent cells to a CSV
            green_path = check_direct(directory_path + separator + "green" + separator)
            file_name = f"{self.name}_tda_green_{self.current_step}.csv"
            np.savetxt(green_path + file_name, green_locations, delimiter=",")
