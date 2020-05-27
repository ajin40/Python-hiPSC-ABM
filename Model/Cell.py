import numpy as np
import math
import random as r
import copy


class Cell:
    """ Class for each cell in the simulation
    """

    def __init__(self, location, radius, motion, booleans, state, diff_counter, div_counter, death_counter,
                 boolean_counter):
        """ location: where the cell is located in the space "[x,y,z]"
            radius: the length of the cell radius
            motion: whether the cell is moving or not "True or False"
            velocity: the velocity as a vector of the cell
            booleans: array of boolean values for each variable in the boolean network
            state: whether the cell is pluripotent or differentiated
            diff_counter: holds the number of steps until the cell differentiates
            div_counter: holds the number of steps until the cell divides
            death_counter: holds the number of steps until the cell dies
            boolean_counter: holds the number of steps until the cell updates it's boolean values
        """
        self.location = location
        self.radius = radius
        self.motion = motion
        self.booleans = booleans
        self.state = state
        self.diff_counter = diff_counter
        self.div_counter = div_counter
        self.death_counter = death_counter
        self.boolean_counter = boolean_counter

        # holds any active forces applied to a cell resulting from motility and division
        self.active_force = np.array([0.0, 0.0, 0.0])

        # holds any inactive forces resulting from adhesion or repulsion
        self.inactive_force = np.array([0.0, 0.0, 0.0])

        # starts the cell off with a zero velocity vector
        self.velocity = np.array([0.0, 0.0, 0.0])

    def motility(self, simulation):
        """ applies forces to each cell based on chemotactic
            or random movement
        """
        # set motion to false if the cell is surrounded by many neighbors
        neighbors = list(simulation.neighbor_graph.neighbors(self))
        if len(neighbors) >= simulation.move_thresh:
            self.motion = False

        # check whether differentiated or pluripotent
        if self.state == "Differentiated":
            # get the neighbors of the cell
            neighbors = list(simulation.neighbor_graph.neighbors(self))

            # directed movement if the cell has neighbors
            if 0 < len(neighbors) < 6:
                # create a vector to hold the sum of normal vectors between a cell and its neighbors
                vector_holder = np.array([0.0, 0.0, 0.0])

                # loop over the neighbors getting the normal and adding to the holder
                for i in range(len(neighbors)):
                    vector = neighbors[i].location - self.location
                    vector_holder += vector

                # get the magnitude of the sum of normals
                magnitude = np.linalg.norm(vector_holder)

                # if for some case, its zero set the new normal vector to zero
                if magnitude == 0:
                    normal = np.array([0.0, 0.0, 0.0])
                else:
                    normal = vector_holder / magnitude

                # move in direction opposite to cells
                self.active_force += simulation.motility_force * normal * -1

            # if there aren't any neighbors and still in motion then move randomly
            elif self.motion:
                self.active_force += random_vector(simulation) * simulation.motility_force

        # for pluripotent cells
        else:
            # apply movement if the cell is "in motion"
            if self.motion:
                if self.booleans[2] == 1:
                    # continue if using Guye et al. movement and if there exists differentiated cells
                    if simulation.guye_move:
                        # get the differentiated neighbors
                        diff_neighbors = list(simulation.diff_graph.neighbors(self))

                        # check to see if there are any differentiated cells nearby
                        if len(diff_neighbors) > 0:
                            # get starting differentiated cell distance
                            vector = diff_neighbors[0].location - self.location
                            magnitude = np.linalg.norm(vector)

                            # loop over all other differentiated cells looking for the closest
                            for i in range(1, len(diff_neighbors)):
                                # get the distance to each of the other cells
                                next_vector = diff_neighbors[i].location - self.location
                                next_magnitude = np.linalg.norm(next_vector)

                                # check to see if the cell is closer than others
                                if next_magnitude < magnitude:
                                    # reset distance and vector for calculating the unit normal
                                    vector = next_vector
                                    magnitude = next_magnitude

                            # move in the direction of the closest differentiated neighbor
                            normal = vector / magnitude
                            self.active_force += normal * simulation.guye_force

                        else:
                            # if no differentiated cells, move randomly
                            self.active_force += random_vector(simulation) * simulation.motility_force
                    else:
                        # if not Guye et al. movement, move randomly
                        self.active_force += random_vector(simulation) * simulation.motility_force
                else:
                    # if not GATA6 high
                    self.active_force += random_vector(simulation) * simulation.motility_force

    def divide(self, simulation):
        """ produces another cell via mitosis
        """
        # halves the division counter and mass, while reducing the radius to a minimum
        self.div_counter = 0
        self.boolean_counter = 0
        self.radius = simulation.min_radius

        # create a deep copy of the object
        cell = copy.deepcopy(self)

        # apply a cell division force moving the cells away from each other
        force_vector = random_vector(simulation) * simulation.division_force
        self.active_force += force_vector
        cell.active_force -= force_vector

        # adds the cell to the simulation
        simulation.cells_to_add = np.append(simulation.cells_to_add, [cell])

    def boolean_function(self, fgf4_bool, simulation):
        """ updates the boolean variables of the cell
        """
        # gets the functions from the simulation
        function_list = simulation.functions

        # xn is equal to the value corresponding to its function
        x1 = fgf4_bool
        x2 = self.booleans[0]
        x3 = self.booleans[1]
        x4 = self.booleans[2]
        x5 = self.booleans[3]

        # evaluate the functions by turning them from strings to math equations
        new_fgf4 = eval(function_list[0]) % simulation.num_states
        new_fgfr = eval(function_list[1]) % simulation.num_states
        new_erk = eval(function_list[2]) % simulation.num_states
        new_gata6 = eval(function_list[3]) % simulation.num_states
        new_nanog = eval(function_list[4]) % simulation.num_states

        # updates self.booleans with the new boolean values and returns the new fgf4 value
        self.booleans = np.array([new_fgfr, new_erk, new_gata6, new_nanog])
        return new_fgf4

    def differentiate(self):
        """ differentiates the cell and updates the boolean values
            and sets the motion to be true
        """
        # change the state to differentiated
        self.state = "Differentiated"

        # set GATA6 high and NANOG low
        self.booleans[2] = 1
        self.booleans[3] = 0

        # allow the cell to move again
        self.motion = True

    def kill_cell(self, simulation):
        """ if the cell is without neighbors,
            increase the counter for death or kill it
        """
        # looks at the neighbors
        neighbors = list(simulation.neighbor_graph.neighbors(self))
        if len(neighbors) < simulation.lonely_cell:
            self.death_counter += 1
        else:
            self.death_counter = 0

        # removes cell if it meets the parameters
        if self.state == "Pluripotent":
            if self.death_counter >= simulation.death_thresh:
                simulation.cells_to_remove = np.append(simulation.cells_to_remove, [self])

    def diff_surround(self, simulation):
        """ calls the object function that determines if
            a cell will differentiate based on the cells
            that surround it
        """
        # checks to see if cell is Pluripotent and GATA6 low
        if self.state == "Pluripotent" and self.booleans[2] == 0:

            # finds neighbors of a cell
            neighbors = list(simulation.neighbor_graph.neighbors(self))

            # holds the current number differentiated neighbors
            num_diff_neighbors = 0

            # loops over the neighbors of a cell
            for i in range(len(neighbors)):
                # checks to see if current neighbor is differentiated if so add it to the counter
                if neighbors[i].state == "Differentiated":
                    num_diff_neighbors += 1

                # if the number of differentiated meets the threshold, increase the diff counter and break the loop
                if num_diff_neighbors >= simulation.diff_surround:
                    self.diff_counter += 1
                    break

    def update_cell(self, simulation):
        """ updates many of the instance variables
            of the cell
        """
        if self.radius < simulation.max_radius:
            if self.state == "Pluripotent":
                self.radius += simulation.pluri_growth
            else:
                self.radius += simulation.diff_growth

        # checks to see if the non-moving cell should divide
        if not self.motion:
            if self.state == "Differentiated" and self.div_counter >= simulation.diff_div_thresh:
                neighbors = list(simulation.neighbor_graph.neighbors(self))
                if len(neighbors) < simulation.contact_inhibit:
                    self.divide(simulation)

            elif self.state == "Pluripotent" and self.div_counter >= simulation.pluri_div_thresh:
                self.divide(simulation)
            else:
                self.div_counter += r.randint(0, 2)

        # activate the following pathway based on if dox has been induced yet
        if simulation.current_step >= simulation.dox_step:
            # coverts position in space into an integer for array location
            x_step = simulation.extracellular[0].dx
            y_step = simulation.extracellular[0].dy
            z_step = simulation.extracellular[0].dz

            half_index_x = self.location[0] // (x_step / 2)
            half_index_y = self.location[1] // (y_step / 2)
            half_index_z = self.location[2] // (z_step / 2)

            index_x = math.ceil(half_index_x / 2)
            index_y = math.ceil(half_index_y / 2)
            index_z = math.ceil(half_index_z / 2)

            # if a certain spot of the grid is less than the max FGF4 it can hold and the cell is NANOG high increase
            # the FGF4 by 1
            if simulation.extracellular[0].diffuse_values[index_x][index_y][index_z] < \
                    simulation.extracellular[0].maximum and self.booleans[3] == 1:
                simulation.extracellular[0].diffuse_values[index_x][index_y][index_z] += 1

            # if the FGF4 amount for the location is greater than 0, set the fgf4_bool value to be 1 for the functions
            if simulation.extracellular[0].diffuse_values[index_x][index_y][index_z] > 0:
                fgf4_bool = 1
            else:
                fgf4_bool = 0

            # temporarily hold the FGFR value
            tempFGFR = self.booleans[0]

            # run the boolean value through the functions
            if self.boolean_counter % simulation.boolean_thresh == 0:
                fgf4 = self.boolean_function(fgf4_bool, simulation)
            else:
                fgf4 = fgf4_bool

            self.boolean_counter += 1

            # if the temporary FGFR value is 0 and the FGF4 value is 1 decrease the amount of FGF4 by 1
            # this simulates FGFR using FGF4

            if tempFGFR == 0 and fgf4 == 1 and \
                    simulation.extracellular[0].diffuse_values[index_x][index_y][index_z] >= 1:
                simulation.extracellular[0].diffuse_values[index_x][index_y][index_z] -= 1

            # if the cell is GATA6 high and Pluripotent increase the differentiation counter by 1
            if self.booleans[2] == 1 and self.state == "Pluripotent":
                self.diff_counter += 1
                # if the differentiation counter is greater than the threshold, differentiate
                if self.diff_counter >= simulation.pluri_to_diff:
                    self.differentiate()


def random_vector(simulation):
    """ Computes a random point on a unit sphere centered at the origin
        Returns - point [x,y,z]
    """
    # gets random angle on the cell
    theta = r.random() * 2 * math.pi

    # gets x,y,z off theta and whether 2D or 3D
    if simulation.size[2] == 0:
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
