import numpy as np
import math
import random as r


class Cell:
    """ Class for each cell in the simulation
    """
    def __init__(self, location, motion, velocity, mass, booleans, state, diff_counter, div_counter, death_counter):
        """ location: where the cell is located in the space "[x,y,z]"
            motion: whether the cell is moving or not "True or False"
            velocity: the velocity as a vector of the cell
            mass: the mass of the entire cell
            booleans: array of boolean values for each variable in the boolean network
            state: whether the cell is pluripotent or differentiated
            diff_counter: holds the number of steps until the cell differentiates
            div_counter: holds the number of steps until the cell divides
            death_counter: holds the number of steps until the cell dies
        """
        self.location = location
        self.motion = motion
        self.velocity = velocity
        self.mass = mass
        self.booleans = booleans
        self.state = state
        self.diff_counter = diff_counter
        self.div_counter = div_counter
        self.death_counter = death_counter

        # Youngs modulus 4000 Pa
        self.youngs_mod = 600

        # Poisson's ratio
        self.poisson = 0.33333

        # starts the cell off with a zero force vector
        self.force = np.array([0.0, 0.0, 0.0])

        # radius of cell is currently determined based on mass and density held as a float
        self.radius = 0.0


    def divide(self, simulation):
        """ produces another cell via mitosis
        """
        # halves the division counter and mass
        self.div_counter *= 0.5
        self.mass *= 1.0

        # places the new cell on the surface on the old cell
        location = self.location + RandomPointOnSphere(simulation) * self.radius

        # makes sure the new cell's location is on the grid
        condition_x = 0 <= location[0] <= simulation.size[0]
        condition_y = 0 <= location[1] <= simulation.size[1]
        condition_z = 0 <= location[2] <= simulation.size[2]

        while not condition_x or not condition_y or not condition_z:
            location = self.location + RandomPointOnSphere(simulation) * self.radius

            # makes sure the new cell's location is on the grid
            condition_x = 0 <= location[0] <= simulation.size[0]
            condition_y = 0 <= location[1] <= simulation.size[1]
            condition_z = 0 <= location[2] <= simulation.size[2]

        # creates a new Cell object
        cell = Cell(location, self.motion, self.velocity, self.mass, self.booleans, self.state, self.diff_counter,
                    self.div_counter, self.death_counter)

        # adjusts the radii of both cells as division has caused them to change
        self.change_size(simulation)
        cell.change_size(simulation)

        # adds the cell to the simulation
        simulation.cells_to_add = np.append(simulation.cells_to_add, cell)

    def change_size(self, simulation):
        """ Increases the mass of the cell
            simulates growth
        """
        # increases mass
        self.mass *= 1.00

        # sets radius depending on if 2D or 3D based on area or volume
        if simulation.size[2] != 0:
            self.radius = ((3 * self.mass)/(4 * 3.14159) / simulation.density) ** (1/3)
        else:
            self.radius = (((1 * self.mass) / 3.14159) / simulation.density) ** 0.5

    def boolean_function(self, fgf4_bool, simulation):
        """ updates the boolean variables of the cell
        """
        function_list = simulation.functions

        # xn is equal to the value corresponding to its function
        x1 = fgf4_bool
        x2 = self.booleans[0]
        x3 = self.booleans[1]
        x4 = self.booleans[2]
        x5 = self.booleans[3]

        # evaluate the functions by turning them from strings to math equations
        new_1 = eval(function_list[0]) % simulation.num_states
        new_2 = eval(function_list[1]) % simulation.num_states
        new_3 = eval(function_list[2]) % simulation.num_states
        new_4 = eval(function_list[3]) % simulation.num_states
        new_5 = eval(function_list[4]) % simulation.num_states

        # updates self.booleans with the new boolean values
        self.booleans = np.array([new_2, new_3, new_4, new_5])

        return new_1

    def differentiate(self):
        """ differentiates the cell and updates the boolean values
            and sets the motion to be true
        """
        self.state = "Differentiated"
        self.booleans[2] = 1
        self.booleans[3] = 0
        self.motion = True

    def kill_cell(self, simulation):
        """ if the cell is without neighbors,
            increase the counter for death or kill it
        """
        # looks at the neighbors
        neighbors = list(simulation.neighbor_graph.neighbors(self))
        if len(neighbors) < 2:
            self.death_counter += 1
        else:
            self.death_counter = 0

        # removes cell if it meets the parameters
        if self.death_counter >= simulation.death_thresh:
            simulation.cells_to_remove = np.append(simulation.cells_to_remove, self)

    def diff_surround(self, simulation):
        """ calls the object function that determines if
            a cell will differentiate based on the cells
            that surround it
        """
        # checks to see if cell is Pluripotent and GATA6 low
        if self.state == "Pluripotent" and self.booleans[2] == 0:

            # finds neighbors of a cell
            neighbors = list(simulation.neighbor_graph.neighbors(self))

            # counts neighbors
            num_neighbors = len(neighbors)

            # if there are enough cells surrounding the cell the differentiation timer will increase
            if num_neighbors >= simulation.diff_surround:
                self.diff_counter += 1

    def update_cell(self, simulation):
        """ updates many of the instance variables
            of the cell
        """
        # if other cells are differentiated around a cell it will stop moving
        if self.state == "Differentiated":
            neighbors = list(simulation.neighbor_graph.neighbors(self))
            for i in range(len(neighbors)):
                if neighbors[i].state == "Differentiated":
                    self.motion = False
                    break

        # if other cells are also pluripotent, gata6 low, and nanog high they will stop moving
        if self.booleans[3] == 1 and self.booleans[2] == 0 and self.state == "Pluripotent":
            neighbors = list(simulation.neighbor_graph.neighbors(self))
            for j in range(len(neighbors)):
                if neighbors[j].booleans[3] == 1 and neighbors[j].booleans[2] == 0 and neighbors[j].state == "Pluripotent":
                    self.motion = False
                    break

        # checks to see if the non-moving cell should divide
        if not self.motion:
            if self.state == "Differentiated" and self.div_counter >= simulation.diff_div_thresh:
                self.divide(simulation)
            elif self.state == "Pluripotent" and self.div_counter >= simulation.pluri_div_thresh:
                self.divide(simulation)
            else:
                self.div_counter += 1

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

        # if a certain spot of the grid is less than the max FGF4 it can hold and the cell is NANOG high increase the
        # FGF4 by 1
        if simulation.extracellular[0].diffuse_values[index_x][index_y][index_z] < simulation.extracellular[0].maximum \
                and self.booleans[3] == 1:
            simulation.extracellular[0].diffuse_values[index_x][index_y][index_z] += 1

        # if the FGF4 amount for the location is greater than 0, set the fgf4_bool value to be 1 for the functions
        if simulation.extracellular[0].diffuse_values[index_x][index_y][index_z] > 0:
            fgf4_bool = 1
        else:
            fgf4_bool = 0

        # temporarily hold the FGFR value
        tempFGFR = self.booleans[0]

        # run the boolean value through the functions
        fgf4 = self.boolean_function(fgf4_bool, simulation)

        # if the temporary FGFR value is 0 and the FGF4 value is 1 decrease the amount of FGF4 by 1
        # this simulates FGFR using FGF4

        if tempFGFR == 0 and fgf4 == 1 and simulation.extracellular[0].diffuse_values[index_x][index_y][index_z] >= 1:
            simulation.extracellular[0].diffuse_values[index_x][index_y][index_z] -= 1

        # if the cell is GATA6 high and Pluripotent increase the differentiation counter by 1
        if self.booleans[2] == 1 and self.state == "Pluripotent":
            self.diff_counter += 1
            # if the differentiation counter is greater than the threshold, differentiate
            if self.diff_counter >= simulation.pluri_to_diff:
                self.differentiate()


def RandomPointOnSphere(simulation):
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