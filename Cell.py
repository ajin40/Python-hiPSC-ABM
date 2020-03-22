#########################################################
# Name:    Cell                                         #
# Author:  Jack Toppen                                  #
# Date:    3/17/20                                      #
#########################################################
import numpy as np
import math
import random as r


class Cell:
    """ Every cell object in the simulation
        will have this class
    """
    def __init__(self, location, motion, velocity, mass, radius, booleans, state, diff_timer,
                 division_timer, death_timer):

        """ location: where the cell is located on the grid "[x,y]"
            radius: the radius of each cell
            ID: the number assigned to each cell "0-number of cells"
            booleans: array of boolean values for each boolean function
            state: whether the cell is pluripotent or differentiated
            diff_timer: counts the total steps per cell needed to differentiate
            division_timer: counts the total steps per cell needed to divide
            motion: whether the cell is in moving or not "True or False"
            spring_constant: strength of force applied based on distance
        """
        self.location = location
        self.motion = motion
        self.velocity = velocity
        self.mass = mass
        self.radius = radius
        self.booleans = booleans
        self.state = state
        self.diff_timer = diff_timer
        self.division_timer = division_timer
        self.death_timer = death_timer


    def divide(self, simulation):
        """ produces another cell via mitosis
        """

        # halve the division timer
        self.division_timer *= 0.5
        self.mass *= 0.5

        location = self.location + RandomPointOnSphere() * self.radius

        while not 0 <= location[0] < simulation.size[0] or not 0 <= location[1] < simulation.size[1] or not 0 <= location[2] < simulation.size[2]:
            location = self.location + RandomPointOnSphere() * 3.0

        cell = Cell(location, self.motion, self.velocity, self.mass, self.radius, self.booleans,
                    self.state, self.diff_timer, self.division_timer, self.death_timer)

        simulation.add_object_to_addition_queue(cell)

    def change_size(self, simulation):
        self.mass *= 1.0

        if simulation.three_D:
            self.radius = ((3/(4 * 3.14159)) / simulation.density) ** (1/3)
        else:
            self.radius = (3 / (4 * 3.14159) / simulation.density) ** 0.5

    def boolean_function(self, fgf4_bool, simulation):
        """ updates the boolean values of the cell
        """
        function_list = simulation.functions

        # xn is equal to the value corresponding to its function
        x1 = fgf4_bool
        x2 = self.booleans[0]
        x3 = self.booleans[1]
        x4 = self.booleans[2]
        x5 = self.booleans[3]

        # evaluate the functions by turning them from strings to math equations
        new_1 = eval(function_list[0]) % simulation.n
        new_2 = eval(function_list[1]) % simulation.n
        new_3 = eval(function_list[2]) % simulation.n
        new_4 = eval(function_list[3]) % simulation.n
        new_5 = eval(function_list[4]) % simulation.n

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
        neighbors = list(simulation.network.neighbors(self))
        if len(neighbors) < 1:
            self.death_timer += 1

        # increases timer
        if self.death_timer >= simulation.death_threshold:
            simulation.add_object_to_removal_queue(self)

    def diff_surround(self, simulation):
        """ calls the object function that determines if
            a cell will differentiate based on the cells
            that surround it
        """

        # checks to see if they are Pluripotent and GATA6 low
        if self.state == "Pluripotent" and self.booleans[2] == 0:

            # finds neighbors of a cell
            neighbors = list(simulation.network.neighbors(self))
            # counts neighbors that are differentiated and in the interaction distance
            counter = 0
            for j in range(len(neighbors)):
                if neighbors[j].state == "Differentiated":
                    dist_vec = neighbors[j].location - self.location
                    dist = np.linalg.norm(dist_vec)
                    if dist <= simulation.neighbor_distance:
                        counter += 1

            # if there are enough cells surrounding the cell the differentiation timer will increase
            if counter >= simulation.diff_surround_value:
                self.diff_timer += 1


    def update_cell(self, simulation):

        # if other cells are differentiated around a cell it will stop moving
        if self.state == "Differentiated":
            nbs = list(simulation.network.neighbors(self))
            for j in range(len(nbs)):
                if nbs[j].state == "Differentiated":
                    self.motion = False
                    break

        # if other cells are pluripotent, gata6 low, and nanog high they will stop moving
        if self.booleans[3] == 1 and self.booleans[2] == 0 and self.state == "Pluripotent":
            nbs = list(simulation.network.neighbors(self))
            for j in range(len(nbs)):
                if nbs[j].booleans[3] == 1 and nbs[j].booleans[2] == 0 and nbs[j].state == "Pluripotent":
                    self.motion = False
                    break

        if not self.motion:
            if self.state == "Differentiated" and self.division_timer >= simulation.diff_div_thresh:
                self.divide(simulation)

            if self.state == "Pluripotent" and self.division_timer >= simulation.pluri_div_thresh:
                self.divide(simulation)
            else:
                self.division_timer += 1

        # coverts position on grid into an integer for array location
        array_location_x = int(math.floor(self.location[0]))
        array_location_y = int(math.floor(self.location[1]))
        array_location_z = int(math.floor(self.location[2]))


        # if a certain spot of the grid is less than the max FGF4 it can hold and the cell is NANOG high increase the
        # FGF4 by 1
        if simulation.gradients[0].grid[array_location_x][array_location_y][array_location_z] < simulation.gradients[0].max_concentration\
                and self.booleans[3] == 1:
            simulation.gradients[0].grid[array_location_x][array_location_y][array_location_z] += 1

        # if the FGF4 amount for the location is greater than 0, set the fgf4_bool value to be 1 for the functions
        if simulation.gradients[0].grid[array_location_x][array_location_y][array_location_z] > 0:
            fgf4_bool = 1

        else:
            fgf4_bool = 0

        # temporarily hold the FGFR value
        tempFGFR = self.booleans[0]

        # run the boolean value through the functions
        fgf4 = self.boolean_function(fgf4_bool, simulation)

        # if the temporary FGFR value is 0 and the FGF4 value is 1 decrease the amount of FGF4 by 1
        # this simulates FGFR using FGF4

        if tempFGFR == 0 and fgf4 == 1 and simulation.gradients[0].grid[array_location_x][array_location_y][array_location_z] >= 1:
            simulation.gradients[0].grid[array_location_x][array_location_y][array_location_z] -= 1

        # if the cell is GATA6 high and Pluripotent increase the differentiation counter by 1
        if self.booleans[2] == 1 and self.state == "Pluripotent":
            self.diff_timer += 1
            # if the differentiation counter is greater than the threshold, differentiate
            if self.diff_timer >= simulation.pluri_to_diff:
                self.differentiate()

def RandomPointOnSphere():
    """ Computes a random point on a sphere
        Returns - a point on a unit sphere [x,y,z] at the origin
    """
    theta = r.random() * 2 * math.pi
    x = math.cos(theta)
    y = math.sin(theta)
    # z = r.random()

    return np.array([x, y, 0.0])