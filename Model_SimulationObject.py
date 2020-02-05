from Model_Setup import *
import random as rand
import math as math
import numpy as np
import matplotlib.path as mpltPath
import Model_Simulation as sim

class StemCell(object):
    """ Every cell object in the simulation
        will have this class
    """
    def __init__(self, location, radius, ID, booleans, state, diff_timer, division_timer, motion):
        """ location: where the cell is located on the grid "[x,y]"
            radius: the radius of each cell
            ID: the number assigned to each cell "0-number of cells"
            booleans: array of boolean values for each boolean function
            state: whether the cell is pluripotent or differentiated
            diff_timer: counts the total steps per cell needed to differentiate
            division_timer: counts the total steps per cell needed to divide
            motion: whether the cell is in moving or not "True or False"
        """
        self.location = location
        self.radius = radius
        self.ID = ID
        self.booleans = booleans
        self.state = state
        self.diff_timer = diff_timer
        self.division_timer = division_timer
        self.motion = motion

        # holds the compression force by a cell's neighbors
        self.compress = 0

        # holds the value of the movement vector of the cell
        self._disp_vec = [0, 0]

        # the bounds of the simulation
        self.bounds = [[0, 0], [0, 1000], [1000, 1000], [1000, 0]]

        # defines the bounds of the simulation using mathplotlib
        if len(self.bounds) > 0:
            self.boundary = mpltPath.Path(self.bounds)
        else:
            # if no bounds are defined, the boundaries are empty
            self.boundary = []

#######################################################################################################################

    def get_spring_constant(self, other):
        """ Returns the spring constant for all spring interactions
            left with parameter for future use with differing spring constants
        """
        return 0.77


    def add_displacement_vec(self, vec):
        """ Adds a vector to the vector representing the movement of the cell
        """
        self._disp_vec = AddVec(self._disp_vec, vec)


    def update_constraints(self):
        """ Updates the boundary constraints of the grid on the objects
            and limits the movement to a magnitude of 5
        """

        # gets magnitude of movement vector
        mag = Mag(self._disp_vec)
        if mag > 5:
            # if the magnitude is greater than 5, it will be scaled down to 5
            n = NormVec(self._disp_vec)
            self._disp_vec = ScaleVec(n, 5.0)

        # new location is the sum of previous location and movement vector
        location = AddVec(self.location, self._disp_vec)

        # if there are bounds, this will check to see if new location is in the grid
        if len(self.bounds) > 0:
            if self.boundary.contains_point(location[0:2]):
                self.location = location

            # if the new location is not in the grid, return to the old point
            else:
                new_loc = SubtractVec(self.location, self._disp_vec)
                if self.boundary.contains_point(new_loc[0:2]):
                    self.location = new_loc
        else:
            self.location = location

        # resets the movement vector to [0,0]
        self._disp_vec = [0, 0]


    def boolean_function(self, sim, fgf4_bool):
        """ preforms the functions defined in model setup as strings
        """
        # call functions from model setup
        function_list = sim.call_functions()

        # xn is equal to the value corresponding to its function
        x1 = self.booleans[0]
        x2 = self.booleans[1]
        x3 = self.booleans[2]
        x4 = self.booleans[3]
        x5 = self.booleans[4]

        # evaluate the functions by turning them from strings to math equations
        new_1 = (eval(function_list[0]) * fgf4_bool) % 2
        new_2 = eval(function_list[1]) % 2
        new_3 = eval(function_list[2]) % 2
        new_4 = eval(function_list[3]) % 2
        new_5 = eval(function_list[4]) % 2

        # updates self.booleans with the new boolean values
        self.booleans = np.array([new_1, new_2, new_3, new_4, new_5])


    def diff_surround_funct(self, sim):
        """ if there are enough differentiated cells surrounding
            a pluripotent cell then it will divide
        """
        # finds neighbors of a cell
        neighbors = np.array(list(sim.network.neighbors(self)))
        # counts neighbors that are differentiated and in the interaction distance
        counter = 0
        for i in range(len(neighbors)):
            if neighbors[i].state == "Differentiated":
                dist_vec = SubtractVec(neighbors[i].location, self.location)
                dist = Mag(dist_vec)
                if dist <= sim.spring_max:
                    counter += 1

        # if there are enough cells surrounding the cell the differentiation timer will increase
        if counter >= sim.diff_surround_value:
            self.diff_timer += 1


    def compress_force(self, sim):
        """ finds the compression force of other cells acting on the cell
            if too great the cell won't divide
        """
        # finds neighbors of a cell
        neighbors = np.array(list(sim.network.neighbors(self)))
        # holds the compression force values
        compress = 0
        # radius of the cell
        rd1 = self.radius
        # loops over all neighbors
        for i in range(len(neighbors)):
            rd2 = neighbors[i].radius
            dist_vec = SubtractVec(neighbors[i].location, self.location)
            dist = Mag(dist_vec)
            cmpr = rd1 + rd2 - dist
            # only counts cells that are touching or overlapping
            compress += max(cmpr, 0)
        compress = float(compress / (1.0 + len(neighbors)))
        self.compress = compress


    def divide(self, sim):
        """ creates new cells if a cell divides shares
            most of the same values and places the cell
            in a random place outside
        """
        # radius of cell
        radius = self.radius

        # if there are boundaries
        if len(self.bounds) > 0:
            count = 0
            # tries to put the cell on the grid
            while count == 0:
                location = RandomPointOnSphere() * radius * 2.0 + self.location
                if self.boundary.contains_point(location[0:2]):
                    count = 1
        else:
            location = RandomPointOnSphere() * radius * 2.0 + self.location

        # halve the division timer
        self.division_timer *= 0.5

        # ID the cell
        ID = sim.get_ID()

        # create new cell and add it to the simulation
        sc = StemCell(location, radius, ID, self.booleans, self.state, self.diff_timer, self.division_timer, self.motion)
        sim.add_object_to_addition_queue(sc)


    def differentiate(self):
        """ differentiates the cell and updates the boolean values
            and sets the motion to be true
        """
        self.state = "Differentiated"
        self.booleans[3] = 1
        self.booleans[4] = 0
        self.motion = True

    def update(self, sim):
        """ Updates the stem cell to decide whether they differentiate
            or divide, changes state, and sets motion
        """
        # if other cells are differentiated around a cell it will stop moving
        if self.state == "Differentiated":
            nbs = np.array(list(sim.network.neighbors(self)))
            differentiated_nbs = 0
            for i in range(len(nbs)):
                if nbs[i].state == "Differentiated":
                    differentiated_nbs += 1

            if differentiated_nbs >= 1:
                self.motion = False

        # if other cells are pluripotent, gata6 low, and nanog high they will stop moving
        if self.booleans[4] == 1 and self.booleans[3] == 0 and self.state == "Pluripotent":
            nbs = np.array(list(sim.network.neighbors(self)))
            pluripotent_nbs = 0
            for i in range(len(nbs)):
                if nbs[i].state == 1 and nbs[i].state == 0 and nbs[i].state == "Pluripotent":
                    pluripotent_nbs += 1

            if pluripotent_nbs >= 1:
                self.motion = False

        # if a non-moving, differentiated cell has a high division threshold and low compression force, it will divide
        if self.division_timer >= sim.diff_div_thresh and self.state == "Differentiated" and self.compress < 2.0 \
                and not self.motion:
            self.divide(sim)

        # otherwise increase the division timer
        else:
            self.division_timer += 1

        # if a non-moving, pluripotent cell has a high division threshold and low compression force, it will divide
        if self.division_timer >= sim.pluri_div_thresh and self.state == "Pluripotent" and self.compress < 2.0 \
                and not self.motion:
            self.divide(sim)

        # otherwise increase the division timer
        else:
            self.division_timer += 1

        # coverts position on grid into an integer for array location
        array_location_x = int(self.location[0])-1
        array_location_y = int(self.location[1])-1

        # if a certain spot of the grid is less than the max FGF4 it can hold and the cell is NANOG high increase the
        # FGF4 by 1
        if sim.grid[np.array([0]), np.array([array_location_x]), np.array([array_location_y])] < 5 and \
                self.booleans[4] == 1:
            sim.grid[np.array([0]), np.array([array_location_x]), np.array([array_location_y])] += 1

        # if the FGF4 amount for the location is greater than 0, set the fgf4_bool value to be 1 for the functions
        if sim.grid[np.array([0]), np.array([array_location_x]), np.array([array_location_y])] > 0:
            fgf4_bool = 1

        else:
            fgf4_bool = 0

        # temporarily hold the FGFR value
        tempFGFR = self.booleans[1]

        # run the boolean value through the functions
        self.boolean_function(sim, fgf4_bool)

        # if the temporary FGFR value is 0 and the FGF4 value is 1 decrease the amount of FGF4 by 1
        if tempFGFR == 0 and self.booleans[1] == 1:
            sim.grid[np.array([0]), np.array([array_location_x]), np.array([array_location_y])] += -1

        # if the cell is GATA6 high and Pluripotent increase the differentiation counter by 1
        if self.booleans[3] == 1 and self.state == "Pluripotent":
            self.diff_timer += 1
            # if the differentiation counter is greater than the threshold, differentiate
            if self.diff_timer >= sim.pluri_to_diff:
                self.differentiate()


#######################################################################################################################
# commonly used math functions

def RandomPointOnSphere():
    """ Computes a random point on a sphere
        Returns - a point on a unit sphere [x,y] at the origin
    """
    theta = rand.random() * 2 * math.pi
    x = math.cos(theta)
    y = math.sin(theta)

    return np.array((x, y))


def AddVec(v1, v2):
    """ Adds two vectors that are in the form [x,y,z]
        Returns - a new vector [x,y,z] as a numpy array
    """
    n = len(v1)
    temp = np.array(v1)
    for i in range(0, n):
        temp[i] += float(v2[i])
    return temp


def SubtractVec(v1, v2):
    """ Subtracts vector [x,y,z] v2 from vector v1
        Returns - a new vector [x,y,z] as a numpy array
    """
    n = len(v1)
    temp = np.array(v1)
    for i in range(0, n):
        temp[i] -= float(v2[i])
    return temp


def ScaleVec(v1, s):
    """ Scales a vector f*[x,y,z] = [fx, fy, fz]
        Returns - a new scaled vector [x,y,z] as a numpy array
    """
    n = len(v1)
    temp = np.array(v1)
    for i in range(0, n):
        temp[i] = temp[i] * s
    return temp


def Mag(v1):
    """ Computes the magnitude of a vector
        Returns - a float representing the vector magnitude
    """
    n = len(v1)
    temp = 0.
    for i in range(0, n):
        temp += (v1[i] * v1[i])
    return math.sqrt(temp)


def NormVec(v1):
    """ Computes a normalized version of the vector v1
        Returns - a normalizerd vector [x,y,z] as a numpy array
    """
    mag = Mag(v1)
    temp = np.array(v1)
    if mag == 0:
        return temp * 0
    return temp / mag