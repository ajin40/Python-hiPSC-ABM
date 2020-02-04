from Model_Setup import *
import random as rand
import math as math
import numpy as np
import matplotlib.path as mpltPath
import Model_Simulation as sim


class StemCell(object):
    """ A stem cell class
    """
    def __init__(self, location, radius, ID, booleans, state, diff_timer, division_timer, motion):
        """ Constructor for a stem cell
            location - the location fo the stem cell
            radius - the size of the stem cell
            ID - the unique ID for the agent
            state - the state of the stem cell
            division_set - the initial division set for the cell
            division_time - the time it takes the cell to divide
            owner_ID - the ID associated with the owner of this agent
        """

        self.diff_timer = diff_timer
        self.division_timer = division_timer
        self.compress = 0
        self.funct_1 = booleans[0]
        self.funct_2 = booleans[1]
        self.funct_3 = booleans[2]
        self.funct_4 = booleans[3]
        self.funct_5 = booleans[4]
        self.state = state
        self.bounds = [[0,0], [0,1000], [1000,1000], [1000,0]]
        self.location = location
        self.radius = radius
        self.ID = ID
        self.motion = motion

        if len(self.bounds) > 0:
            self.boundary = mpltPath.Path(self.bounds)
        else:
            self.boundary = []

        self._disp_vec = [0, 0]


    def get_spring_constant(self, other):
        """ Gets the spring constant of the object
            Returns: 1.0 by default
            NOTE: Meant to be overwritten by a base class if more
                  functionality is required
        """
        return 0.77


    def add_displacement_vec(self, vec):
        """ Adds a vector to the optimization vector
        """
        self._disp_vec = AddVec(self._disp_vec, vec)


    def update_constraints(self, dt):
        """ Updates all of the constraints on the object
        """

        mag = Mag(self._disp_vec)
        if mag > 5:
            n = NormVec(self._disp_vec)
            self._disp_vec = ScaleVec(n, 5.0)
        location = AddVec(self.location, self._disp_vec)
        if len(self.bounds)>0:
            if self.boundary.contains_point(location[0:2]):
                self.location=location
            else:
                new_loc = SubtractVec(self.location, self._disp_vec)
                if self.boundary.contains_point(new_loc[0:2]):
                    self.location=new_loc
        else:
            self.location=location


        self._disp_vec = [0,0]


    def boolean_function(self,sim):

        funct_list = sim.call_functions()
        x1 = self.funct_1
        x2 = self.funct_2
        x3 = self.funct_3
        x4 = self.funct_4
        x5 = self.funct_5

        new_1 = eval(funct_list[0]) % 2
        new_2 = eval(funct_list[1]) % 2
        new_3 = eval(funct_list[2]) % 2
        new_4 = eval(funct_list[3]) % 2
        new_5 = eval(funct_list[4]) % 2


        return np.array([new_1, new_2, new_3, new_4, new_5])

    def diff_surround_funct(self, sim):
        nbs = np.array(list(sim.network.neighbors(self)))
        rd1 = self.radius
        counter = 0
        for i in range(len(nbs)):
            counter = 0
            dist_vec = SubtractVec(nbs[i].location, self.location)

            dist = Mag(dist_vec)
            if dist <= sim.spring_max:
                counter += 1

        if counter >= sim.diff_surround_value:
            self.diff_timer += 1


    def compress_force(self, sim):
        nbs= np.array(list(sim.network.neighbors(self)))
        compress=0
        rd1=self.radius
        cmpr_dir=0
        
        for i in range(len(nbs)):
            rd2=nbs[i].radius
            dist_vec=SubtractVec(nbs[i].location,self.location)
            dist=Mag(dist_vec)
            norm=NormVec(dist_vec)
            cmpr=rd1+rd2-dist
            cmpr_dir+=norm
            compress+=max(cmpr,0)
        compress=float(compress/(1.0+len(nbs)))
        self.compress=compress


    def divide(self, sim):

        # get the radius
        radius = self.radius

        if len(self.bounds) > 0:
            count = 0
            while count == 0:
                location = RandomPointOnSphere() * radius*2.0 + self.location
                if self.boundary.contains_point(location[0:2]):
                    count = 1
        else:
            location = RandomPointOnSphere() * radius*2.0 + self.location

        self.division_timer *= 0.5
        # get the ID
        ID = sim.get_ID()
        sc = StemCell(location, radius, ID, np.array([self.funct_1, self.funct_2, self.funct_3, self.funct_4, self.funct_5]),
                      self.state, self.diff_timer, self.division_timer, self.motion)
        sim.add_object_to_addition_queue(sc)


    def differentiate(self):
        self.state = "Differentiated"
        self.x4 = 1
        self.x5 = 0
        self.motion = True

    def update(self, sim, dt):
        """ Updates the stem cell to decide whether they differentiate
            or divide
        """

        if self.state == "Differentiated":
            nbs = np.array(list(sim.network.neighbors(self)))
            differentiated_nbs = 0
            for i in range(len(nbs)):
                if nbs[i].state == "Differentiated":
                    differentiated_nbs += 1

            if differentiated_nbs >= 1:
                self.motion = False

        if self.funct_5 == 1 and self.funct_4 == 0 and self.state == "Pluripotent":
            nbs = np.array(list(sim.network.neighbors(self)))
            pluripotent_nbs = 0
            for i in range(len(nbs)):
                if nbs[i].state == 1 and nbs[i].state == 0 and nbs[i].state == "Pluripotent":
                    pluripotent_nbs += 1

            if pluripotent_nbs >= 1:
                self.motion = False

        if self.division_timer >= sim.diff_div_thresh and self.state == "Differentiated":
            if self.compress < 2.0 and not self.motion:
                self.divide(sim)
        else:
            self.division_timer += 1

        if self.division_timer >= sim.pluri_div_thresh and self.state == "Pluripotent":
            if self.compress < 2.0:
                self.divide(sim)
        else:
            self.division_timer += 1

        array_location_x = int(self.location[0])
        array_location_y = int(self.location[1])

        if sim.grid[np.array([0]), np.array([array_location_x]), np.array([array_location_y])] < 5 and self.funct_5 == 1:
            sim.grid[np.array([0]), np.array([array_location_x]), np.array([array_location_y])] += 1

        tempFGFR = self.funct_2

        bVals = self.boolean_function(sim)
        self.x1 = bVals[0]
        self.x2 = bVals[1]
        self.x3 = bVals[2]
        self.x4 = bVals[3]
        self.x5 = bVals[4]

        if tempFGFR == 0 and self.funct_2 == 1:
            sim.grid[np.array([0]), np.array([array_location_x]), np.array([array_location_y])] += -1

        if self.funct_4 == 1 and self.state == "Pluripotent":
            self.diff_timer += 1
            if self.diff_timer >= sim.pluri_to_diff:
                self.differentiate()


#######################################################################################################################

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