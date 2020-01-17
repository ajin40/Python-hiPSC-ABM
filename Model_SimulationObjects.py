from Model_Setup import *
from Model_SimulationMath import *
import random as rand
import math as math
import numpy as np
import matplotlib.path as mpltPath
import Model_Simulation as sim


class SimulationObject(object):
    """ Base class from which all simulation objects must inherit
    """

    def __init__(self, location, radius, ID ,owner_ID, sim_type):
        """ Base class which defines properties all sim objects MUST have
            location - the location of the sphere
            radius - the radius of the sphere
            ID - the ID of the object #WARNING# This ID is used ot hash objects
                  it MUST BE UNIQUE
            owner_ID - usually the same as the ID, also must be unique between
                       agents, unless all agents are part of a larger group
                       i.e. this is the mechanism for multi-agent agents
            sim_type - the type of object the simulation object is
        """
        self.division_timer = 0
        self.location = location
        self.radius = radius
        self.sim_type = sim_type
        self.ID = ID
        self.owner_ID = owner_ID
        self.bounds= [[0,0], [0,1000], [1000,1000], [1000,0]]
        if len(self.bounds)>0:
            self.boundary = mpltPath.Path(self.bounds)
        else:
            self.boundary = []


        #keep track of the opt and col vecs
        if len(self.location)==3:
            self._disp_vec = [0,0,0]
            self._fixed_constraint_vec = [0,0,0]
            self._v = [0,0,0]
        else:
            self._disp_vec = [0,0]
            self._fixed_constraint_vec = [0,0]
            self._v = [0,0]

        #keep track of production consumptions values
        self.gradient_source_sink_coeff = dict()
        #keep track of the relative indices in the gradient array
        self.gradient_position = dict()
        #keep track of the value of the gradient associated with these agents
        self.gradient_value = dict()


    def update(self, sim, dt):
        """ Updates the simulation object
        """
        pass

    def setC(self,C,ind):
        self.C[ind]=C

    def get_max_interaction_length(self):
        """ Get the max interaction length of the object
        """
        return self.radius*2.0 #in um

    def get_interaction_length(self):
        return self.radius #in um

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

    def add_fixed_constraint_vec(self, vec):
        """ Adds a vector to the optimization vector
        """
        self._fixed_constraint_vec = AddVec(self._fixed_constraint_vec, vec)




    def update_constraints(self, dt):
        """ Updates all of the constraints on the object
        """
        #first update the position by the col and opt vectors
        #make sure neither of these vectors is greater than error
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
        #then update the the pos using the fixed vectors
        mag = Mag(self._fixed_constraint_vec)
        if mag > 5:
            n = NormVec(self._fixed_constraint_vec)
            self._fixed_contraint_vec = ScaleVec(n, 5.0)

        location = AddVec(self.location, self._fixed_constraint_vec)
        if len(self.bounds)>0:
            if self.boundary.contains_point(location[0:2]):
                self.location=location
            else:
                new_loc = SubtractVec(self.location, self._fixed_constraint_vec)
                if self.boundary.contains_point(new_loc[0:2]):
                    self.location=new_loc

        else:
            self.location=location

        self._fixed_constraint_vec = [0,0]


    def __repr__(self):
        """ Returns a string representation of the object
        """
        return self.sim_type+": "+repr(self.ID)+" "+repr(self.location)

    def __eq__(self, other):
        """ Handles the equal operator for the object
        """
        if isinstance(other, SimulationObject):
            return self.ID == other.ID
        #otherwise
        return False

    def __hash__(self):
        """ Handles the hashing operator for the object
        """
        return hash(self.ID)




class StemCell(SimulationObject):
    """ A stem cell class
    """
    def __init__(self, location, radius, ID,src_snk, x1,x2,x3,x4,x5, state, diff_timer, division_timer, division_time, owner_ID = None):
        """ Constructor for a stem cell
            location - the location fo the stem cell
            radius - the size of the stem cell
            ID - the unique ID for the agent
            state - the state of the stem cell
            division_set - the initial division set for the cell
            division_time - the time it takes the cell to divide
            owner_ID - the ID associated with the owner of this agent
        """

        #define some variables
        if owner_ID == None:
            owner_ID = ID

        self.src_snk=src_snk
        self.diff_timer = diff_timer
        self.division_timer = division_timer
        self.division_time = division_time
        self.compress=0
        self.cmpr_direct=[0,0,0]
        self.funct_1 = x1
        self.funct_2 = x2
        self.funct_3 = x3
        self.funct_4 = x4
        self.funct_5 = x5
        self.state = state
        self.bounds = [[0,0], [0,1000], [1000,1000], [1000,0]]

        #call the parent constructor
        super(StemCell, self).__init__(location,radius,ID,owner_ID,"stemcell")
        


        
    def __hash__(self):
        return hash(self.ID)

    def get_interaction_length(self):
        """ Gets the interaction length for the cell. Overrides parent
            Returns - the length of any interactions with this cell (float)
        """
        return self.radius+1.0




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



        return [new_1, new_2, new_3, new_4, new_5]


    def compress_force(self, sim):
        nbs=list(sim.network.neighbors(self))
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
        self.cmpr_direct=cmpr_dir
        self.compress=compress


    def divide(self, sim):

        # 2D monolayer culture growth
        n = 2
        # get the radius
        radius = self.radius

        if len(self.bounds) > 0:
            count = 0
            while count == 0:
                location = RandomPointOnSphere(n) * radius / 2.0 + self.location
                if self.boundary.contains_point(location[0:2]):
                    count = 1
        else:
            location = RandomPointOnSphere(n) * radius / 2.0 + self.location


        # get the ID
        ID = sim.get_ID()
        sc = StemCell(location, radius, ID, self.src_snk, self.funct_1, self.funct_2, self.funct_3, self.funct_4, self.funct_5, self.state, self.diff_timer, self.division_timer,self.division_time)
        sim.add_object_to_addition_queue(sc)
        # reset the division time
        self.division_timer *= 0.5

    def differentiate(self):
        self.state = "Differentiated"
        self.x4 = 1
        self.x5 = 0




    def update(self, sim, dt):
        """ Updates the stem cell to decide whether they differentiate
            or divide
        """

        if self.division_timer >= self.division_time and self.state == "Differentiated":
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
            if self.diff_timer >= sim.diff_threshold:
                self.differentiate()



        


