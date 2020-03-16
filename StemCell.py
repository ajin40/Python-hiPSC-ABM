#########################################################
# Name:    Classes                                      #
# Author:  Jack Toppen                                  #
# Date:    3/4/20                                       #
#########################################################
import numpy as np




class StemCell:
    """ Every cell object in the simulation
        will have this class
    """
    def __init__(self, ID, location, motion, mass, nuclear_radius, cytoplasm_radius, booleans, state, diff_timer,
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
        self.ID = ID
        self.location = location
        self.motion = motion
        self.mass = mass
        self.nuclear_radius = nuclear_radius
        self.cytoplasm_radius = cytoplasm_radius
        self.booleans = booleans
        self.state = state
        self.diff_timer = diff_timer
        self.division_timer = division_timer
        self.death_timer = death_timer

        # holds the value of the movement vector of the cell
        self._disp_vec = np.array([0.0, 0.0])

        self.velocity = np.array([0.0, 0.0])





