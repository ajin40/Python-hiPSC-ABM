import numpy as np
import math
import random as r
import copy


class Cell:
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
        self.active_force = np.array([0.0, 0.0, 0.0], dtype=float)

        # holds any inactive forces resulting from adhesion or repulsion
        self.inactive_force = np.array([0.0, 0.0, 0.0], dtype=float)

        # create an empty array used for holding the neighbors
        self.neighbors = np.array([], np.object)

        # a pointer to the closest differentiated cell
        self.closest_diff = None

    def motility(self, simulation):
        """ applies forces to each cell based on chemotactic
            or random movement
        """
        # set motion to false if the cell is surrounded by many neighbors
        neighbors = self.neighbors
        if len(neighbors) >= simulation.move_thresh:
            self.motion = False

        # check whether differentiated or pluripotent
        if self.state == "Differentiated":
            # get the neighbors of the cell
            neighbors = self.neighbors

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
                    if simulation.guye_move and self.closest_diff is not None:
                        # get the differentiated neighbors
                        guye_neighbor = self.closest_diff

                        vector = guye_neighbor.location - self.location
                        magnitude = np.linalg.norm(vector)

                        # move in the direction of the closest differentiated neighbor
                        normal = vector / magnitude
                        self.active_force += normal * simulation.guye_force

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
        self.neighbors = np.array([], np.object)
        self.closest_diff = None

        # create a deep copy of the object
        cell = copy.deepcopy(self)

        # move the cells to a position that is representative of the new locations of daughter cells
        position = random_vector(simulation) * (simulation.max_radius - simulation.min_radius)
        self.location += position
        cell.location -= position

        # adds the cell to the simulation
        simulation.cells_to_add = np.append(simulation.cells_to_add, [cell])


    def kill_cell(self, simulation):
        """ if the cell is without neighbors,
            increase the counter for death or kill it
        """
        # looks at the neighbors
        neighbors = self.neighbors
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
            neighbors = self.neighbors

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
