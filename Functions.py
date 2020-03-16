#########################################################
# Name:    Functions                                    #
# Author:  Jack Toppen                                  #
# Date:    3/4/20                                       #
#########################################################
import numpy as np
import Parallel
import math
import StemCell
import random as r
import os

def handle_collisions(self):

    if self.parallel:
        Parallel.handle_collisions_gpu(self)
    else:
        time_counter = 0
        while time_counter <= self.move_max_time:
            time_counter += self.move_time_step

            edges = list(self.network.edges())

            for i in range(len(edges)):
                obj1 = edges[i][0]
                obj2 = edges[i][1]

                displacement = obj1.location - obj2.location

                if np.linalg.norm(displacement) < obj1.nuclear_radius + obj1.cytoplasm_radius + obj2.nuclear_radius + obj2.cytoplasm_radius:
                    displacement_normal = displacement / np.linalg.norm(displacement)

                    obj1_displacement = (obj1.nuclear_radius + obj1.cytoplasm_radius) * displacement_normal
                    obj2_displacement = (obj2.nuclear_radius + obj1.cytoplasm_radius) * displacement_normal

                    real_displacement = (displacement - (obj1_displacement + obj2_displacement)) / 2

                    obj1.velocity[0] -= real_displacement[0] * (self.energy_kept * self.spring_constant / obj1.mass) ** 0.5
                    obj1.velocity[1] -= real_displacement[1] * (self.energy_kept * self.spring_constant / obj1.mass) ** 0.5

                    obj2.velocity[0] += real_displacement[0] * (self.energy_kept * self.spring_constant / obj2.mass) ** 0.5
                    obj2.velocity[1] += real_displacement[1] * (self.energy_kept * self.spring_constant / obj2.mass) ** 0.5


            for i in range(len(self.objects)):
                velocity = self.objects[i].velocity

                movement = velocity * self.move_time_step
                self.objects[i]._disp_vec += movement

                new_velocity = np.array([0.0, 0.0])

                new_velocity[0] = np.sign(velocity[0]) * max((velocity[0]) ** 2 - 2 * self.friction * abs(movement[0]), 0.0) ** 0.5
                new_velocity[1] = np.sign(velocity[0]) * max((velocity[1]) ** 2 - 2 * self.friction * abs(movement[1]), 0.0) ** 0.5

                self.objects[i].velocity = new_velocity

            update_constraints(self)

            Parallel.check_edge_gpu(self)


def update(self):
    for i in range(len(self.objects)):

        # if other cells are differentiated around a cell it will stop moving
        if self.objects[i].state == "Differentiated":
            nbs = list(self.network.neighbors(self.objects[i]))
            for j in range(len(nbs)):
                if nbs[j].state == "Differentiated":
                    self.objects[i].motion = False
                    break

        # if other cells are pluripotent, gata6 low, and nanog high they will stop moving
        if self.objects[i].booleans[3] == 1 and self.objects[i].booleans[2] == 0 and self.objects[i].state == "Pluripotent":
            nbs = list(self.network.neighbors(self.objects[i]))
            for j in range(len(nbs)):
                if nbs[j].booleans[3] == 1 and nbs[j].booleans[2] == 0 and nbs[j].state == "Pluripotent":
                    self.objects[i].motion = False
                    break

        if not self.objects[i].motion:
            if self.objects[i].state == "Differentiated" and self.objects[i].division_timer >= self.diff_div_thresh:
                divide(self, self.objects[i])

            if self.objects[i].state == "Pluripotent" and self.objects[i].division_timer >= self.pluri_div_thresh:
                divide(self, self.objects[i])

            else:
                self.objects[i].division_timer += 1

        # coverts position on grid into an integer for array location
        array_location_x = int(math.floor(self.objects[i].location[0]))
        array_location_y = int(math.floor(self.objects[i].location[1]))

        # if a certain spot of the grid is less than the max FGF4 it can hold and the cell is NANOG high increase the
        # FGF4 by 1
        if self.grid[[0], [array_location_x], [array_location_y]] < self.max_fgf4 and self.objects[i].booleans[3] == 1:
            self.grid[[0], [array_location_x], [array_location_y]] += 1

        # if the FGF4 amount for the location is greater than 0, set the fgf4_bool value to be 1 for the functions
        if self.grid[[0], [array_location_x], [array_location_y]] > 0:
            fgf4_bool = 1

        else:
            fgf4_bool = 0

        # temporarily hold the FGFR value
        tempFGFR = self.objects[i].booleans[0]

        # run the boolean value through the functions
        fgf4 = boolean_function(self, self.objects[i], fgf4_bool)

        # if the temporary FGFR value is 0 and the FGF4 value is 1 decrease the amount of FGF4 by 1
        # this simulates FGFR using FGF4

        if tempFGFR == 0 and fgf4 == 1 and \
                self.grid[[0], [array_location_x], [array_location_y]] >= 1:
            self.grid[[0], [array_location_x], [array_location_y]] -= 1

        # if the cell is GATA6 high and Pluripotent increase the differentiation counter by 1
        if self.objects[i].booleans[2] == 1 and self.objects[i].state == "Pluripotent":
            self.objects[i].diff_timer += 1
            # if the differentiation counter is greater than the threshold, differentiate
            if self.objects[i].diff_timer >= self.pluri_to_diff:
                differentiate(self.objects[i])


def diff_surround(self):
    """ calls the object function that determines if
        a cell will differentiate based on the cells
        that surround it
    """
    # loops over all objects
    for i in range(len(self.objects)):
        # checks to see if they are Pluripotent and GATA6 low
        if self.objects[i].state == "Pluripotent" and self.objects[i].booleans[2] == 0:

            # finds neighbors of a cell
            neighbors = list(self.network.neighbors(self.objects[i]))
            # counts neighbors that are differentiated and in the interaction distance
            counter = 0
            for j in range(len(neighbors)):
                if neighbors[j].state == "Differentiated":
                    dist_vec = neighbors[j].location - self.objects[i].location
                    dist = np.linalg.norm(dist_vec)
                    if dist <= self.neighbor_distance:
                        counter += 1

            # if there are enough cells surrounding the cell the differentiation timer will increase
            if counter >= self.diff_surround_value:
                self.objects[i].diff_timer += 1

def update_object_queue(self):
    """ Updates the object add and remove queue
    """
    print("Adding " + str(len(self._objects_to_add)) + " objects...")
    print("Removing " + str(len(self._objects_to_remove)) + " objects...")

    # loops over all objects to remove
    for i in range(len(self._objects_to_remove)):
        remove_object(self, self._objects_to_remove[i])

    # loops over all objects to add
    for i in range(len(self._objects_to_add)):
        add_object(self, self._objects_to_add[i])

    # clear the arrays
    self._objects_to_remove = np.array([])
    self._objects_to_add = np.array([])

def boolean_function(self, thing, fgf4_bool):
    function_list = self.functions

    # xn is equal to the value corresponding to its function
    x1 = fgf4_bool
    x2 = thing.booleans[0]
    x3 = thing.booleans[1]
    x4 = thing.booleans[2]
    x5 = thing.booleans[3]

    # evaluate the functions by turning them from strings to math equations
    new_1 = eval(function_list[0]) % 2
    new_2 = eval(function_list[1]) % 2
    new_3 = eval(function_list[2]) % 2
    new_4 = eval(function_list[3]) % 2
    new_5 = eval(function_list[4]) % 2

    # updates self.booleans with the new boolean values
    thing.booleans = np.array([new_2, new_3, new_4, new_5])

    return new_1


def divide(self, thing):
    # radius of cell
    radius = thing.nuclear_radius + thing.cytoplasm_radius
    location = thing.location

    # if there are boundaries
    if len(self.bounds) > 0:
        count = 0
        # tries to put the cell on the grid
        while count == 0:
            location = RandomPointOnSphere() * radius * 2.0 + thing.location
            if self.boundary.contains_point(location):
                count = 1
    else:
        location = RandomPointOnSphere() * radius * 2.0 + thing.location

    # halve the division timer
    thing.division_timer *= 0.5

    # ID the cell
    ID = self._current_ID

    # create new cell and add it to the simulation

    sc = StemCell.StemCell(ID, location, thing.motion, thing.mass, thing.nuclear_radius, thing.cytoplasm_radius,
                           thing.booleans, thing.state, thing.diff_timer, thing.division_timer, thing.death_timer)

    add_object_to_addition_queue(self, sc)


def check_edges(self):
    """ checks all of the distances between cells
        if it is less than a set value create a
        connection between two cells.
    """

    self.network.clear()

    if self.parallel:
        Parallel.check_edge_gpu(self)
    else:
        # loops over all objects
        for i in range(len(self.objects)):
            # loops over all objects not check already
            for j in range(i + 1, len(self.objects)):

                # max distance between cells to have a connection
                interaction_length = self.neighbor_distance

                # get the distance between cells
                dist_vec = self.objects[i].location - self.objects[j].location

                # get the magnitude of the distance vector
                dist = np.linalg.norm(dist_vec)

                for i in range(len(self.objects)):
                    self.network.add_node(self.objects[i])

                if dist <= interaction_length:
                    self.network.add_edge(self.objects[i], self.objects[j])



def add_object(self, sim_object):
    """ Adds the specified object to the array
        and the graph
    """
    # adds it to the array
    self.objects = np.append(self.objects, sim_object)

    # adds it to the graph
    self.network.add_node(sim_object)

def remove_object(self, sim_object):
    """ Removes the specified object from the array
        and the graph
    """
    # removes it from the array
    self.objects = self.objects[self.objects != sim_object]

    # removes it from the graph
    self.network.remove_node(sim_object)

def add_object_to_addition_queue(self, sim_object):
    """ Will add an object to the simulation object queue
        which will be added to the simulation at the end of
        the update phase.
    """
    # adds object to array
    self._objects_to_add = np.append(self._objects_to_add, sim_object)

    # increments the current ID
    self._current_ID += 1

def add_object_to_removal_queue(self, sim_object):
    """ Will add an object to the simulation object queue
        which will be removed from the simulation at the end of
        the update phase.
    """
    # adds object to array
    self._objects_to_remove = np.append(self._objects_to_remove, sim_object)


def update_constraints(self):
    for i in range(len(self.objects)):

        self.objects[i].location += self.objects[i]._disp_vec

        if not 0 <= self.objects[i].location[0] <= 1000:
            self.objects[i].location[0] -= 2 * self.objects[i]._disp_vec[0]

        if not 0 <= self.objects[i].location[1] <= 1000:
            self.objects[i].location[1] -= 2 * self.objects[i]._disp_vec[1]

        # resets the movement vector to [0,0]
        self.objects[i]._disp_vec = np.array([0.0, 0.0])


def random_movement(self):
    """ has the objects that are in motion
        move in a random way
    """
    # loops over all objects
    for i in range(len(self.objects)):
        # finds the objects in motion
        if self.objects[i].motion:
            # new location of 10 times a random float from -1 to 1
            self.objects[i]._disp_vec[0] =+ r.uniform(-1, 1) * 10
            self.objects[i]._disp_vec[1] =+ r.uniform(-1, 1) * 10

    update_constraints(self)


def kill_cells(self):
    for i in range(len(self.objects)):
        neighbors = list(self.network.neighbors(self.objects[i]))
        if len(neighbors) < 1:
            self.objects[i].death_timer += 1

        if self.objects[i].death_timer >= self.death_threshold:
            add_object_to_removal_queue(self, self.objects[i])

def differentiate(thing):
    """ differentiates the cell and updates the boolean values
        and sets the motion to be true
    """
    thing.state = "Differentiated"
    thing.booleans[2] = 1
    thing.booleans[3] = 0
    thing.motion = True

def RandomPointOnSphere():
    """ Computes a random point on a sphere
        Returns - a point on a unit sphere [x,y] at the origin
    """
    theta = r.random() * 2 * math.pi
    x = math.cos(theta)
    y = math.sin(theta)

    return np.array((x, y))

def inc_current_ID(self):
    """Increments the ID of cell by 1 each time called
    """
    self._current_ID += 1

def check_name(self):
    """Renames the file if need be
    """
    while True:
        try:
            os.mkdir(self.path + self.sep + self.name)
            break
        except OSError:
            print("Directory already exists")
            user = input("Would you like to overwrite the existing simulation? (y/n): ")
            if user == "n":
                self.name = input("New name: ")
            if user == "y":
                try:
                    os.mkdir(self.path + self.sep + self.name)
                except OSError:
                    print("Overwriting directory")
                    break


def info(self):
    print("Time: " + str(self.time_counter))
    print("Number of objects: " + str(len(self.objects)))
