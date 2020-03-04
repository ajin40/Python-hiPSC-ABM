#########################################################
# Name:    Functions                                    #
# Author:  Jack Toppen                                  #
# Date:    3/4/20                                       #
#########################################################
import numpy as np
import Parallel
import math
import Classes
import random as r
import os
import shutil

def handle_collisions(self):
    time_counter = 0
    while time_counter <= self.move_max_time:
        time_counter += self.move_time_step

        edges = list(self.network.edges())

        for i in range(len(edges)):
            obj1 = edges[i][0]
            obj2 = edges[i][1]

            displacement = obj1.location - obj2.location

            if np.linalg.norm(
                    displacement) < obj1.nuclear_radius + obj1.cytoplasm_radius + obj2.nuclear_radius + obj2.cytoplasm_radius:
                displacement_normal = displacement / np.linalg.norm(displacement)
                obj1_displacement = (obj1.nuclear_radius + obj1.cytoplasm_radius) * displacement_normal
                obj2_displacement = (obj2.nuclear_radius + obj1.cytoplasm_radius) * displacement_normal

                real_displacement = displacement - (obj1_displacement + obj2_displacement)

                obj1.velocity -= real_displacement * (self.energy_kept * self.spring_constant / obj1.mass)

                obj2.velocity += real_displacement * (self.energy_kept * self.spring_constant / obj2.mass)

        for i in range(len(self.objects)):
            velocity = self.objects[i].velocity

            movement = velocity * self.move_max_time
            self.objects[i]._disp_vec += movement

            if np.linalg.norm(velocity) == 0.0:
                velocity_normal = 0.0
            else:
                velocity_normal = velocity / np.linalg.norm(velocity)

            velocity_mag = np.linalg.norm(velocity)
            movement_mag = np.linalg.norm(movement)

            new_velocity = velocity_normal * max(velocity_mag ** 2 - 2 * self.friction * movement_mag, 0.0) ** 0.5

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
        if self.grid[np.array([0]), np.array([array_location_x]), np.array([array_location_y])] < self.max_fgf4 and \
                self.objects[i].booleans[3] == 1:
            self.grid[np.array([0]), np.array([array_location_x]), np.array([array_location_y])] += 1

        # if the FGF4 amount for the location is greater than 0, set the fgf4_bool value to be 1 for the functions
        if self.grid[np.array([0]), np.array([array_location_x]), np.array([array_location_y])] > 0:
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
                self.grid[np.array([0]), np.array([array_location_x]), np.array([array_location_y])] >= 1:
            self.grid[np.array([0]), np.array([array_location_x]), np.array([array_location_y])] -= 1

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
            if self.boundary.contains_point(location[0:2]):
                count = 1
    else:
        location = RandomPointOnSphere() * radius * 2.0 + thing.location

    # halve the division timer
    thing.division_timer *= 0.5

    # ID the cell
    ID = self._current_ID

    # create new cell and add it to the simulation

    sc = Classes.StemCell(ID, location, thing.motion, thing.mass, thing.nuclear_radius, thing.cytoplasm_radius,
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

        location = self.objects[i].location + self.objects[i]._disp_vec

        # if there are bounds, this will check to see if new location is in the grid
        if len(self.bounds) > 0:
            if self.boundary.contains_point(location[0:2]):
                self.objects[i].location = location

            # if the new location is not in the grid, try opposite
            else:
                new_loc = self.objects[i].location - self.objects[i]._disp_vec
                if self.boundary.contains_point(new_loc[0:2]):
                    self.objects[i].location = new_loc
        else:
            self.objects[i].location = location

        # resets the movement vector to [0,0]
        self.objects[i]._disp_vec = np.array([0.0, 0.0])

def initialize_grid(self):
    """ sets up the grid and the patches
        with a random amount of FGF4
    """
    # loops over all rows
    for i in range(self.size[1]):
        # loops over all columns
        for j in range(self.size[2]):
            self.grid[np.array([0]), np.array([i]), np.array([j])] = r.randint(0, self.max_fgf4)

def update_grid(self):

    if self.parallel:
        Parallel.update_grid_gpu(self)
    else:
        for i in range(self.size[1]):
            for j in range(self.size[2]):
                if self.grid[np.array([0]), np.array([i]), np.array([j])] >= 1:
                    self.grid[np.array([0]), np.array([i]), np.array([j])] += -1



def random_movement(self):
    """ has the objects that are in motion
        move in a random way
    """
    # loops over all objects
    for i in range(len(self.objects)):
        # finds the objects in motion
        if self.objects[i].motion:
            # new location of 10 times a random float from -1 to 1
            temp_x = self.objects[i].location[0] + r.uniform(-1, 1) * 10
            temp_y = self.objects[i].location[1] + r.uniform(-1, 1) * 10
            # if the new location would be outside the grid don't move it
            if 1000 >= temp_x >= 0 and 1000 >= temp_y >= 0:
                self.objects[i].location[0] = temp_x
                self.objects[i].location[1] = temp_y

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

    shutil.copy("Initials.txt", self.path + self._sep + self.name + self._sep)
    try:
        os.mkdir(self.path + self._sep + self.name)
    except OSError:
        # directory already exists overwrite it
        print("Directory already exists... overwriting directory")

def info(self):
    print("Time: " + str(self.time_counter))
    print("Number of objects: " + str(len(self.objects)))
    # increments the time by time step
    self.time_counter += self.time_step