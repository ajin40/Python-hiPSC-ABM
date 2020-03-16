#########################################################
# Name:    Functions                                    #
# Author:  Jack Toppen                                  #
# Date:    3/4/20                                       #
#########################################################
import numpy as np
import Parallel
import math
import Cell
import random as r

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
            self.add_object_to_removal_queue(self.objects[i])



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





