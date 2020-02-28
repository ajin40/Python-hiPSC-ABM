#########################################################
# Name:    Model_Simulation                             #
# Author:  Jack Toppen                                  #
# Date:    2/15/20                                      #
#########################################################
import os
import shutil
import platform
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw
import time
import math
import cv2
import random as r
import matplotlib.path as mpltPath
import csv
from Model_StemCells import StemCell

setup_file = open(os.getcwd() + "/Setup.txt")
setup_list = setup_file.readlines()
parameters = []
for i in range(len(setup_list)):
    if i % 3 == 1:
        parameters.append(setup_list[i][2:-3])

if bool(parameters[3]):
    from Model_CUDA import *


def Mag(v1):
    """ Computes the magnitude of a vector
        Returns - a float representing the vector magnitude
    """
    temp = v1[0] ** 2 + v1[1] ** 2
    return math.sqrt(temp)


class Simulation(object):
    """ called once holds important information about the
        simulation
    """

    def __init__(self, name, path, end_time, time_step, pluri_div_thresh, diff_div_thresh, pluri_to_diff, size,
                 diff_surround_value, functions, parallel, max_fgf4, bounds, death_threshold, move_time_step,
                 move_max_time, spring_constant, friction, energy_kept, neighbor_distance):

        """ Initialization function for the simulation setup.
            name: the simulation name
            path: the path to save the simulation information to
            start_time: the start time for the simulation
            end_time: the end time for the simulation
            time_step: the time step to increment the simulation by
            pluri_div_thresh: threshold for pluripotent cells to divide
            diff_div_thresh:  threshold for differentiated cells to divide
            pluri_to_diff: threshold for pluripotent cells to differentiate
            size: the size of the grid (dimension, rows, columns)
            spring_max: represents the max distance for spring and other interactions
            diff_surround_value: the amount of differentiated cells needed to surround
                a pluripotent cell inducing its differentiation
            functions: the boolean functions as a string from Model_Setup
            itrs: the max amount of times optimize will run
            error: the max error allowed for optimize
            parallel: whether some aspects are run on the gpu
            max_fgf4: the limit a patch will hold for fgf4
            bounds: the bounds of the simulation
            spring_constant: strength of spring
        """
        self.name = name
        self.path = path
        self.end_time = end_time
        self.time_step = time_step
        self.pluri_div_thresh = pluri_div_thresh
        self.diff_div_thresh = diff_div_thresh
        self.pluri_to_diff = pluri_to_diff
        self.size = size
        self.diff_surround_value = diff_surround_value
        self.functions = functions
        self.parallel = parallel
        self.max_fgf4 = max_fgf4
        self.bounds = bounds
        self.death_threshold = death_threshold
        self.move_time_step = move_time_step
        self.move_max_time = move_max_time
        self.spring_constant = spring_constant
        self.friction = friction
        self.energy_kept = energy_kept
        self.neighbor_distance = neighbor_distance

        # the array that represents the grid and all its patches
        self.grid = np.zeros(self.size)

        # counts how many times an image is created for making videos
        self.image_counter = 0

        # keeps a running count of the time
        self.time_counter = 1.0

        # array to hold all of the stem cell objects
        self.objects = np.array([])

        # graph representing all the objects and their connections to other objects
        self.network = nx.Graph()

        # holds the objects until they are added or removed from the simulation
        self._objects_to_remove = np.array([])
        self._objects_to_add = np.array([])

        # keeps track of the current cell ID
        self._current_ID = 0

        # which file separator to use
        if platform.system() == "Windows":
            # windows
            self._sep = "\\"
        else:
            # linux/unix
            self._sep = "/"

        self.boundary = mpltPath.Path(self.bounds)

    #######################################################################################################################

    def run(self):
        """ Runs all elements of the simulation until
            the total time is met
        """
        # tries to make a new directory for the simulation
        try:
            os.mkdir(self.path + self._sep + self.name)
        except OSError:
            # directory already exists overwrite it
            print("Directory already exists... overwriting directory")

        # setup grid and patches

        self.initialize_grid()

        # run check_edge() to create connections between cells

        if self.parallel:
            check_edge_gpu(self)
        else:
            self.check_edges()

        self.handle_collisions()

        # save the first image and data of simulation
        self.save_file()

        shutil.copy("Setup.txt", self.path + self._sep + self.name + self._sep)

        # run simulation until end time
        while self.time_counter <= self.end_time:

            np.random.shuffle(self.objects)

            print("Time: " + str(self.time_counter))
            print("Number of objects: " + str(len(self.objects)))

            self.kill_cells()

            # updates all of the objects (motion, state, booleans)

            if self.parallel:
                update_grid_gpu(self)
            else:
                self.update_grid()

            self.update()

            # sees if cells can differentiate based on pluripotent cells surrounding by differentiated cells
            self.diff_surround()

            # adds/removes all objects from the simulation
            self.update_object_queue()

            # create/break connections between cells depending on distance apart
            if self.parallel:
                check_edge_gpu(self)
            else:
                self.check_edges()

            # moves cells in "motion" in a random fashion
            # self.random_movement()

            # optimizes the simulation by handling springs until error is less than threshold

            self.handle_collisions()

            # increments the time by time step
            self.time_counter += self.time_step

            # saves the image file and txt file with all important information
            self.save_file()

        # turns all images into a video at the end
        self.image_to_video()

    #######################################################################################################################

    def get_ID(self):
        """ Returns the current unique ID the simulation is on
        """
        return self._current_ID

    def inc_current_ID(self):
        """Increments the ID of cell by 1 each time called
        """
        self._current_ID += 1

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

    #######################################################################################################################

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
            self.objects[i].cell_death(self)
            if self.objects[i].death_timer >= self.death_threshold:
                self.add_object_to_removal_queue(self.objects[i])

    def update_object_queue(self):
        """ Updates the object add and remove queue
        """
        print("Adding " + str(len(self._objects_to_add)) + " objects...")
        print("Removing " + str(len(self._objects_to_remove)) + " objects...")

        # loops over all objects to remove
        for i in range(len(self._objects_to_remove)):
            self.remove_object(self._objects_to_remove[i])

        # loops over all objects to add
        for i in range(len(self._objects_to_add)):
            self.add_object(self._objects_to_add[i])

        # clear the arrays
        self._objects_to_remove = np.array([])
        self._objects_to_add = np.array([])


    def diff_surround(self):
        """ calls the object function that determines if
            a cell will differentiate based on the cells
            that surround it
        """
        # loops over all objects
        for i in range(len(self.objects)):
            # checks to see if they are Pluripotent and GATA6 low
            if self.objects[i].state == "Pluripotent" and self.objects[i].booleans[2] == 0:
                self.objects[i].diff_surround_funct(self)

    def update(self):
        """ Updates all of the objects in the simulation
            and degrades the FGF4 amount by 1 for all patches
        """

        # loops over all objects and updates each
        for i in range(len(self.objects)):
            self.objects[i].update_StemCell(self)


    def check_edges(self):
        """ checks all of the distances between cells
            if it is less than a set value create a
            connection between two cells.
        """
        self.network.clear()

        # loops over all objects
        for i in range(len(self.objects)):
            # loops over all objects not check already
            for j in range(i + 1, len(self.objects)):

                # max distance between cells to have a connection
                interaction_length = self.neighbor_distance

                # get the distance between cells
                dist_vec = self.objects[i].location - self.objects[j].location

                # get the magnitude of the distance vector
                dist = Mag(dist_vec)

                for i in range(len(self.objects)):
                    self.network.add_node(self.objects[i])

                if dist <= interaction_length:
                    self.network.add_edge(self.objects[i], self.objects[j])


    def handle_collisions(self):
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

            for i in range(len(self.objects)):
                self.objects[i].update_constraints(self)

            check_edge_gpu(self)

    #######################################################################################################################
    # image, video, and csv saving

    def draw_cell_image(self, network, path):
        """Turns the graph into an image at each timestep
        """
        # increases the image counter by 1 each time this is called
        self.image_counter += 1

        # get list of all objects/nodes in the simulation
        cells = list(network.nodes)

        # draws the background of the image
        image1 = Image.new("RGB", (1500, 1500), color=(53, 54, 65))
        draw = ImageDraw.Draw(image1)

        # bounds of the simulation used for drawing patch
        # inherit
        bounds = self.bounds

        # loops over all of the cells/nodes and draws a circle with corresponding color
        for i in range(len(cells)):
            node = cells[i]
            x, y = node.location
            r = node.nuclear_radius

            # if node.state == "Pluripotent" or node.state == "Differentiated":
            #     if node.booleans[3] == 0 and node.booleans[2] == 1:
            #         col = (255, 0, 0)
            #     elif node.booleans[3] == 1 and node.booleans[2] == 0:
            #         col = (17, 235, 24)
            #     elif node.booleans[3] == 1 and node.booleans[2] == 1:
            #         col = (245, 213, 7)
            #     else:
            #         col = (60, 0, 255)

            # if node.state == "Differentiated":
            #     col = (255, 255, 255)

            if node.state == "Pluripotent":
                col = 'white'

            else:
                col = 'black'

            out = "black"
            draw.ellipse((x - r + 250, y - r + 250, x + r + 250, y + r + 250), outline=out, fill=col)

        # loops over all of the bounds and draws lines to represent the grid
        for i in range(len(bounds)):
            x, y = bounds[i]
            if i < len(bounds) - 1:
                x1, y1 = bounds[i + 1]
            else:
                x1, y1 = bounds[0]
            r = 4
            draw.ellipse((x - r + 250, y - r + 250, x + r + 250, y + r + 250), outline='white', fill='white')
            draw.line((x + 250, y + 250, x1 + 250, y1 + 250), fill='white', width=10)

        # saves the image as a .png
        image1.save(path + ".png", 'PNG')

    def image_to_video(self):
        """ Creates a video out of all the png images at
            the end of the simulation
        """
        # gets base path
        base_path = self.path + self._sep + self.name + self._sep

        # image list to hold all image objects
        img_array = []

        # loops over all images created
        for i in range(self.image_counter + 1):
            img = cv2.imread(base_path + 'network_image' + str(i) + ".png")
            img_array.append(img)

        # output file for the video
        out = cv2.VideoWriter(base_path + 'network_video.avi', cv2.VideoWriter_fourcc("M", "J", "P", "G"), 1.0,
                              (1500, 1500))

        # adds image to output file
        for i in range(len(img_array)):
            out.write(img_array[i])

        # releases the file
        out.release()

    def location_to_text(self, path):
        """Outputs a txt file of the cell coordinates and the boolean values
        """
        # opens file
        new_file = open(path, "w")

        object_writer = csv.writer(new_file)
        object_writer.writerow(['ID', 'x_coord', 'y_coord', 'State', 'FGFR', 'ERK', 'GATA6', 'NANOG', 'Motion',
                                'diff_count', 'div_count'])

        for i in range(len(self.objects)):
            ID = str(self.objects[i].ID)
            x_coord = str(round(self.objects[i].location[0], 1))
            y_coord = str(round(self.objects[i].location[1], 1))
            x1 = str(self.objects[i].booleans[0])
            x2 = str(self.objects[i].booleans[1])
            x3 = str(self.objects[i].booleans[2])
            x4 = str(self.objects[i].booleans[3])
            diff = str(round(self.objects[i].diff_timer, 1))
            div = str(round(self.objects[i].division_timer, 1))
            state = str(self.objects[i].state)
            motion = str(self.objects[i].motion)

            object_writer.writerow([ID, x_coord, y_coord, state, x1, x2, x3, x4, motion, diff, div])

    def save_file(self):
        """ Saves the simulation txt files
            and image files
        """
        # get the base path
        base_path = self.path + self._sep + self.name + self._sep

        # saves the txt file with all the key information
        n2_path = base_path + "network_values" + str(int(self.time_counter)) + ".csv"
        self.location_to_text(n2_path)

        # draws the image of the simulation
        self.draw_cell_image(self.network, base_path + "network_image" + str(int(self.time_counter)) + ".0")


#######################################################################################################################
_name = str(parameters[0])
_path = str(parameters[1])
_parallel = bool(parameters[2])
_end_time = float(parameters[3])
_time_step = float(parameters[4])
_num_GATA6 = int(parameters[5])
_num_NANOG = int(parameters[6])
_stochastic = bool(parameters[7])
_size = eval(parameters[8])
_functions = eval(parameters[9])
_pluri_div_thresh = float(parameters[10])
_diff_div_thresh = float(parameters[11])
_pluri_to_diff = float(parameters[12])
_diff_surround_value = int(parameters[13])
_bounds = eval(parameters[14])
_max_fgf4 = int(parameters[15])
_death_threshold = int(parameters[16])
_move_time_step = float(parameters[17])
_move_max_time = float(parameters[18])
_spring_constant = float(parameters[19])
_friction = float(parameters[20])
_energy_kept = float(parameters[21])
_neighbor_distance = float(parameters[22])
_mass = float(parameters[23])
_nuclear_radius = float(parameters[24])
_cytoplasm_radius = float(parameters[25])

# initializes simulation class which holds all information about the simulation
simulation = Simulation(_name, _path, _end_time, _time_step, _pluri_div_thresh, _diff_div_thresh, _pluri_to_diff,
                        _size, _diff_surround_value, _functions, _parallel, _max_fgf4, _bounds, _death_threshold,
                        _move_time_step, _move_max_time, _spring_constant, _friction, _energy_kept, _neighbor_distance)

# loops over all NANOG_high cells and creates a stem cell object for each one with given parameters
for i in range(_num_NANOG):
    ID = i
    location = np.array([r.random() * _size[1], r.random() * _size[2]])
    state = "Pluripotent"
    motion = True
    mass = _mass
    if _stochastic:
        booleans = np.array([r.randint(0, 1), r.randint(0, 1), 0, 1])
    else:
        booleans = np.array([0, 0, 0, 1])

    nuclear_radius = _nuclear_radius
    cytoplasm_radius = _cytoplasm_radius

    diff_timer = _pluri_to_diff * r.random() * 0.5
    division_timer = _pluri_div_thresh * r.random()
    death_timer = _death_threshold * r.random() * 0.5

    sim_obj = StemCell(ID, location, motion, mass, nuclear_radius, cytoplasm_radius, booleans, state, diff_timer,
                       division_timer, death_timer)

    simulation.add_object(sim_obj)
    simulation.inc_current_ID()

# loops over all GATA6_high cells and creates a stem cell object for each one with given parameters
for i in range(_num_GATA6):
    ID = i + _num_NANOG
    location = np.array([r.random() * _size[1], r.random() * _size[2]])
    state = "Pluripotent"
    motion = True
    mass = _mass
    if _stochastic:
        booleans = np.array([r.randint(0, 1), r.randint(0, 1), 1, 0])
    else:
        booleans = np.array([0, 0, 1, 0])

    nuclear_radius = _nuclear_radius
    cytoplasm_radius = _cytoplasm_radius

    diff_timer = _pluri_to_diff * r.random()
    division_timer = _pluri_div_thresh * r.random()
    death_timer = _death_threshold * r.random()

    sim_obj = StemCell(ID, location, motion, mass, nuclear_radius, cytoplasm_radius, booleans, state, diff_timer,
                       division_timer, death_timer)

    simulation.add_object(sim_obj)
    simulation.inc_current_ID()

# runs the model
simulation.run()
