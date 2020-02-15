#########################################################
# Name:    Model_Simulation                             #
# Author:  Jack Toppen                                  #
# Date:    2/5/20                                       #
#########################################################
import os
import platform
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw
import time
import cv2
import random as r
import matplotlib.path as mpltPath
from Model_Math import *
import csv

# from numba import cuda


class Simulation(object):
    """ called once holds important information about the
        simulation
    """
    def __init__(self, name, path, start_time, end_time, time_step, pluri_div_thresh, diff_div_thresh, pluri_to_diff,
                 size, spring_max, diff_surround_value, functions, itrs, error, parallel, max_fgf4, bounds, spring_constant):
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
        """

        # make sure name and path are strings
        if type(name) is str:
            self.name = name
        else:
            self.name = str(name)
        if type(path) is str:
            self.path = path
        else:
            self.path = str(path)

        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.time_step = float(time_step)
        self.pluri_div_thresh = pluri_div_thresh
        self.diff_div_thresh = diff_div_thresh
        self.pluri_to_diff = pluri_to_diff
        self.size = size
        self.spring_max = spring_max
        self.diff_surround_value = diff_surround_value
        self.functions = functions
        self.itrs = itrs
        self.error = error
        self.parallel = parallel
        self.max_fgf4 = max_fgf4
        self.bounds = bounds
        self.spring_constant = spring_constant

        # the array that represents the grid and all its patches
        self.grid = np.zeros(self.size)

        # counts how many times an image is created for making videos
        self.image_counter = 0

        # keeps a running count of the time
        self.time_counter = float(start_time)

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

        # defines the bounds of the simulation using mathplotlib
        if len(self.bounds) > 0:
            self.boundary = mpltPath.Path(self.bounds)
        else:
            # if no bounds are defined, the boundaries are empty
            self.boundary = []

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

        if self.parallel:
            self.initialize_grid_gpu()
        else:
            self.initialize_grid()

        # run collide() to create connections between cells
        if self.parallel:
            self.check_edge_gpu()
        else:
            self.check_edge_run()

        self.optimize(self.error, self.itrs)

        # save the first image and data of simulation
        self.save_file()

        # run simulation until end time
        while self.time_counter <= self.end_time:

            print("Time: " + str(self.time_counter))
            print("Number of objects: " + str(len(self.objects)))

            # updates all of the objects (motion, state, booleans)

            if self.parallel:
                self.update_grid_gpu()
            else:
                self.update_grid()

            self.update()

            # sees if cells can differentiate based on pluripotent cells surrounding by differentiated cells
            self.diff_surround()

            # adds/removes all objects from the simulation
            self.update_object_queue()

            # create/break connections between cells depending on distance apart
            if self.parallel:
                self.check_edge_gpu()
            else:
                self.check_edge_run()

            # moves cells in "motion" in a random fashion
            self.random_movement()

            # calculates how much compression force is on each cell
            self.calculate_compression()

            # optimizes the simulation by handling springs until error is less than threshold
            self.optimize(self.error, self.itrs)

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
        self.objects = np.delete(self.objects, sim_object)

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
                self.grid[np.array([0]), np.array([i]), np.array([j])] = r.randint(0, 10)


    def initialize_grid_gpu(self):
        from numba import cuda
        an_array = self.grid
        an_array_gpu = cuda.to_device(an_array)
        threads_per_block = (32, 32)
        blocks_per_grid_x = math.ceil(an_array.shape[0] / threads_per_block[0])
        blocks_per_grid_y = math.ceil(an_array.shape[1] / threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        initialize_grid_cuda[blocks_per_grid, threads_per_block](an_array_gpu)

        self.grid = an_array_gpu.copy_to_host()


    def update_grid(self):
        for i in range(self.size[1]):
            for j in range(self.size[2]):
                if self.grid[np.array([0]), np.array([i]), np.array([j])] >= 1:
                    self.grid[np.array([0]), np.array([i]), np.array([j])] += -1


    def update_grid_gpu(self):
        from numba import cuda
        an_array = self.grid
        an_array_gpu = cuda.to_device(an_array)
        threads_per_block = (32, 32)
        blocks_per_grid_x = math.ceil(an_array.shape[0] / threads_per_block[0])
        blocks_per_grid_y = math.ceil(an_array.shape[1] / threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        update_grid_cuda[blocks_per_grid, threads_per_block](an_array_gpu)

        self.grid = an_array_gpu.copy_to_host()


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

        
    def calculate_compression(self):
        """ calculates the compress force for all objects"""

        objects = self.objects

        # loops over all objects
        for i in range(len(objects)):
            objects[i].compress_force(self)


    def diff_surround(self):
        """ calls the object function that determines if
            a cell will differentiate based on the cells
            that surround it
        """
        objects = self.objects

        # loops over all objects
        for i in range(len(objects)):
            # checks to see if they are Pluripotent and GATA6 low
            if objects[i].state == "Pluripotent" and objects[i].booleans[2] == 0:
                objects[i].diff_surround_funct(self)

            
    def update(self):
        """ Updates all of the objects in the simulation
            and degrades the FGF4 amount by 1 for all patches
        """

        # loops over all objects and updates each
        for i in range(len(self.objects)):
            self.objects[i].update_StemCell(self)
                

    def check_edge(self):
        """ checks all of the distances between cells
            if it is less than a set value create a
            connection between two cells. (Only run at
            beginning)
        """
        # loops over all objects
        for i in range(len(self.objects)):
            # loops over all objects not check already
            for j in range(i+1, len(self.objects)):

                # max distance between cells to have a connection
                interaction_length = self.spring_max * 2

                # get the distance between cells
                dist_vec = SubtractVec(self.objects[i].location, self.objects[j].location)

                # get the magnitude of the distance vector
                dist = Mag(dist_vec)

                if dist <= interaction_length:
                    # if correct length, add a edge in the graph representing a connection
                    self.network.add_edge(self.objects[i], self.objects[j])


    def check_edge_run(self):
        """ checks all of the distances between cells
            if it is less than a set value create a
            connection between two cells.
        """
        # loops over all objects
        for i in range(len(self.objects)):
            # loops over all objects not check already
            for j in range(i + 1, len(self.objects)):

                # max distance between cells to have a connection
                interaction_length = self.spring_max * 2

                # get the distance between cells
                dist_vec = SubtractVec(self.objects[i].location, self.objects[j].location)

                # get the magnitude of the distance vector
                dist = Mag(dist_vec)

                # if the distance is greater than the interaction length try to remove it
                if dist > interaction_length:
                    try:
                        self.network.remove_edge(self.objects[i], self.objects[j])
                    except:
                        pass

                # if the distance is less than or equal to interaction length add connection unless both
                # cells aren't in motion
                if dist <= interaction_length and (self.objects[i].motion or self.objects[j].motion):
                    self.network.add_edge(self.objects[i], self.objects[j])


    def check_edge_gpu(self):
        from numba import cuda
        rows = len(self.objects)
        columns = len(self.objects)
        edges_array = np.zeros((rows, columns))

        location_array = np.empty((0, 2), int)

        for i in range(len(self.objects)):
            location_array = np.append(location_array, np.array([self.objects[i].location]), axis=0)

        location_array_device_in = cuda.to_device(location_array)
        edges_array_device_in = cuda.to_device(edges_array)

        threads_per_block = (32, 32)
        blocks_per_grid_x = math.ceil(edges_array.shape[0] / threads_per_block[0])
        blocks_per_grid_y = math.ceil(edges_array.shape[1] / threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        check_edge_cuda[blocks_per_grid, threads_per_block](location_array_device_in, edges_array_device_in)

        output = edges_array_device_in.copy_to_host()

        edges = np.argwhere(output == 1)

        for i in range(len(edges)):
            self.network.add_edge(self.objects[edges[i][0]], self.objects[edges[i][1]])


    def optimize(self, error_max, max_itrs):
        """ tries to correct for the error by applying
            spring forces and keeping cells in the grid
        """
        # amount of times optimize has run
        itrs = 0
        # baseline for starting while loop
        error = error_max * 2

        while itrs < max_itrs and error > error_max:
            # checks the interaction connections
            if self.parallel:
                self.check_edge_gpu()
            else:
                self.check_edge_run()

            # returns total vector after running spring force function
            vector = self.handle_springs()

            # runs through all objects and scales vector so that the cells don't move around too much
            for i in range(len(self.objects)):
                self.objects[i].update_constraints(self)

            error = Mag(vector)
            print("optimize error: " + str(error))

            # increment the iterations
            itrs += 1
        print("optimize iterations: " + str(itrs))


    def handle_springs(self):
        """ loops over all edges of the graph and performs
            spring collisions between the cells
        """
        # gets edges
        edges = np.array(self.network.edges())

        # a vector to hold the total vector change of all movement vectors added to it
        vector = [0, 0]

        # loops over all of the edges
        for i in range(len(edges)):
            edge = edges[i]
            # edge has two objects representing the nodes
            obj1 = edge[0]
            obj2 = edge[1]

            # check to make sure not the same object
            if obj1.ID != obj2.ID:

                # if they both aren't in motion remove edge between them
                if not obj1.motion and not obj2.motion:
                    self.network.remove_edge(obj1, obj2)

                # if one is in motion and the other is not
                if not obj1.motion and obj2.motion:
                    # get the locations
                    v1 = obj1.location
                    v2 = obj2.location
                    # get distance vector, magnitude, and normalize it
                    v12 = SubtractVec(v2, v1)
                    dist = Mag(v12)
                    norm = NormVec(v12)
                    # find the interaction length
                    interaction_length = self.spring_max * 2
                    # recalculate distance
                    dist = dist - interaction_length
                    # now get the spring constant strength
                    k = self.spring_constant
                    # scale the new distance by the spring constant
                    dist *= k
                    # direction of original distance vector
                    temp = ScaleVec(norm, dist)
                    # add vector to cell in motion where cell not in motion is anchor
                    obj2.add_displacement_vec(-temp)

                # if one is in motion and the other is not
                if obj1.motion and not obj2.motion:
                    # get the locations
                    v1 = obj1.location
                    v2 = obj2.location
                    # get distance vector, magnitude, and normalize it
                    v12 = SubtractVec(v2, v1)
                    dist = Mag(v12)
                    norm = NormVec(v12)
                    # find the interaction length
                    interaction_length = self.spring_max * 2
                    # recalculate distance
                    dist = dist - interaction_length
                    # now get the spring constant strength
                    k = self.spring_constant
                    # scale the new distance by the spring constant
                    dist *= k
                    # direction of original distance vector
                    temp = ScaleVec(norm, dist)
                    # add vector to cell in motion where cell not in motion is anchor
                    obj1.add_displacement_vec(temp)

                else:
                    # get the locations
                    v1 = obj1.location
                    v2 = obj2.location
                    # get distance vector, magnitude, and normalize it
                    v12 = SubtractVec(v2, v1)
                    dist = Mag(v12)
                    norm = NormVec(v12)
                    # find the interaction length
                    interaction_length = self.spring_max * 2
                    # recalculate distance
                    dist = dist - interaction_length
                    # now get the spring constant strength
                    k = self.spring_constant
                    # now we can apply the spring constraint to this
                    dist = (dist / 2.0) * k
                    # direction of original distance vector
                    temp = ScaleVec(norm, dist)
                    # add these vectors to the object vectors
                    obj1.add_displacement_vec(temp)
                    obj2.add_displacement_vec(-temp)

            # add movement vector to running count vector
            vector = AddVec(vector, temp)

        # return total vector
        return vector

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
        image1 = Image.new("RGB", (1500, 1500), color='white')
        draw = ImageDraw.Draw(image1)

        # bounds of the simulation used for drawing patch
        # inherit
        bounds = self.bounds

        # determines color and outline of the cells
        col_dict = {'Pluripotent': 'red', 'Differentiated': 'blue'}
        outline_dict = {'Pluripotent': 'red', 'Differentiated': 'blue'}

        # loops over all of the cells/nodes and draws a circle with corresponding color
        for i in range(len(cells)):
            node = cells[i]
            x, y = node.location
            r = node.radius
            col = col_dict[node.state]
            out = outline_dict[node.state]
            draw.ellipse((x - r + 250, y - r + 250, x + r + 250, y + r + 250), outline=out, fill=col)

        # loops over all of the bounds and draws lines to represent the grid
        for i in range(len(bounds)):
            x, y = bounds[i]
            if i < len(bounds) - 1:
                x1, y1 = bounds[i + 1]
            else:
                x1, y1 = bounds[0]
            r = 4
            draw.ellipse((x - r + 250, y - r + 250, x + r + 250, y + r + 250), outline='black', fill='black')
            draw.line((x + 250, y + 250, x1 + 250, y1 + 250), fill='black', width=10)

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
        self.draw_cell_image(self.network, base_path + "network_image" + str(int(self.time_counter)))


#######################################################################################################################
# CUDA functions


# @cuda.jit
# def initialize_grid_cuda(grid_array):
#     x, y = cuda.grid(2)
#     if x < grid_array.shape[1] and y < grid_array.shape[2]:
#         grid_array[0][x, y] += 10
#
# @cuda.jit
# def update_grid_cuda(grid_array):
#     x, y = cuda.grid(2)
#     if x < grid_array.shape[1] and y < grid_array.shape[2] and grid_array[0][x, y] >= 1:
#         grid_array[0][x, y] -= 1
#
# @cuda.jit
# def check_edge_cuda(locations, edges_array):
#     x, y = cuda.grid(2)
#     if x < edges_array.shape[0] and y < edges_array.shape[1]:
#         location_x1 = locations[x][0]
#         location_y1 = locations[x][1]
#         location_x2 = locations[y][0]
#         location_y2 = locations[y][1]
#         mag = ((location_x1 - location_x2)**2 + (location_y1 - location_y2)**2)**0.5
#         if mag <= 12 and x != y:
#             edges_array[x, y] = 1
