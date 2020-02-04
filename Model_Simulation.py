import os, shutil
import platform
import networkx as nx
import numpy as np
from Model_SimulationObject import *
from scipy.spatial import *
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
import pickle
import time
import cv2
import glob
import random as r


class Simulation(object):

    def __init__(self, name, path, start_time, end_time, time_step, pluri_div_thresh, diff_div_thresh,pluri_to_diff, size, spring_max, diff_surround_value, functions):
        """ Initialization function for the simulation setup.
            name - the simulation name (string)
            path - the path to save the simulation information to (string)
            start_time - the start time for the simulation (float)
            end_time - the end time for the simulation (float)
            time_step - the time step to increment the simulation by (float)
            division_times - Division time for each cell type
            source_sink_params - The production and degradation constants for each extracellular molecule and each cell type
            size - The size of the array (dimension, rows, columns)
        """

        #set the base parameters
        #do some basic type checking
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
        self._time = float(start_time)
        self.functions = functions
        self.size = size
        self.grid = np.zeros(self.size)
        self.pluri_div_thresh = pluri_div_thresh
        self.diff_div_thresh = diff_div_thresh
        self.pluri_to_diff = pluri_to_diff
        self.spring_max = spring_max
        self.diff_surround_value = diff_surround_value
        self.image_counter = 0

        #make a list to keep track of the sim objects
        self.objects = np.array([])

        self.network = nx.Graph()
        #add the add/remove buffers
        self._objects_to_remove = np.array([])
        self._objects_to_add = np.array([])

        #also keep track of the current sim object ID
        self._current_ID = 0

        #keep track of the file separator to use
        if platform.system() == "Windows":
            #windows
            self._sep = "\\"
        else:
            #linux/unix
            self._sep = "/"

#######################################################################################################################



    def call_functions(self):
        """returns functions defined in Model_Setup
        """
        return self.functions

    def add_object(self, sim_object):
        """ Adds the specified object to the list
        """

        self.objects = np.append(self.objects, sim_object)
        #also add it to the network
        self.network.add_node(sim_object)

    def inc_current_ID(self):
        """Increments the ID of cell by 1 each time called
        """
        self._current_ID += 1

    def remove_object(self, sim_object):
        """ Removes the specified object from the list
        """
        self.objects = np.delete(self.objects, sim_object)
        #remove it from the network
        self.network.remove_node(sim_object)

    def add_object_to_addition_queue(self, sim_object):
        """ Will add an object to the simulation object queue
            which will be added to the simulation at the end of
            the update phase.
        """
        self._objects_to_add = np.append(self._objects_to_add, sim_object)
        #increment the sim ID
        self._current_ID += 1

    def add_object_to_removal_queue(self, sim_object):
        """ Will add an object to the simulation object queue
            which will be removed from the simulation at the end of
            the update phase.
        """
        #place the object in the removal queue
        self._objects_to_remove = np.append(self._objects_to_remove, sim_object)

    def get_ID(self):
        """ Returns the current unique ID the simulation is on
        """
        return self._current_ID


    def run(self):
        """ Runs the simulation until either the stopping criteria is reached
            or the simulation time runs out.
        """
        self._time = self.start_time


        #try to make a new directory for the simulation
        try:
            os.mkdir(self.path + self._sep + self.name)
        except OSError:
            #direcotry already exsists... overwrite it
            print("Directory already exists... overwriting directory")

        for i in range(self.size[1]):
            for j in range(self.size[2]):

                self.grid[np.array([0]),np.array([i]),np.array([j])] = r.randint(0,10)

        #save the initial state configuration
        self.save_file()
        self.collide()
        #now run the loop
        while self._time <= self.end_time:
            print("Time: " + str(self._time))
            print("Number of objects: " + str(len(self.objects)))

            #Update the objects and gradients
            time1 = time.time()

            self.update()

            time2 = time.time()
            print("update", time2 - time1)

            time1 = time.time()

            self.diff_surround()

            time2 = time.time()
            print("diff_surround", time2 - time1)

            #remove/add any objects
            time1 = time.time()
            self.update_object_queue()

            time2 = time.time()
            print("update_object_queue", time2 - time1)

            time1 = time.time()
            self.collide_run()
            time2 = time.time()
            print("collide", time2 - time1)

            time1 = time.time()
            self.random_movement()
            time2 = time.time()
            print("random", time2 - time1)

            time1 = time.time()
            self.calculate_compression()
            time2 = time.time()
            print("compression", time2 - time1)

            time1 = time.time()
            self.optimize(0.00001, 20)
            time2 = time.time()
            print("optimize", time2 - time1)

            #increment the time
            self._time += self.time_step
            #save
            time1 = time.time()
            self.save_file()
            time2 = time.time()
            print("image",time2-time1)
        self.image_to_video()


    def random_movement(self):
        for i in range(len(self.objects)):
            if self.objects[i].motion:
                temp_x = self.objects[i].location[0] + r.uniform(-1, 1) * 10
                temp_y = self.objects[i].location[1] + r.uniform(-1, 1) * 10
                if temp_x <= 1000 and temp_x >= 0 and temp_y <= 1000 and temp_y >= 0:
                    self.objects[i].location[0] = temp_x
                    self.objects[i].location[1] = temp_y




    def image_to_video(self):
        """Creates a video out of the png images
        """
        base_path = self.path + self._sep + self.name + self._sep

        # path = "C:\\Python27\\Bool Model\\2.0\\*png"
        img_array = []
        for i in range(self.image_counter):
            img = cv2.imread(base_path + 'network_image' + str(1.0 + int(i)/10) + ".png")

            img_array.append(img)

        out = cv2.VideoWriter(base_path + 'network_video.mp4', cv2.VideoWriter_fourcc(*"DIVX"), 2.0, (1500, 1500))

        for i in range(len(img_array)):
            out.write(img_array[i])

        out.release()


    def update_object_queue(self):
        """ Updates the object add and remove queue
        """
        print("Adding " + str(len(self._objects_to_add)) + " objects...")
##        print("Removing " + str(len(self._objects_to_remove)) + " objects...")
        for i in range(0, len(self._objects_to_remove)):
            self.remove_object(self._objects_to_remove[i])
        for i in range(0, len(self._objects_to_add)):
            self.add_object(self._objects_to_add[i])
        #then clear these lists
        self._objects_to_remove = np.array([])
        self._objects_to_add= np.array([])

        
    def calculate_compression(self):
        objects=self.objects
        for i in range(len(objects)):
            objects[i].compress_force(self)


    def diff_surround(self):
        objects=self.objects
        for i in range(len(objects)):
            objects[i].diff_surround_funct(self)

            
    def update(self):
        """ Updates all of the objects in the simulation
        """

        for i in range(self.size[1]):
            for j in range(self.size[2]):
                self.grid[np.array([0]),np.array([i]),np.array([j])] += -1

        split = 1
        for p in range(0, split):
            dt = self.time_step / float(split)
            for i in range(0, len(self.objects)):
                self.objects[i].update(self, dt)
                

    def collide(self):

        #Create connection between all cells 
        for i in range(0,len(self.objects)):
            for j in range(i+1,len(self.objects)):
                l1 = self.spring_max
                l2 = self.spring_max
                # add these together to get the connection length
                interaction_length = l1 + l2
                # get the distance
                dist_vec = SubtractVec(self.objects[i].location, self.objects[j].location)
                # get the magnitude
                dist = Mag(dist_vec)
                if dist <= interaction_length:
                    self.network.add_edge(self.objects[i], self.objects[j])


    def collide_run(self):
        # Create connection between all cells
        for i in range(0, len(self.objects)):
            for j in range(i + 1, len(self.objects)):
                l1 = self.spring_max
                l2 = self.spring_max
                # add these together to get the connection length
                interaction_length = l1 + l2
                # get the distance
                dist_vec = SubtractVec(self.objects[i].location, self.objects[j].location)
                # get the magnitude
                dist = Mag(dist_vec)
                if dist > interaction_length:
                    try:
                        self.network.remove_edge(self.objects[i], self.objects[j])
                    except:
                        pass
                if dist <= interaction_length and (self.objects[i].motion or self.objects[j].motion):
                    self.network.add_edge(self.objects[i], self.objects[j])

    def check_interaction_length(self):

        for i in range(0, len(self.objects)):
            for j in range(i + 1, len(self.objects)):
                l1 = self.spring_max
                l2 = self.spring_max
                # add these together to get the connection length
                interaction_length = l1 + l2
                # get the distance
                dist_vec = SubtractVec(self.objects[i].location, self.objects[j].location)
                # get the magnitude
                dist = Mag(dist_vec)
                if dist > interaction_length:
                    try:
                        self.network.remove_edge(self.objects[i], self.objects[j])
                    except:
                        pass
                if dist <= interaction_length and (self.objects[i].motion or self.objects[j].motion):
                    self.network.add_edge(self.objects[i], self.objects[j])




    def optimize(self, error_max, max_itrs):

        itrs = 0
        error = error_max * 2
        while itrs < max_itrs and error > error_max :
            self.check_interaction_length()
            vector = self.handle_springs()

            #now loop over and update all of the constraints
            for i in range(0, len(self.objects)):
                #update these collision and optimization vectors
                #as well as any other constraints
                self.objects[i].update_constraints(self.time_step)

            vector = ScaleVec(vector, 1.0)
            error = Mag(vector)
            #increment the iterations
            itrs += 1
        print(itrs)


    def handle_springs(self):
        edges = np.array(self.network.edges())
        vector = [0,0]
        for i in range(len(edges)):
            edge=edges[i]
            obj1=edge[0]
            obj2=edge[1]
            if obj1.ID != obj2.ID:

                if not obj1.motion and not obj2.motion:
                    self.network.remove_edge(obj1, obj2)

                if not obj1.motion and obj2.motion:
                    v1 = obj1.location
                    v2 = obj2.location
                    v12 = SubtractVec(v2, v1)
                    dist = Mag(v12)
                    norm = NormVec(v12)
                    l1 = self.spring_max
                    l2 = self.spring_max
                    # now figure out how far off the connection length  is
                    # from that distance
                    dist = dist - (l1 + l2)
                    # now get the spring constant strength
                    k1 = obj1.get_spring_constant(obj2)
                    k2 = obj2.get_spring_constant(obj1)
                    k = min(k1, k2)
                    # now we can apply the spring constraint to this
                    dist *=  k
                    # make sure it has the correct direction
                    temp = ScaleVec(norm, dist)
                    # add these vectors to the object vectors
                    obj2.add_displacement_vec(-temp)

                if obj1.motion and not obj2.motion:
                    v1 = obj1.location
                    v2 = obj2.location
                    v12 = SubtractVec(v2, v1)
                    dist = Mag(v12)
                    norm = NormVec(v12)
                    l1 = self.spring_max
                    l2 = self.spring_max
                    # now figure out how far off the connection length  is
                    # from that distance
                    dist = dist - (l1 + l2)
                    # now get the spring constant strength
                    k1 = obj1.get_spring_constant(obj2)
                    k2 = obj2.get_spring_constant(obj1)
                    k = min(k1, k2)
                    # now we can apply the spring constraint to this
                    dist *= k
                    # make sure it has the correct direction
                    temp = ScaleVec(norm, dist)
                    # add these vectors to the object vectors
                    obj1.add_displacement_vec(temp)

                else:
                    v1 = obj1.location
                    v2 = obj2.location
                    v12 = SubtractVec(v2, v1)
                    dist = Mag(v12)
                    norm = NormVec(v12)
                    l1 = self.spring_max
                    l2 = self.spring_max
                    # now figure out how far off the connection length  is
                    # from that distance
                    dist = dist - (l1 + l2)
                    # now get the spring constant strength
                    k1 = obj1.get_spring_constant(obj2)
                    k2 = obj2.get_spring_constant(obj1)
                    k = min(k1, k2)
                    # now we can apply the spring constraint to this
                    dist = (dist / 2.0) * k
                    # make sure it has the correct direction
                    temp = ScaleVec(norm, dist)
                    # add these vectors to the object vectors
                    obj1.add_displacement_vec(temp)
                    obj2.add_displacement_vec(-temp)
            vector = AddVec(vector, temp)

        return vector





    def draw_cell_image(self, network, path):
        """Turns the graph into an image at each timestep
        """
        self.image_counter += 1
        cells = list(network.nodes)
        image1 = Image.new("RGB", (1500, 1500), color='white')
        draw = ImageDraw.Draw(image1)
        bounds = [[0,0], [0,1000], [1000,1000], [1000,0]]

        col_dict = {'Pluripotent': 'red', 'Differentiated': 'blue'}
        outline_dict = {'Pluripotent': 'red', 'Differentiated': 'blue'}

        for i in range(len(cells)):
            node = cells[i]
            x, y = node.location
            r = node.radius
            col = col_dict[node.state]
            out = outline_dict[node.state]
            draw.ellipse((x - r + 200, y - r + 200, x + r + 200, y + r + 200), outline=out, fill=col)

        for i in range(len(bounds)):
            x, y = bounds[i]
            if i < len(bounds) - 1:
                x1, y1 = bounds[i + 1]
            else:
                x1, y1 = bounds[0]
            r = 4
            draw.ellipse((x - r + 200, y - r + 200, x + r + 200, y + r + 200), outline='black', fill='black')
            draw.line((x + 200, y + 200, x1 + 200, y1 + 200), fill='black', width=10)

        image1.save(path + ".png", 'PNG')


    def location_to_text(self, path):
        """Outputs a txt file of the cell coordinates and the boolean values
        """
        new_file = open(path, "w")

        for i in range(0, len(self.objects)):

            ID = str(self.objects[i].ID) + ","
            x_coord = str(round(self.objects[i].location[0],1))+ ","
            y_coord = str(round(self.objects[i].location[1],1))+ ","
            x1 = str(self.objects[i].funct_1) + ","
            x2 = str(self.objects[i].funct_2) + ","
            x3 = str(self.objects[i].funct_3) + ","
            x4 = str(self.objects[i].funct_4) + ","
            x5 = str(self.objects[i].funct_5) + ","
            diff = str(round(self.objects[i].diff_timer,1)) + ","
            div = str(round(self.objects[i].division_timer,1)) + ","
            state = self.objects[i].state + ","
            motion = str(self.objects[i].motion)
            line = ID + x_coord + y_coord + state + x1 + x2 + x3 + x4 + x5 + motion + diff + div
            new_file.write(line + "\n")


    def save_file(self):
        """ Saves the simulation snapshot.
        """
        #get the base path
        base_path = self.path +self._sep +self.name + self._sep
        #First save the network files
        n2_path = base_path + "network_values" + str(int(self._time)) + ".txt"
        self.location_to_text(n2_path)

        self.draw_cell_image(self.network, base_path  + "network_image" + str(int(self._time)))


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