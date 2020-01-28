import os, shutil
import platform
import networkx as nx
import numpy as np
from Model_SimulationMath import *
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

    def __init__(self, name, path, start_time, end_time, time_step, pluri_div_thresh, diff_div_thresh,pluri_to_diff, size, interaction_max, diff_surround_value, functions):
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
        self.time = float(start_time)
        self.functions = functions
        self.size = size
        self.grid = np.zeros(self.size)
        self.pluri_div_thresh = pluri_div_thresh
        self.diff_div_thresh = diff_div_thresh
        self.pluri_to_diff = pluri_to_diff
        self.interaction_max = interaction_max
        self.diff_surround_value = diff_surround_value

        

            
        
        #make a list to keep track of the sim objects
        self.objects = []
        

        #keep track of the fixed constraints
        self._fixed_constraints = nx.Graph()
        self.network = nx.Graph()
        #add the add/remove buffers
        self._objects_to_remove = []
        self._objects_to_add = []

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

        self.objects.append(sim_object)
        #also add it to the network
        self.network.add_node(sim_object)


    def inc_current_ID(self):
        """Increments the ID of cell by 1 each time called
        """
        self._current_ID += 1


    def remove_object(self, sim_object):
        """ Removes the specified object from the list
        """
        self.objects.remove(sim_object)
        #remove it from the network
        self.network.remove_node(sim_object)
        #also remove it from the fixed network
        try:
            self._fixed_constraints.remove_node(sim_object)
        except nx.NetworkXError:
            pass


    def add_object_to_addition_queue(self, sim_object):
        """ Will add an object to the simulation object queue
            which will be added to the simulation at the end of
            the update phase.
        """
        self._objects_to_add.append(sim_object)
        #increment the sim ID
        self._current_ID += 1

        
    def add_object_to_removal_queue(self, sim_object):
        """ Will add an object to the simulation object queue
            which will be removed from the simulation at the end of
            the update phase.
        """
        #place the object in the removal queue
        self._objects_to_remove.append(sim_object)


    def add_fixed_constraint(self, obj1, obj2):
        """ Adds a fixed immutable constraint between two objects which is
            processed with the other optimization constraints
        """
        self._fixed_constraints.add_edge(obj1, obj2)


    def get_ID(self):
        """ Returns the current unique ID the simulation is on
        """
        return self._current_ID

        
    def run(self):
        """ Runs the simulation until either the stopping criteria is reached
            or the simulation time runs out.
        """
        self.time = self.start_time

        #try to make a new directory for the simulation
        try:
            os.mkdir(self.path + self._sep + self.name)
        except OSError:
            #direcotry already exsists... overwrite it
            print("Directory already exists... overwriting directory")
        struct_path = os.getcwd()

        for i in range(self.size[1]):
            for j in range(self.size[2]):

                self.grid[np.array([0]),np.array([i]),np.array([j])] = r.randint(0,10)


        #save the initial state configuration
        self.save_file()
        self.collide_lowDens()
        #now run the loop
        while self.time <= self.end_time:
            print("Time: " + str(self.time))
            print("Number of objects: " + str(len(self.objects)))

            #Update the objects and gradients
            self.update()
            self.diff_surround()
            #remove/add any objects
            self.update_object_queue()
            #perform physics
            # try:
            #     self.collide()
            # except:
            #     print "manual collide"
            #     self.collide_lowDens()

            self.collide_lowDens_run()
                #now optimize the resultant constraints

            self.random_movement()


            self.calculate_compression()
            self.optimize()


            #increment the time
            self.time += self.time_step
            #save 
            self.save_file()
        self.image_to_video()

    def random_movement(self):
        print("running")
        for i in range(len(self.objects)):
            if self.objects[i].state == "Pluripotent":
                temp_x = self.objects[i].location[0] + r.uniform(-1,1) * 30
                temp_y = self.objects[i].location[1] + r.uniform(-1,1) * 30
                if temp_x <= 1000 and temp_x >= 0 and temp_y <= 1000 and temp_y >= 0:
                    self.objects[i].location[0] = temp_x
                    self.objects[i].location[1] = temp_y




    def image_to_video(self):
        """Creates a video out of the png images
        """
        base_path = self.path + self._sep + self.name + self._sep

        # path = "C:\\Python27\\Bool Model\\2.0\\*png"
        img_array = []
        for i in range(int(self.end_time)+2):
            img = cv2.imread(base_path + 'network_image' + str(int(i)) + ".png")

            img_array.append(img)

        out = cv2.VideoWriter(base_path + 'network_video.mp4', cv2.VideoWriter_fourcc(*"DIVX"), 0.5, (1500, 1500))

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
        self._objects_to_remove = []
        self._objects_to_add= []

        
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
                
                
    # def collide(self):
    #     """ Handles the collision map generation
    #     """
    #     #first make a list of points for the delaunay to use
    #     k=0
    #     for i in range(0,len(self.objects)):
    #         k+=self.objects[i].location[2]
    #     if k==0:
    #         n=2
    #     else:
    #         n=3
    #     points = np.zeros((len(self.objects), n))
    #
    #     for i in range(0, len(self.objects)):
    #         #Now we can add these points to the list
    #         if n==2:
    #             points[i] = [self.objects[i].location[0], self.objects[i].location[1]]
    #         else:
    #             points[i]=self.objects[i].location
    #     #now perform the nearest neighbor assessment by building a delauny triangulation
    #
    #     tri = Delaunay(points)
    #     #keep track of this as a network
    #     self.network = nx.Graph()
    #     #add all the simobjects
    #     self.network.add_nodes_from(self.objects)
    #     #get the data
    #     nbs = tri.nx.vertices
    #     #iterate over all the nbs to cull the data for entry into the list
    #     for i in range(0, len(nbs)):
    #         #loop over all of the combination in this list
    #         ns = nbs[i]
    #         for a in range(0, len(ns)):
    #             for b in range(a+1, len(ns)):
    #
    #                 self.network.add_edge(self.objects[ns[a]], self.objects[ns[b]])
    #
    #
    #
    #     #now loop over all of these cells and cull the interaction lists
    #     edges = list(self.network.edges())
    #     #keep track of interactions checked
    #
    #     for i in range(0, len(edges)):
    #         #get the edge in question
    #         edge = edges[i]
    #         #first find the distance of the edge
    #         obj1 = edge[0]
    #         obj2 = edge[1]
    #         #figure out if the interaction is ok
    #         #gte the minimum interaction length
    #         l1 = obj1.get_max_interaction_length()
    #         l2 = obj2.get_max_interaction_length()
    #         #add these together to get the connection length
    #         interaction_length = l1 + l2
    #         #get the distance
    #         dist_vec = SubtractVec(obj2.location, obj1.location)
    #         #get the magnitude
    #         dist = Mag(dist_vec)
    #         #if it does not meet the criterion, remove it from the list
    #         if dist > interaction_length:
    #             self.network.remove_edge(obj1, obj2)


    def collide_lowDens(self):
        """ If there are too few cells, or the first iteration of the model. 
        """

        #keep track of this as a network
        # self.network = nx.Graph()
        #add all the simobjects
        self.network.add_nodes_from(self.objects)

        #Create connection between all cells 
        for i in range(0,len(self.objects)):
            for j in range(i+1,len(self.objects)):
                l1 = self.interaction_max
                l2 = self.interaction_max
                # add these together to get the connection length
                interaction_length = l1 + l2
                # get the distance
                dist_vec = SubtractVec(self.objects[i].location, self.objects[j].location)
                # get the magnitude
                dist = Mag(dist_vec)
                if dist <= interaction_length:
                    self.network.add_edge(self.objects[i], self.objects[j])

    def collide_lowDens_run(self):
        """ If there are too few cells, or the first iteration of the model.
        """

        # keep track of this as a network
        # self.network = nx.Graph()
        # add all the simobjects
        self.network.add_nodes_from(self.objects)

        # Create connection between all cells
        for i in range(0, len(self.objects)):
            for j in range(i + 1, len(self.objects)):
                l1 = self.interaction_max
                l2 = self.interaction_max
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
                if dist <= interaction_length:
                    self.network.add_edge(self.objects[i], self.objects[j])

    def optimize(self):
        #apply constraints from each object and update the positions
        #keep track of the global col and opt vectors
        opt = 2
        col = 2
        fixed = 2
        itrs = 0
        max_itrs = 50
        avg_error = 0.2 # um
        while (opt + col + fixed) >= avg_error and itrs < max_itrs:
            #handle the spring constraints
            opt, col = self._handle_spring_constraints()
            #handle to fixed constraints
            fixed = self._handle_fixed_constraints()
            #now loop over and update all of the constraints
            for i in range(0, len(self.objects)):
                #update these collision and optimization vectors
                #as well as any other constraints
                self.objects[i].update_constraints(self.time_step)
            #increment the itrations
            itrs += 1






    def _handle_spring_constraints(self):
        """
        """
        col = 0
        opt = 0
        edges = list(self.network.edges())
        cent=self.get_center()
        rad_avg=0.75*self.getAverageRadialDistance()
        for i in range(len(edges)):
            edge=edges[i]
            obj1=edge[0]
            obj2=edge[1]
            if obj1.ID != obj2.ID:
                v1=obj1.location
                v2=obj2.location
                v12=SubtractVec(v2,v1)
                dist=Mag(v12)
                norm=NormVec(v12)
                r_sum=obj1.radius+obj2.radius
#               comp_diff=abs(obj1.compress-obj2.compress)
#                 if r_sum >= dist:
                # #                     d1=Distance(v1,cent)
                # #                     d2=Distance(v2,cent)
                # #                     c1 = obj1.cmpr_direct
                # #                     c2 = obj2.cmpr_direct
                # #                     d3=(d1+d2)/2.0 - rad_avg
                # #                     mod=0.5*d3/(abs(d3)+(rad_avg/2.0))
                # #                     d = -norm*((r_sum-dist)*0.35*(1-mod))
                # #                     d_= -norm*((r_sum-dist)*0.45*(1+mod))
                # #                     obj2.add_fixed_constraint_vec(-d)
                # #                     obj1.add_fixed_constraint_vec(d)
                # #                     if d1>d2:
                # #                         obj2.add_fixed_constraint_vec(d_)
                # #                         obj2.add_fixed_constraint_vec(-0.4*c2)
                # #                         obj1.add_fixed_constraint_vec(-0.4*c1)
                # #                     else:
                # #                         obj1.add_fixed_constraint_vec(-d_)
                # #                         obj2.add_fixed_constraint_vec(-0.4*c2)
                # #                         obj1.add_fixed_constraint_vec(-0.4*c1)
                # #                     col+=Mag(d)*2
 
        #return the average opt and col values
                l1 = obj1.get_interaction_length()
                l2 = obj2.get_interaction_length()
                #now figure out how far off the connection length  is
                #from that distance
                dist = dist - (l1 + l2)
                #now get the spring constant strength
                k1 = obj1.get_spring_constant(obj2)
                k2 = obj2.get_spring_constant(obj1)
                k = min(k1, k2)
                #now we can apply the spring constraint to this
                dist = (dist/2.0) * k
                #make sure it has the correct direction
                temp = ScaleVec(norm, dist)
                #add these vectors to the object vectors
                obj1.add_displacement_vec(temp)
                obj2.add_displacement_vec(-temp)
                
                #add to the global opt vec
                opt += Mag(temp)
                
        opt = opt / (len(edges)*2.0)
        col = col / (len(edges)*2.0)
        return opt, col


    def _handle_fixed_constraints(self):
        """
        """
        error = 0
        edges = list(self._fixed_constraints.edges())
        for i in range(0, len(edges)):
            #for each edge optimize the spring interaction
            edge = edges[i]
            #get the objects
            obj1 = edge[0]
            obj2 = edge[1]
            dist_vec = SubtractVec(obj2.location, obj1.location)
            if obj1.z==0 and obj2.z==0:
                dist_vec[2]=0
            ########  remove vertical displacement
            dist_vec[2]=0
            #############
            dist = Mag(dist_vec)
            #also compute the normal
            norm = NormVec(dist_vec)
            #get the object radii
            r_sum = obj1.radius + obj2.radius
            #then apply the collision
            d = -norm*((r_sum-dist)*0.5)
            obj1.add_fixed_constraint_vec(d)
            obj2.add_fixed_constraint_vec(-d)

            #add this to the collision vec
            error += Mag(d)*2
        #calculate the average error
        try:
            error = error / len(edges)
        except ZeroDivisionError:
            error = 0
        return error


    def get_center(self):
        """ Returns the center of the simulation
            return - point in the form of (x,y,z)
        """
        n=len(self.objects[1].location)
        if n==3:
            cent = (0,0,0)
        else:
            cent=(0,0)
                
        for i in range(0, len(self.objects)):            
            cent = AddVec(self.objects[i].location, cent)
        #then scale the vector
        cent = ScaleVec(cent, 1.0/ len(self.objects))
        #and return it
        return cent


    def getAverageRadialDistance(self):
        cent=self.get_center()
        agents=self.objects
        radii = []
        for i in range(0, len(agents)):
            radius = SubtractVec(agents[i].location, cent)
            #add tot he list
            radii.append(Mag(radius))
        
        return np.average(radii)


    def draw_cell_image(self, network, path):
        """Turns the graph into an image at each timestep
        """
        nodes = network.nodes()
        image1 = Image.new("RGB", (1500, 1500), color='white')
        draw = ImageDraw.Draw(image1)
        # bounds = nodes[0].bounds
        bounds = [[0,0], [0,1000], [1000,1000], [1000,0]]


        cmin = 100
        cmax = 0
        col_dict = {'Pluripotent': 'red', 'Differentiated': 'blue'}
        outline_dict = {'Pluripotent': 'red', 'Differentiated': 'blue'}

        for i in range(len(nodes)):
            node = nodes[i]
            x, y = node.location
            r = node.radius
            col = col_dict[node.state]
            out = outline_dict[node.state]
            draw.ellipse((x - r + 200, y - r + 200, x + r + 200, y + r + 200), outline=out, fill=col)


        # for i in range(1000):
        #     for j in range(1000):
        #         if self.grid[np.array([0]), np.array([i]), np.array([j])] > 0:
        #             rad = 0.1
        #             draw.ellipse((i - rad + 200, j - rad + 200, i + rad + 200, j + rad + 200), outline='green', fill = "green")

        for i in range(len(bounds)):
            x, y = bounds[i]
            if i < len(bounds) - 1:
                x1, y1 = bounds[i + 1]
            else:
                x1, y1 = bounds[0]
            r = 4
            draw.ellipse((x - r + 200, y - r + 200, x + r + 200, y + r + 200), outline='black', fill='black')
            draw.line((x + 200, y + 200, x1 + 200, y1 + 200), fill='black', width=10)




        image1.save(path + "network_image" + str(int(self.time)) + ".png", 'PNG')




    def location_to_text(self, path):
        """Outputs a txt file of the cell coordinates and the boolean values
        """
        new_file = open(path, "w")


        for i in range(0, len(self.objects)):
            self.objects[i].funct_1 = int(self.objects[i].funct_1)
            self.objects[i].funct_2 = int(self.objects[i].funct_2)
            self.objects[i].funct_3 = int(self.objects[i].funct_3)
            self.objects[i].funct_4 = int(self.objects[i].funct_4)
            self.objects[i].funct_5 = int(self.objects[i].funct_5)

            ID = str(self.objects[i].ID) + ","
            x_coord = str(round(self.objects[i].location[0],1))+ ","
            y_coord = str(round(self.objects[i].location[1],1))+ ","
            x1 = str(self.objects[i].funct_1) + ","
            x2 = str(self.objects[i].funct_2) + ","
            x3 = str(self.objects[i].funct_3) + ","
            x4 = str(self.objects[i].funct_4) + ","
            x5 = str(self.objects[i].funct_5) + ","
            diff = str(self.objects[i].diff_timer) + ","
            div = str(self.objects[i].division_timer) + ","
            state = self.objects[i].state + ","
            line = ID + x_coord + y_coord + state + x1 + x2 + x3 + x4 + x5 + diff + div

            new_file.write(line + "\n")


    def save_file(self):
        """ Saves the simulation snapshot.
        """

        #get the base path
        base_path = self.path +self._sep +self.name + self._sep
        #First save the network files
        n1_path = base_path + "network_pickle" + str(int(self.time)) + ".gpickle"
        n2_path = base_path + "network_values" + str(int(self.time)) + ".txt"

        # nx.write_gpickle(self.network, n1_path)
        self.location_to_text(n2_path)
        self.draw_cell_image(self.network, base_path)




