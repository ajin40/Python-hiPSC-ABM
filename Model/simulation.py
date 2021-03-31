import numpy as np
import random as r
import math
import time
import igraph
import os
import csv
import cv2
import pickle
import psutil
from abc import ABC, abstractmethod
from numba import cuda

from backend import *


class Simulation(ABC):
    """ This abstract class is the base for the CellSimulation object. It's used to
        make sure that any subclasses have necessary simulation attributes.
    """
    def __init__(self, paths):
        self.paths = paths    # the Paths object which holds any output paths

        # the running number of agents and the step to begin at (altered by continuation mode)
        self.number_agents = 0
        self.beginning_step = 1

        # hold the neighbors within a fixed radius of all agents with a graph
        self.neighbor_graph = igraph.Graph()
        self.graph_names = ["neighbor_graph"]

        # arrays to store the agents set to divide or to be removed
        self.agents_to_divide = np.array([], dtype=int)
        self.agents_to_remove = np.array([], dtype=int)

        # various other holders
        self.agent_array_names = list()  # store the variable names of each agent array
        self.agent_types = dict()  # hold the names of agent types defined in cellsimulation.py
        self.method_times = dict()  # store the runtimes of selected methods, used by record_time() decorator

    @abstractmethod
    def agent_initials(self):
        """ Abstract method in which the Simulation class should override.
        """
        pass

    @abstractmethod
    def steps(self):
        """ Abstract method in which the Simulation class should override.
        """
        pass

    def add_agents(self, number, agent_type=None):
        """ Add agents to the Simulation object and potentially add a agent type
            with bounds for defining alternative initial parameters.

                number (int): the number of agents being added to the Simulation object
                agent_type (str): the name of a agent type that can be used by agent_array() to only apply
                    initial parameters to these agents, instead of the entire array.
        """
        # add specified number of agents to each graph
        for graph_name in self.graph_names:
            self.__dict__[graph_name].add_vertices(number)

        # update the running number of agents and determine bounds for slice if agent_type is used
        begin = self.number_agents
        self.number_agents += number

        # if a agent type name is passed, hold the slice bounds for that particular agent type
        if agent_type is not None:
            self.agent_types[agent_type] = (begin, self.number_agents)

    def agent_array(self, array_name, agent_type=None, dtype=float, vector=None, func=None, override=None):
        """ Create a agent array in the Simulation object used to hold values
            for all agents and optionally specify initial parameters.

                array_name (str): the name of the variable made for the agent array in the Simulation object
                agent_type (str): see add_agents()
                dtype (object): the data type of the array, defaults to float
                vector (int): the length of the vector for each agent in the array
                func (object): a function called for each index of the array to specify initial parameters
                override (array): use the array passed instead of generating a new array
        """
        # if using existing array for agent array
        if override is not None:
            # make sure array have correct length, otherwise raise error
            if override.shape[0] != self.number_agents:
                raise Exception("Length of override array does not match number of agents in simulation!")

            # use the array and add to list of agent array names
            else:
                self.__dict__[array_name] = override
                self.agent_array_names.append(array_name)

        # otherwise make sure a default agent array exists for initial parameters
        else:
            # if no agent array in Simulation object, make one
            if not hasattr(self, array_name):
                # add the array name to a list for automatic addition/removal when agents divide/die
                self.agent_array_names.append(array_name)

                # get the dimensions of the array
                if vector is None:
                    size = self.number_agents  # 1-dimensional array
                else:
                    size = (self.number_agents, vector)  # 2-dimensional array (1-dimensional of vectors)

                # if using python string data type, use object data type instead
                if dtype == str or dtype == object:
                    # create agent array in Simulation object with NoneType as default value
                    self.__dict__[array_name] = np.empty(size, dtype=object)

                else:
                    # create agent array in Simulation object, with zeros as default values
                    self.__dict__[array_name] = np.zeros(size, dtype=dtype)

        # if no agent type parameter passed
        if agent_type is None:
            # if function is passed, apply initial parameter
            if func is not None:
                for i in range(self.number_agents):
                    self.__dict__[array_name][i] = func()

        # otherwise a agent type is passed
        else:
            # get the bounds of the slice
            begin = self.agent_types[agent_type][0]
            end = self.agent_types[agent_type][1]

            # if function is passed, apply initial parameter to slice
            if func is not None:
                for i in range(begin, end):
                    self.__dict__[array_name][i] = func()

    def info(self):
        """ Records the beginning of the step in real time and
            print out info about the simulation.
        """
        # records when the step begins, used for measuring efficiency
        self.step_start = time.perf_counter()  # time.perf_counter() is more accurate than time.time()

        # prints the current step number and the number of agents
        print("Step: " + str(self.current_step))
        print("Number of agents: " + str(self.number_agents))

    def assign_bins(self, distance, max_agents):
        """ Generalizes agent locations to a bin within lattice imposed on
            the agent space, used for a parallel fixed-radius neighbor search.
        """
        # If there is enough space for all agents that should be in a bin, break out of the loop. If there isn't
        # update the amount of needed space and put all the agents in bins. This will run once if the prediction
        # of max neighbors suffices, twice if it isn't right the first time.
        while True:
            # calculate the size of the array used to represent the bins and the bins helper array, include extra bins
            # for agents that may fall outside of the space
            bins_help_size = np.ceil(self.size / distance).astype(int) + 3
            bins_size = np.append(bins_help_size, max_agents)

            # create the arrays for "bins" and "bins_help"
            bins_help = np.zeros(bins_help_size, dtype=int)  # holds the number of agents currently in a bin
            bins = np.empty(bins_size, dtype=int)  # holds the indices of agents in a bin

            # generalize the agent locations to bin indices and offset by 1 to prevent missing agents that fall out of
            # the self space
            bin_locations = np.floor_divide(self.locations, distance).astype(int)
            bin_locations += 1

            # use jit function to speed up placement of agents
            bins, bins_help = assign_bins_jit(self.number_agents, bin_locations, bins, bins_help)

            # either break the loop if all agents were accounted for or revalue the maximum number of agents based on
            # the output of the function call and double it future calls
            new_max_agents = np.amax(bins_help)
            if max_agents >= new_max_agents:
                break
            else:
                max_agents = new_max_agents * 2  # double to prevent continual updating

        return bins, bins_help, bin_locations, max_agents

    @record_time
    def get_neighbors(self, distance=0.00002):
        """ For all agents, determines which agents fall within a fixed radius to
            denote a neighbor then stores this information in a graph (uses a bin/
            bucket sorting method).
        """
        # if a static variable has not been created to hold the maximum number of neighbors for a agent, create one
        if not hasattr(self, "gn_max_neighbors"):
            # begin with a low number of neighbors that can be revalued if the max neighbors exceeds this value
            self.gn_max_neighbors = 5

        # if a static variable has not been created to hold the maximum number of agents in a bin, create one
        if not hasattr(self, "gn_max_agents"):
            # begin with a low number of agents that can be revalued if the max number of agents exceeds this value
            self.gn_max_agents = 5

        # clear all of the edges in the neighbor graph
        self.neighbor_graph.delete_edges(None)

        # calls the function that generates an array of bins that generalize the agent locations in addition to a
        # creating a helper array that assists the search method in counting agents for a particular bin
        bins, bins_help, bin_locations, max_agents = self.assign_bins(distance, self.gn_max_agents)

        # update the value of the max number of agents in a bin
        self.gn_max_agents = max_agents

        # this will run once if all edges are included in edge_holder, breaking the loop. if not, this will
        # run a second time with an updated value for the number of predicted neighbors such that all edges are included
        while True:
            # create array used to hold edges, array to say if edge exists, and array to count the edges per agent
            length = self.number_agents * self.gn_max_neighbors
            edge_holder = np.zeros((length, 2), dtype=int)
            if_edge = np.zeros(length, dtype=bool)
            edge_count = np.zeros(self.number_agents, dtype=int)

            # call the nvidia gpu version
            if self.parallel:
                # send the following as arrays to the gpu
                bin_locations = cuda.to_device(bin_locations)
                locations = cuda.to_device(self.locations)
                bins = cuda.to_device(bins)
                bins_help = cuda.to_device(bins_help)
                distance = cuda.to_device(distance)
                edge_holder = cuda.to_device(edge_holder)
                if_edge = cuda.to_device(if_edge)
                edge_count = cuda.to_device(edge_count)
                max_neighbors = cuda.to_device(self.gn_max_neighbors)

                # allocate threads and blocks for gpu memory "threads per block" and "blocks per grid"
                tpb = 72
                bpg = math.ceil(self.number_agents / tpb)

                # call the cuda kernel with new gpu arrays
                get_neighbors_gpu[bpg, tpb](bin_locations, locations, bins, bins_help, distance, edge_holder, if_edge,
                                            edge_count, max_neighbors)

                # return the only the following array(s) back from the gpu
                edge_holder = edge_holder.copy_to_host()
                if_edge = if_edge.copy_to_host()
                edge_count = edge_count.copy_to_host()

            # call the jit cpu version
            else:
                edge_holder, if_edge, edge_count = get_neighbors_cpu(self.number_agents, bin_locations, self.locations,
                                                                     bins, bins_help, distance, edge_holder, if_edge,
                                                                     edge_count, self.gn_max_neighbors)

            # either break the loop if all neighbors were accounted for or revalue the maximum number of neighbors
            # based on the output of the function call and double it for future calls
            max_neighbors = np.amax(edge_count)
            if self.gn_max_neighbors >= max_neighbors:
                break
            else:
                self.gn_max_neighbors = max_neighbors * 2

        # reduce the edges to only edges that actually exist
        edge_holder = edge_holder[if_edge]

        # add the edges to the neighbor graph
        self.neighbor_graph.add_edges(edge_holder)

    def random_vector(self):
        """ Computes a random vector on the unit sphere centered
            at the origin.
        """
        # random angle on the agent
        theta = r.random() * 2 * math.pi

        # 2D vector: [x, y, 0]
        if self.size[2] == 0:
            return np.array([math.cos(theta), math.sin(theta), 0])

        # 3D vector: [x, y, z]
        else:
            phi = r.random() * 2 * math.pi
            radius = math.cos(phi)
            return np.array([radius * math.cos(theta), radius * math.sin(theta), math.sin(phi)])

    @record_time
    def temp(self):
        """ Pickle the current state of the Simulation object which can be used
            to continue a past simulation without losing information.
        """
        # get file name, use f-string
        file_name = f"{self.paths.name}_temp.pkl"

        # open the file in binary mode
        with open(self.paths.main_path + file_name, "wb") as file:
            # use the highest protocol: -1 for pickling the instance
            pickle.dump(self, file, -1)

    @record_time
    def step_values(self, arrays=None):
        """ Outputs a CSV file with value from the agent arrays with each
            row corresponding to a particular agent index.

            arrays -> (list) If arrays is None, all agent arrays are outputted otherwise only the
                agent arrays named in the list will be outputted.
        """
        # only continue if outputting agent values
        if self.output_values:
            # if arrays is None automatically output all agent arrays
            if arrays is None:
                agent_array_names = self.agent_array_names

            # otherwise only output arrays specified in list
            else:
                agent_array_names = arrays

            # get path and make sure directory exists
            directory_path = check_direct(self.paths.values)

            # get file name, use f-string
            file_name = f"{self.paths.name}_values_{self.current_step}.csv"

            # open the file
            with open(directory_path + file_name, "w", newline="") as file:
                # create CSV object and the following lists
                csv_file = csv.writer(file)
                header = list()    # header of the CSV (first row)
                data = list()    # holds the agent arrays

                # go through each of the agent arrays
                for array_name in agent_array_names:
                    # get the agent array
                    agent_array = self.__dict__[array_name]

                    # if the array is one dimensional
                    if agent_array.ndim == 1:
                        header.append(array_name)    # add the array name to the header
                        agent_array = np.reshape(agent_array, (-1, 1))  # resize array from 1D to 2D

                    # if the array is not one dimensional
                    else:
                        # create name for column based on slice of array ex. locations[0], locations[1], locations[2]
                        for i in range(agent_array.shape[1]):
                            header.append(array_name + "[" + str(i) + "]")

                    # add the array to the data holder
                    data.append(agent_array)

                # write header as the first row of the CSV
                csv_file.writerow(header)

                # stack the arrays to create rows for the CSV file and save to CSV
                data = np.hstack(data)
                csv_file.writerows(data)

    def data(self):
        """ Creates/adds a new line to the running CSV for data about the simulation
            such as memory, step time, number of agents and method profiling.
        """
        # get file name, use f-string
        file_name = f"{self.paths.name}_data.csv"

        # open the file
        with open(self.paths.main_path + file_name, "a", newline="") as file_object:
            # create CSV object
            csv_object = csv.writer(file_object)

            # create header if this is the beginning of a new simulation
            if self.current_step == 1:
                # header names
                header = ["Step Number", "Number Cells", "Step Time", "Memory (MB)"]

                # header with all the names of the functions with the "record_time" decorator
                functions_header = list(self.method_times.keys())

                # merge the headers together and write the row to the CSV
                csv_object.writerow(header + functions_header)

            # calculate the total step time and get memory of current python process in megabytes
            step_time = time.perf_counter() - self.step_start
            process = psutil.Process(os.getpid())
            memory = process.memory_info()[0] / 1024 ** 2

            # write the row with the corresponding values
            columns = [self.current_step, self.number_agents, step_time, memory]
            function_times = list(self.method_times.values())
            csv_object.writerow(columns + function_times)

    def create_video(self):
        """ Write all of the step images from a simulation to video file in the
            main simulation directory.
        """
        # continue if there is an image directory
        if os.path.isdir(self.paths.images):
            # get all of the images in the directory and the number of images
            file_list = [file for file in os.listdir(self.paths.images) if file.endswith(".png")]
            image_count = len(file_list)

            # only continue if image directory has images in it
            if image_count > 0:
                print("\nCreating video...")

                # sort the file list so "2, 20, 3, 31..." becomes "2, 3, ..., 20, ..., 31"
                file_list = sorted(file_list, key=sort_naturally)

                # sample the first image to get the dimensions of the image, and then scale the image
                size = cv2.imread(self.paths.images + file_list[0]).shape[0:2]
                scale = self.video_quality / size[1]
                new_size = (self.video_quality, int(scale * size[0]))

                # get the video file path, use f-string
                file_name = f"{self.paths.name}_video.mp4"
                video_path = self.paths.main_path + file_name

                # create the file object with parameters from simulation and above
                codec = cv2.VideoWriter_fourcc(*"mp4v")
                video_object = cv2.VideoWriter(video_path, codec, self.fps, new_size)

                # go through sorted image list, reading and writing each image to the video object
                for i in range(image_count):
                    image = cv2.imread(self.paths.images + file_list[i])    # read image from directory
                    if size != new_size:
                        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)    # scale down if necessary
                    video_object.write(image)    # write to video
                    progress_bar(i, image_count)    # show progress

                # close the video file
                video_object.release()

        # print end statement...super important...don't remove or model won't run!
        print("\n\nThe simulation is finished. May the force be with you.\n")
