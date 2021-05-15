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
    """ This abstract class makes sure any subclasses have the necessary
        simulation attributes.
    """
    def __init__(self, paths, name):
        self.paths = paths  # the Paths object which holds any output paths
        self.name = name    # name of the simulation

        # the running number of agents and the step to begin at (updated by continuation mode)
        self.number_agents = 0
        self.beginning_step = 1

        # hold the names of the agent arrays and the names of any graphs (each agent is a node)
        self.agent_array_names = list()
        self.graph_names = list()

        # store the runtimes of methods with @record_time decorator
        self.method_times = dict()

    @abstractmethod
    def agent_initials(self):
        """ Make sure subclass has agent_initials method. """
        pass

    @abstractmethod
    def steps(self):
        """ Make sure subclass has steps method. """
        pass

    def add_agents(self, number, agent_type=None):
        """ Adds number of agents to the simulation potentially with agent_type marker.

            - number: the number of agents being added
            - agent_type: string marker used to apply initial conditions to only these
              agents
        """
        # determine bounds for array slice and increase total agents
        begin = self.number_agents
        self.number_agents += number

        # if an agent type is passed
        if agent_type is not None:
            # make sure holder for types exists
            if not hasattr(self, "agent_types"):
                self.agent_types = dict()

            # set key value to tuple of the array slice
            self.agent_types[agent_type] = (begin, self.number_agents)

    def agent_array(self, array_name, agent_type=None, dtype=float, vector=None, func=None, override=None):
        """ Adds an agent array to the simulation used to hold values for all agents.

            - array_name: the name of the variable made for the agent array
            - agent_type: string marker from add_agents()
            - dtype: the data type of the array
            - vector: if 2-dimensional, the length of the vector for each agent
            - func: a function called for each index of the array to specify initial
              parameters
            - override: pass existing array instead of generating a new array
        """
        # if using existing array
        if override is not None:
            # make sure array has correct length
            if override.shape[0] != self.number_agents:
                raise Exception("Length of override array does not match number of agents in simulation!")

            # create instance variable and add array name to holder
            else:
                self.__dict__[array_name] = override
                self.agent_array_names.append(array_name)

        # otherwise check if instance variable exists and try to make new array
        elif not hasattr(self, array_name):
            # add array name to holder
            self.agent_array_names.append(array_name)

            # get the dimensions of the array
            if vector is None:
                size = self.number_agents  # 1-dimensional array
            else:
                size = (self.number_agents, vector)  # 2-dimensional array (1-dimensional of vectors)

            # if using object types, make NoneType array, otherwise make array of zeros
            if dtype == str or dtype == object:
                self.__dict__[array_name] = np.empty(size, dtype=object)
            else:
                self.__dict__[array_name] = np.zeros(size, dtype=dtype)

        # only apply initial condition if not NoneType
        if func is not None:
            # get bounds for applying initial conditions to array
            if agent_type is None:
                begin = 0
                end = self.number_agents
            else:
                begin = self.agent_types[agent_type][0]
                end = self.agent_types[agent_type][1]

            # iterate through array applying function
            for i in range(begin, end):
                self.__dict__[array_name][i] = func()

    def agent_graph(self, graph_name):
        """ Adds graph to the simulation.

            - graph_name: the name of the instance variable made for the graph
        """
        # create instance variable for graph and add graph name to holder
        self.__dict__[graph_name] = Graph(self.number_agents)
        self.graph_names.append(graph_name)

    def assign_bins(self, max_agents, distance):
        """ Generalizes agent locations to a bins within lattice imposed on
            the agent space, used for accelerating neighbor searches.

            - max_agents: the current maximum number of agents that can fit
              into a bin
            - distance: the radius of search length
        """
        # run until all agents have been put into bins
        while True:
            # calculate the dimensions of the bins array and the bins helper array, include extra bins for agents that
            # may fall outside of the simulation space
            bins_help_size = np.ceil(self.size / distance).astype(int) + 2
            bins_size = np.append(bins_help_size, max_agents)

            # create the bins arrays
            bins_help = np.zeros(bins_help_size, dtype=int)  # holds the number of agents in each bin
            bins = np.zeros(bins_size, dtype=int)  # holds the indices of each agent in a bin

            # generalize the agent locations to bin indices and offset by 1 to prevent missing agents outside space
            bin_locations = np.floor_divide(self.locations, distance).astype(int) + 1

            # use JIT function from backend.py to speed up placement of agents
            bins, bins_help = assign_bins_jit(self.number_agents, bin_locations, bins, bins_help)

            # break the loop if all agents were accounted for or revalue the maximum number of agents based on and run
            # one more time
            current_max_agents = np.amax(bins_help)
            if max_agents >= current_max_agents:
                break
            else:
                max_agents = current_max_agents * 2  # double to prevent continual updating

        return bins, bins_help, bin_locations, max_agents

    @record_time
    def get_neighbors(self, graph_name, distance, clear=True):
        """ Finds all neighbors, within fixed radius, for each each agent.

            - graph_name: name of the instance variable pointing to the graph
            - distance: the radius of search length
            - clear: if removing existing edges, otherwise all edges are saved
        """
        # get graph object reference and if desired, remove all existing edges in the graph
        graph = self.__dict__[graph_name]
        if clear:
            graph.delete_edges(None)

        # assign each of the agents to bins, updating the max agents in a bin (if necessary)
        bins, bins_help, bin_locations, graph.max_agents = self.assign_bins(graph.max_agents, distance)

        # run until all edges are accounted for
        while True:
            # get the total amount of edges able to be stored and make the following arrays
            length = self.number_agents * graph.max_neighbors
            edge_holder = np.zeros((length, 2), dtype=int)         # hold all edges
            if_edge = np.zeros(length, dtype=bool)                 # say if each edge exists
            edge_count = np.zeros(self.number_agents, dtype=int)   # hold count of edges per agent

            # if using CUDA GPU
            if self.cuda:
                # allow the following arrays to be passed to the GPU
                edge_holder = cuda.to_device(edge_holder)
                if_edge = cuda.to_device(if_edge)
                edge_count = cuda.to_device(edge_count)

                # specify threads-per-block and blocks-per-grid values
                tpb = 72
                bpg = math.ceil(self.number_agents / tpb)

                # call the CUDA kernel, sending arrays to GPU
                get_neighbors_gpu[bpg, tpb](cuda.to_device(bin_locations), cuda.to_device(self.locations),
                                            cuda.to_device(bins), cuda.to_device(bins_help), cuda.to_device(distance),
                                            edge_holder, if_edge, edge_count, cuda.to_device(graph.max_neighbors))

                # return the following arrays back from the GPU
                edge_holder = edge_holder.copy_to_host()
                if_edge = if_edge.copy_to_host()
                edge_count = edge_count.copy_to_host()

            # otherwise use parallelized JIT function
            else:
                edge_holder, if_edge, edge_count = get_neighbors_cpu(self.number_agents, bin_locations, self.locations,
                                                                     bins, bins_help, distance, edge_holder, if_edge,
                                                                     edge_count, graph.max_neighbors)

            # break the loop if all neighbors were accounted for or revalue the maximum number of neighbors
            max_neighbors = np.amax(edge_count)
            if graph.max_neighbors >= max_neighbors:
                break
            else:
                graph.max_neighbors = max_neighbors * 2

        # reduce the edges to edges that actually exist and add those edges to graph
        edge_holder = edge_holder[if_edge]
        graph.add_edges(edge_holder)

        # simplify the graph's edges if not clearing the graph at the start
        if not clear:
            graph.simplify()

    @record_time
    def temp(self):
        """ Pickle the current state of the simulation which can be used
            to continue a past simulation without losing information.
        """
        # get file name and save in binary mode
        file_name = f"{self.name}_temp.pkl"
        with open(self.paths.main_path + file_name, "wb") as file:
            pickle.dump(self, file, -1)    # use the highest protocol -1 for pickling

    @record_time
    def step_values(self, arrays=None):
        """ Outputs a CSV file containing values from the agent arrays with each
            row corresponding to a particular agent index.

            - arrays: a list of agent array names to output, if None then all
              arrays are outputted
        """
        # only continue if outputting agent values
        if self.output_values:
            # if arrays is None automatically output all agent arrays
            if arrays is None:
                arrays = self.agent_array_names

            # make sure directory exists and get file name
            check_direct(self.paths.values)
            file_name = f"{self.name}_values_{self.current_step}.csv"

            # open the file
            with open(self.paths.values + file_name, "w", newline="") as file:
                # create CSV object and the following lists
                csv_file = csv.writer(file)
                header = list()    # header of the CSV (first row)
                data = list()    # holds the agent arrays

                # go through each of the agent arrays
                for array_name in arrays:
                    # get the agent array
                    agent_array = self.__dict__[array_name]

                    # if the array is one dimensional
                    if agent_array.ndim == 1:
                        header.append(array_name)    # add the array name to the header
                        agent_array = np.reshape(agent_array, (-1, 1))  # resize array from 1D to 2D
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
        """ Adds a new line to a running CSV holding data about the simulation
            such as memory, step time, number of agents and method profiling.
        """
        # get file name and open the file
        file_name = f"{self.name}_data.csv"
        with open(self.paths.main_path + file_name, "a", newline="") as file_object:
            # create CSV object
            csv_object = csv.writer(file_object)

            # create header if this is the beginning of a new simulation
            if self.current_step == 1:
                # get list of column names for non-method values and method values
                main_header = ["Step Number", "Number Cells", "Step Time", "Memory (MB)"]
                methods_header = list(self.method_times.keys())

                # merge the headers together and write the row to the CSV
                csv_object.writerow(main_header + methods_header)

            # calculate the total step time and get memory of process in megabytes
            step_time = time.perf_counter() - self.step_start
            process = psutil.Process(os.getpid())
            memory = process.memory_info()[0] / 1024 ** 2

            # write the row with the corresponding values
            columns = [self.current_step, self.number_agents, step_time, memory]
            function_times = list(self.method_times.values())
            csv_object.writerow(columns + function_times)

    def create_video(self):
        """ Write all of the step images from a simulation to a video file in the
            main simulation directory.
        """
        # continue if there is an image directory
        if os.path.isdir(self.paths.images):
            # get all of the images in the directory and count images
            file_list = [file for file in os.listdir(self.paths.images) if file.endswith(".png")]
            image_count = len(file_list)

            # only continue if image directory has images in it
            if image_count > 0:
                # print statement and sort the file list so "2, 20, 3, 31..." becomes "2, 3, ..., 20, ..., 31"
                print("\nCreating video...")
                file_list = sorted(file_list, key=sort_naturally)

                # sample the first image to get the dimensions of the image and then scale the image
                size = cv2.imread(self.paths.images + file_list[0]).shape[0:2]
                scale = self.video_quality / size[1]
                new_size = (self.video_quality, int(scale * size[0]))

                # get file name and create the video object
                file_name = f"{self.name}_video.mp4"
                codec = cv2.VideoWriter_fourcc(*"mp4v")
                video_object = cv2.VideoWriter(self.paths.main_path + file_name, codec, self.fps, new_size)

                # go through sorted image list, reading and writing each image to the video object
                for i in range(image_count):
                    image = cv2.imread(self.paths.images + file_list[i])    # read image from directory
                    if size != new_size:
                        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)    # scale down if necessary
                    video_object.write(image)    # write to video
                    progress_bar(i, image_count)    # show progress

                # close the video file
                video_object.release()

        # end statement
        print("\n\nDone!\n")

    def info(self):
        """ Records start time of the step for measuring efficiency and
            prints out info about the simulation.
        """
        # time.perf_counter() is more accurate than time.time()
        self.step_start = time.perf_counter()

        # current step and number of agents
        print("Step: " + str(self.current_step))
        print("Number of agents: " + str(self.number_agents))

    def random_vector(self):
        """ Computes a random vector on the unit sphere centered
            at the origin.
        """
        # random angle on the agent
        theta = r.random() * 2 * math.pi

        # if 2-dimensional set z=0
        if self.size[2] == 0:
            return np.array([math.cos(theta), math.sin(theta), 0])
        else:
            phi = r.random() * 2 * math.pi
            radius = math.cos(phi)
            return np.array([radius * math.cos(theta), radius * math.sin(theta), math.sin(phi)])
