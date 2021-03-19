import numpy as np
import random as r
import math
import time
import sys
import igraph

from abc import ABC, abstractmethod
from numba import jit, cuda, prange
from functools import wraps
from outputs import record_time


class Base(ABC):
    """ This abstract class is the base for the Simulation object. It's used to
        make sure that the Simulation object has certain attributes.
    """
    def __init__(self, paths, name):
        self.paths = paths    # the Paths object which holds any output paths
        self.name = name    # the name of the simulation

        # the running number of agents and the step to begin at (altered by continuation mode)
        self.number_agents = 0
        self.beginning_step = 1

        # hold the neighbors within a fixed radius of all cells with a graph
        self.neighbor_graph = igraph.Graph()
        self.graph_names = ["neighbor_graph"]

        # arrays to store the agents set to divide or to be removed
        self.agents_to_divide = np.array([], dtype=int)
        self.agents_to_remove = np.array([], dtype=int)

        # various other holders
        self.agent_array_names = list()  # store the variable names of each agent array
        self.agent_types = dict()  # hold the names of agent types defined in parameters.py
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
        """ Add cells to the Simulation object and potentially add a cell type
            with bounds for defining alternative initial parameters.

                number (int): the number of cells being added to the Simulation object
                cell_type (str): the name of a cell type that can be used by cell_array() to only apply
                    initial parameters to these cells, instead of the entire array.
        """
        # add specified number of cells to each graph
        for graph_name in self.graph_names:
            self.__dict__[graph_name].add_vertices(number)

        # update the running number of cells and determine bounds for slice if cell_type is used
        begin = self.number_agents
        self.number_agents += number

        # if a cell type name is passed, hold the slice bounds for that particular cell type
        if agent_type is not None:
            self.agent_types[agent_type] = (begin, self.number_agents)

    def agent_array(self, array_name, agent_type=None, dtype=float, vector=None, func=None, override=None):
        """ Create a cell array in the Simulation object used to hold values
            for all cells and optionally specify initial parameters.

                array_name (str): the name of the variable made for the cell array in the Simulation object
                cell_type (str): see add_cells()
                dtype (object): the data type of the array, defaults to float
                vector (int): the length of the vector for each cell in the array
                func (object): a function called for each index of the array to specify initial parameters
                override (array): use the array passed instead of generating a new array
        """
        # if using existing array for cell array
        if override is not None:
            # make sure array have correct length, otherwise raise error
            if override.shape[0] != self.number_agents:
                raise Exception("Length of override array does not match number of cells in simulation!")

            # use the array and add to list of cell array names
            else:
                self.__dict__[array_name] = override
                self.agent_array_names.append(array_name)

        # otherwise make sure a default cell array exists for initial parameters
        else:
            # if no cell array in Simulation object, make one
            if not hasattr(self, array_name):
                # add the array name to a list for automatic addition/removal when cells divide/die
                self.agent_array_names.append(array_name)

                # get the dimensions of the array
                if vector is None:
                    size = self.number_agents  # 1-dimensional array
                else:
                    size = (self.number_agents, vector)  # 2-dimensional array (1-dimensional of vectors)

                # if using python string data type, use object data type instead
                if dtype == str or dtype == object:
                    # create cell array in Simulation object with NoneType as default value
                    self.__dict__[array_name] = np.empty(size, dtype=object)

                else:
                    # create cell array in Simulation object, with zeros as default values
                    self.__dict__[array_name] = np.zeros(size, dtype=dtype)

        # if no cell type parameter passed
        if agent_type is None:
            # if function is passed, apply initial parameter
            if func is not None:
                for i in range(self.number_agents):
                    self.__dict__[array_name][i] = func()

        # otherwise a cell type is passed
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

        # prints the current step number and the number of cells
        print("Step: " + str(self.current_step))
        print("Number of cells: " + str(self.number_agents))

    def assign_bins(self, distance, max_cells):
        """ Generalizes cell locations to a bin within lattice imposed on
            the cell space, used for a parallel fixed-radius neighbor search.
        """
        # If there is enough space for all cells that should be in a bin, break out of the loop. If there isn't
        # update the amount of needed space and put all the cells in bins. This will run once if the prediction
        # of max neighbors suffices, twice if it isn't right the first time.
        while True:
            # calculate the size of the array used to represent the bins and the bins helper array, include extra bins
            # for cells that may fall outside of the space
            bins_help_size = np.ceil(self.size / distance).astype(int) + 3
            bins_size = np.append(bins_help_size, max_cells)

            # create the arrays for "bins" and "bins_help"
            bins_help = np.zeros(bins_help_size, dtype=int)  # holds the number of cells currently in a bin
            bins = np.empty(bins_size, dtype=int)  # holds the indices of cells in a bin

            # generalize the cell locations to bin indices and offset by 1 to prevent missing cells that fall out of the
            # self space
            bin_locations = np.floor_divide(self.locations, distance).astype(int)
            bin_locations += 1

            # use jit function to speed up placement of cells
            bins, bins_help = assign_bins_jit(self.number_agents, bin_locations, bins, bins_help)

            # either break the loop if all cells were accounted for or revalue the maximum number of cells based on
            # the output of the function call and double it future calls
            new_max_cells = np.amax(bins_help)
            if max_cells >= new_max_cells:
                break
            else:
                max_cells = new_max_cells * 2  # double to prevent continual updating

        return bins, bins_help, bin_locations, max_cells

    @record_time
    def get_neighbors(self, distance=0.00002):
        """ For all cells, determines which cells fall within a fixed radius to
            denote a neighbor then stores this information in a graph (uses a bin/
            bucket sorting method).
        """
        # if a static variable has not been created to hold the maximum number of neighbors for a cell, create one
        if not hasattr(Base.get_neighbors, "max_neighbors"):
            # begin with a low number of neighbors that can be revalued if the max neighbors exceeds this value
            Base.get_neighbors.max_neighbors = 5

        # if a static variable has not been created to hold the maximum number of cells in a bin, create one
        if not hasattr(Base.get_neighbors, "max_cells"):
            # begin with a low number of cells that can be revalued if the max number of cells exceeds this value
            Base.get_neighbors.max_cells = 5

        # clear all of the edges in the neighbor graph
        self.neighbor_graph.delete_edges(None)

        # calls the function that generates an array of bins that generalize the cell locations in addition to a
        # creating a helper array that assists the search method in counting cells for a particular bin
        bins, bins_help, bin_locations, max_cells = self.assign_bins(distance, Base.get_neighbors.max_cells)

        # update the value of the max number of cells in a bin
        Base.get_neighbors.max_cells = max_cells

        # this will run once if all edges are included in edge_holder, breaking the loop. if not, this will
        # run a second time with an updated value for the number of predicted neighbors such that all edges are included
        while True:
            # create array used to hold edges, array to say if edge exists, and array to count the edges per cell
            length = self.number_agents * Base.get_neighbors.max_neighbors
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
                max_neighbors = cuda.to_device(Base.get_neighbors.max_neighbors)

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
                                                                     edge_count, Base.get_neighbors.max_neighbors)

            # either break the loop if all neighbors were accounted for or revalue the maximum number of neighbors
            # based on the output of the function call and double it for future calls
            max_neighbors = np.amax(edge_count)
            if Base.get_neighbors.max_neighbors >= max_neighbors:
                break
            else:
                Base.get_neighbors.max_neighbors = max_neighbors * 2

        # reduce the edges to only edges that actually exist
        edge_holder = edge_holder[if_edge]

        # add the edges to the neighbor graph
        self.neighbor_graph.add_edges(edge_holder)

    def random_vector(self):
        """ Computes a random vector on the unit sphere centered
            at the origin.
        """
        # random angle on the cell
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
        file_name = f"{self.name}_temp.pkl"

        # open the file in binary mode
        with open(self.paths.main + file_name, "wb") as file:
            # use the highest protocol: -1 for pickling the instance
            pickle.dump(self, file, -1)

    @record_time
    def step_values(self, arrays=None):
        """ Outputs a CSV file with value from the cell arrays with each
            row corresponding to a particular cell index.

            arrays -> (list) If arrays is None, all cell arrays are outputted otherwise only the
                cell arrays named in the list will be outputted.
        """
        # only continue if outputting cell values
        if self.output_values:
            # if arrays is None automatically output all cell arrays
            if arrays is None:
                agent_array_names = self.agent_array_names

            # otherwise only output arrays specified in list
            else:
                agent_array_names = arrays

            # get path and make sure directory exists
            directory_path = check_direct(self.paths.values)

            # get file name, use f-string
            file_name = f"{self.name}_values_{self.current_step}.csv"

            # open the file
            with open(directory_path + file_name, "w", newline="") as file:
                # create CSV object and the following lists
                csv_file = csv.writer(file)
                header = list()    # header of the CSV (first row)
                data = list()    # holds the cell arrays

                # go through each of the cell arrays
                for array_name in agent_array_names:
                    # get the cell array
                    cell_array = self.__dict__[array_name]

                    # if the array is one dimensional
                    if cell_array.ndim == 1:
                        header.append(array_name)    # add the array name to the header
                        cell_array = np.reshape(cell_array, (-1, 1))  # resize array from 1D to 2D

                    # if the array is not one dimensional
                    else:
                        # create name for column based on slice of array ex. locations[0], locations[1], locations[2]
                        for i in range(cell_array.shape[1]):
                            header.append(array_name + "[" + str(i) + "]")

                    # add the array to the data holder
                    data.append(cell_array)

                # write header as the first row of the CSV
                csv_file.writerow(header)

                # stack the arrays to create rows for the CSV file and save to CSV
                data = np.hstack(data)
                csv_file.writerows(data)

    def data(self):
        """ Creates/adds a new line to the running CSV for data about the simulation
            such as memory, step time, number of cells and method profiling.
        """
        # get file name, use f-string
        file_name = f"{self.name}_data.csv"

        # open the file
        with open(self.paths.main + file_name, "a", newline="") as file_object:
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
                file_name = f"{self.name}_video.mp4"
                video_path = self.paths.main + file_name

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


class Paths:
    """ Hold any important paths for a particular simulation. For a continued
        simulation, this will update the Paths object in case the path(s) change.
    """
    def __init__(self, name, main, templates, separator):
        self.main = main    # the path to the main directory for this simulation
        self.templates = templates    # the path to the .txt template directory
        self.separator = separator    # file separator

        # these directories are sub-directories under the main simulation directory
        general = main + name
        self.images = general + "_images" + separator    # the images output directory
        self.values = general + "_values" + separator    # the cell array values output directory
        self.gradients = general + "_gradients" + separator    # the gradients output directory
        self.tda = general + "_tda" + separator    # the topological data analysis output directory


@jit(nopython=True, cache=True)
def assign_bins_jit(number_agents, bin_locations, bins, bins_help):
    """ A just-in-time compiled function for assign_bins() that places
        the cells in their respective bins.
    """
    # go through all cells
    for index in range(number_agents):
        # get the indices of the generalized cell location
        x, y, z = bin_locations[index]

        # use the help array to get the new index for the cell in the bin
        place = bins_help[x][y][z]

        # adds the index in the cell array to the bin
        bins[x][y][z][place] = index

        # update the number of cells in a bin
        bins_help[x][y][z] += 1

    # return the arrays now filled with cell indices
    return bins, bins_help


@cuda.jit
def get_neighbors_gpu(bin_locations, locations, bins, bins_help, distance, edge_holder, if_edge, edge_count,
                      max_neighbors):
    """ A just-in-time compiled cuda kernel for the get_neighbors()
        method that performs the actual calculations.
    """
    # get the index in the array
    focus = cuda.grid(1)

    # get the starting index for writing to the edge holder array
    start = focus * max_neighbors[0]

    # double check that focus index is within the array
    if focus < bin_locations.shape[0]:
        # holds the total amount of edges for a given cell
        cell_edge_count = 0

        # get the bin location of the cell
        x, y, z = bin_locations[focus]

        # go through the surrounding bins including the bin the cell is in
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of cells for the current bin
                    bin_count = bins_help[x + i][y + j][z + k]

                    # go through the bin determining if a cell is a neighbor
                    for l in range(bin_count):
                        # get the index of the current potential neighbor
                        current = bins[x + i][y + j][z + k][l]

                        # check to see if that cell is within the search radius and only continue if the current cell
                        # has a higher index to prevent double counting edges
                        if magnitude(locations[focus], locations[current]) <= distance[0] and focus < current:
                            # if less than the max edges, add the edge
                            if cell_edge_count < max_neighbors[0]:
                                # get the index for the edge
                                index = start + cell_edge_count

                                # update the edge array and identify that this edge exists
                                edge_holder[index][0] = focus
                                edge_holder[index][1] = current
                                if_edge[index] = 1

                            # increase the count of edges for a cell and the index for the next edge
                            cell_edge_count += 1

        # update the array with number of edges for the cell
        edge_count[focus] = cell_edge_count


@jit(nopython=True, parallel=True, cache=True)
def get_neighbors_cpu(number_agents, bin_locations, locations, bins, bins_help, distance, edge_holder, if_edge,
                      edge_count, max_neighbors):
    """ A just-in-time compiled function for the get_neighbors()
        method that performs the actual calculations.
    """
    # loops over all cells, with the current cell index being the focus
    for focus in prange(number_agents):
        # get the starting index for writing to the edge holder array
        start = focus * max_neighbors

        # holds the total amount of edges for a given cell
        cell_edge_count = 0

        # get the bin location of the cell
        x, y, z = bin_locations[focus]

        # go through the surrounding bins including the bin the cell is in
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of cells for the current bin
                    bin_count = bins_help[x + i][y + j][z + k]

                    # go through the bin determining if a cell is a neighbor
                    for l in range(bin_count):
                        # get the index of the current potential neighbor
                        current = bins[x + i][y + j][z + k][l]

                        # check to see if that cell is within the search radius and only continue if the current cell
                        # has a higher index to prevent double counting edges
                        if np.linalg.norm(locations[current] - locations[focus]) <= distance and focus < current:
                            # if less than the max edges, add the edge
                            if cell_edge_count < max_neighbors:
                                # get the index for the edge
                                index = start + cell_edge_count

                                # update the edge array and identify that this edge exists
                                edge_holder[index][0] = focus
                                edge_holder[index][1] = current
                                if_edge[index] = 1

                            # increase the count of edges for a cell and the index for the next edge
                            cell_edge_count += 1

        # update the array with number of edges for the cell
        edge_count[focus] = cell_edge_count

    return edge_holder, if_edge, edge_count


@cuda.jit
def jkr_neighbors_gpu(bin_locations, locations, radii, bins, bins_help, edge_holder, if_edge, edge_count,
                      max_neighbors):
    """ A just-in-time compiled cuda kernel for the jkr_neighbors()
        method that performs the actual calculations.
    """
    # get the index in the array
    focus = cuda.grid(1)

    # get the starting index for writing to the edge holder array
    start = focus * max_neighbors[0]

    # double check that focus index is within the array
    if focus < locations.shape[0]:
        # holds the total amount of edges for a given cell
        cell_edge_count = 0

        # get the bin location of the cell
        x, y, z = bin_locations[focus]

        # go through the surrounding bins including the bin the cell is in
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of cells for the current bin
                    bin_count = bins_help[x + i][y + j][z + k]

                    # go through the bin determining if a cell is a neighbor
                    for l in range(bin_count):
                        # get the index of the current potential neighbor
                        current = bins[x + i][y + j][z + k][l]

                        # get the magnitude of the distance vector between the cell locations
                        mag = magnitude(locations[focus], locations[current])

                        # calculate the overlap of the cells
                        overlap = radii[focus] + radii[current] - mag

                        # if there is 0 or more overlap and if the current cell has a higher index to prevent double
                        # counting edges
                        if overlap >= 0 and focus < current:
                            # if less than the max edges, add the edge
                            if cell_edge_count < max_neighbors[0]:
                                # get the index for the edge
                                index = start + cell_edge_count

                                # update the edge array and identify that this edge exists
                                edge_holder[index][0] = focus
                                edge_holder[index][1] = current
                                if_edge[index] = 1

                            # increase the count of edges for a cell and the index for the next edge
                            cell_edge_count += 1

        # update the array with number of edges for the cell
        edge_count[focus] = cell_edge_count


@jit(nopython=True, parallel=True, cache=True)
def jkr_neighbors_cpu(number_agents, bin_locations, locations, radii, bins, bins_help, edge_holder,
                      if_edge, edge_count, max_neighbors):
    """ A just-in-time compiled function for the jkr_neighbors()
        method that performs the actual calculations.
    """
    # loops over all cells, with the current cell index being the focus
    for focus in prange(number_agents):
        # get the starting index for writing to the edge holder array
        start = focus * max_neighbors

        # holds the total amount of edges for a given cell
        cell_edge_count = 0

        # get the bin location of the cell
        x, y, z = bin_locations[focus]

        # go through the surrounding bins including the bin the cell is in
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of cells for the current bin
                    bin_count = bins_help[x + i][y + j][z + k]

                    # go through the bin determining if a cell is a neighbor
                    for l in range(bin_count):
                        # get the index of the current potential neighbor
                        current = bins[x + i][y + j][z + k][l]

                        # get the magnitude of the distance vector between the cell locations
                        mag = np.linalg.norm(locations[current] - locations[focus])

                        # calculate the overlap of the cells
                        overlap = radii[current] + radii[focus] - mag

                        # if there is 0 or more overlap and if the current cell has a higher index to prevent double
                        # counting edges
                        if overlap >= 0 and focus < current:
                            # if less than the max edges, add the edge
                            if cell_edge_count < max_neighbors:
                                # get the index for the edge
                                index = start + cell_edge_count

                                # update the edge array and identify that this edge exists
                                edge_holder[index][0] = focus
                                edge_holder[index][1] = current
                                if_edge[index] = 1

                            # increase the count of edges for a cell and the index for the next edge
                            cell_edge_count += 1

        # update the array with number of edges for the cell
        edge_count[focus] = cell_edge_count

    return edge_holder, if_edge, edge_count


@cuda.jit
def jkr_forces_gpu(jkr_edges, delete_edges, locations, radii, jkr_forces, poisson, youngs, adhesion_const):
    """ A just-in-time compiled cuda kernel for the jkr_forces()
        method that performs the actual calculations.
    """
    # get the index in the edges array
    edge_index = cuda.grid(1)

    # double check that index is within the array
    if edge_index < jkr_edges.shape[0]:
        # get the cell indices of the edge
        cell_1 = jkr_edges[edge_index][0]
        cell_2 = jkr_edges[edge_index][1]

        # get the locations of the two cells
        location_1 = locations[cell_1]
        location_2 = locations[cell_2]

        # get the magnitude of the distance between the cells
        mag = magnitude(location_1, location_2)

        # get the overlap of the cells
        overlap = radii[cell_1] + radii[cell_2] - mag

        # get two values used for JKR calculation
        e_hat = (((1 - poisson[0] ** 2) / youngs[0]) + ((1 - poisson[0] ** 2) / youngs[0])) ** -1
        r_hat = ((1 / radii[cell_1]) + (1 / radii[cell_2])) ** -1

        # value used to calculate the max adhesive distance after bond has been already formed
        overlap_ = (((math.pi * adhesion_const[0]) / e_hat) ** (2 / 3)) * (r_hat ** (1 / 3))

        # get the nondimensionalized overlap
        d = overlap / overlap_

        # check to see if the cells will have a force interaction based on the nondimensionalized distance
        if d > -0.360562:
            # plug the value of d into polynomial approximation for nondimensionalized force
            f = (-0.0204 * d ** 3) + (0.4942 * d ** 2) + (1.0801 * d) - 1.324

            # convert from the nondimensionalized force to find the JKR force
            jkr_force = f * math.pi * adhesion_const[0] * r_hat

            # loops over all directions of space
            for i in range(3):
                # get the vector by axis between the two cells
                vector = location_1[i] - location_2[i]

                # if the magnitude is 0 use the zero vector, otherwise find the normalized vector for each axis
                if mag != 0:
                    normal = vector / mag
                else:
                    normal = 0

                # adds the adhesive force as a vector in opposite directions to each cell's force holder
                jkr_forces[cell_1][i] += jkr_force * normal
                jkr_forces[cell_2][i] -= jkr_force * normal

        # remove the edge if the it fails to meet the criteria for distance, simulating that the bond is broken
        else:
            delete_edges[edge_index] = 1


@jit(nopython=True, parallel=True, cache=True)
def jkr_forces_cpu(number_edges, jkr_edges, delete_edges, locations, radii, jkr_forces, poisson, youngs,
                   adhesion_const):
    """ A just-in-time compiled function for the jkr_forces()
        method that performs the actual calculations.
    """
    # go through the edges array
    for edge_index in prange(number_edges):
        # get the cell indices of the edge
        cell_1 = jkr_edges[edge_index][0]
        cell_2 = jkr_edges[edge_index][1]

        # get the vector between the centers of the cells and the magnitude of this vector
        vector = locations[cell_1] - locations[cell_2]
        mag = np.linalg.norm(vector)

        # get the overlap of the cells
        overlap = radii[cell_1] + radii[cell_2] - mag

        # get two values used for JKR calculation
        e_hat = (((1 - poisson ** 2) / youngs) + ((1 - poisson ** 2) / youngs)) ** -1
        r_hat = ((1 / radii[cell_1]) + (1 / radii[cell_2])) ** -1

        # value used to calculate the max adhesive distance after bond has been already formed
        overlap_ = (((math.pi * adhesion_const) / e_hat) ** (2 / 3)) * (r_hat ** (1 / 3))

        # get the nondimensionalized overlap
        d = overlap / overlap_

        # check to see if the cells will have a force interaction based on the nondimensionalized distance
        if d > -0.360562:
            # plug the value of d into polynomial approximation for nondimensionalized force
            f = (-0.0204 * d ** 3) + (0.4942 * d ** 2) + (1.0801 * d) - 1.324

            # convert from the nondimensionalized force to find the JKR force
            jkr_force = f * math.pi * adhesion_const * r_hat

            # if the magnitude is 0 use the zero vector, otherwise find the normalized vector for each axis. numba's
            # jit prefers a reduction instead of generating a new normalized array
            normal = np.array([0.0, 0.0, 0.0])
            if mag != 0:
                normal += vector / mag

            # adds the adhesive force as a vector in opposite directions to each cell's force holder
            jkr_forces[cell_1] += jkr_force * normal
            jkr_forces[cell_2] -= jkr_force * normal

        # remove the edge if the it fails to meet the criteria for distance, simulating that the bond is broken
        else:
            delete_edges[edge_index] = 1

    return jkr_forces, delete_edges


@cuda.jit
def apply_forces_gpu(jkr_force, motility_force, locations, radii, viscosity, size, move_dt):
    """ A just-in-time compiled cuda kernel for the apply_forces()
        method that performs the actual calculations.
    """
    # get the index in the array
    index = cuda.grid(1)

    # double check that index is within the array
    if index < locations.shape[0]:
        # stokes law for velocity based on force and fluid viscosity (friction)
        stokes_friction = 6 * math.pi * viscosity[0] * radii[index]

        # loop over all directions of space
        for i in range(3):
            # update the velocity of the cell based on stokes
            velocity = (jkr_force[index][i] + motility_force[index][i]) / stokes_friction

            # set the new location
            new_location = locations[index][i] + velocity * move_dt[0]

            # check if new location is in the simulation space, if not set values at space limits
            if new_location > size[i]:
                locations[index][i] = size[i]
            elif new_location < 0:
                locations[index][i] = 0
            else:
                locations[index][i] = new_location


@jit(nopython=True, parallel=True, cache=True)
def apply_forces_cpu(number_agents, jkr_force, motility_force, locations, radii, viscosity, size, move_dt):
    """ A just-in-time compiled function for the apply_forces()
        method that performs the actual calculations.
    """
    # loop over all cells
    for i in prange(number_agents):
        # stokes law for velocity based on force and fluid viscosity (friction)
        stokes_friction = 6 * math.pi * viscosity * radii[i]

        # update the velocity of the cell based on stokes
        velocity = (motility_force[i] + jkr_force[i]) / stokes_friction

        # set the new location
        new_location = locations[i] + velocity * move_dt

        # loop over all directions of space
        for j in range(0, 3):
            # check if new location is in the space, if not return it to the space limits
            if new_location[j] > size[j]:
                locations[i][j] = size[j]
            elif new_location[j] < 0:
                locations[i][j] = 0
            else:
                locations[i][j] = new_location[j]

    return locations


@cuda.jit
def nearest_gpu(bin_locations, locations, bins, bins_help, distance, if_diff, gata6, nanog, nearest_gata6,
                nearest_nanog, nearest_diff):
    """ A just-in-time compiled cuda kernel for the nearest()
        method that performs the actual calculations.
    """
    # get the index in the array
    focus = cuda.grid(1)

    # double check that the index is within the array
    if focus < locations.shape[0]:
        # get the bin location of the cell
        x, y, z = bin_locations[focus]

        # initialize the nearest indices with -1 which will be interpreted as no cell by the motility function
        nearest_gata6_index, nearest_nanog_index, nearest_diff_index = -1, -1, -1

        # initialize the distance for each with double the search radius to provide a starting point
        nearest_gata6_dist, nearest_nanog_dist, nearest_diff_dist = distance[0] * 2, distance[0] * 2, distance[0] * 2

        # go through the surrounding bins including the bin the cell is in
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of cells for the current bin
                    bin_count = bins_help[x + i][y + j][z + k]

                    # go through the bin
                    for l in range(bin_count):
                        # get the index of the current potential nearest cell
                        current = bins[x + i][y + j][z + k][l]

                        # get the magnitude of the distance vector between the cells
                        mag = magnitude(locations[focus], locations[current])

                        # check to see if the current cell is within the search radius and not the same cell
                        if mag <= distance[0] and focus != current:
                            # if the current cell is differentiated
                            if if_diff[current]:
                                # if it's closer than the last cell, update the distance and index
                                if mag < nearest_diff_dist:
                                    nearest_diff_index = current
                                    nearest_diff_dist = mag

                            # if the current cell is gata6 high
                            elif gata6[current] > nanog[current]:
                                # if it's closer than the last cell, update the distance and index
                                if mag < nearest_gata6_dist:
                                    nearest_gata6_index = current
                                    nearest_gata6_dist = mag

                            # if the current cell is nanog high
                            elif gata6[current] < nanog[current]:
                                # if it's closer than the last cell, update the distance and index
                                if mag < nearest_nanog_dist:
                                    nearest_nanog_index = current
                                    nearest_nanog_dist = mag

        # update the arrays
        nearest_gata6[focus] = nearest_gata6_index
        nearest_nanog[focus] = nearest_nanog_index
        nearest_diff[focus] = nearest_diff_index


@jit(nopython=True, parallel=True, cache=True)
def nearest_cpu(number_agents, bin_locations, locations, bins, bins_help, distance, if_diff, gata6, nanog,
                nearest_gata6, nearest_nanog, nearest_diff):
    """ A just-in-time compiled function for the nearest()
        method that performs the actual calculations.
    """
    # loop over all cells
    for focus in prange(number_agents):
        # get the bin location of the cell
        x, y, z = bin_locations[focus]

        # initialize the nearest indices with -1 which will be interpreted as no cell by the motility function
        nearest_gata6_index, nearest_nanog_index, nearest_diff_index = -1, -1, -1

        # initialize the distance for each with double the search radius to provide a starting point
        nearest_gata6_dist, nearest_nanog_dist, nearest_diff_dist = distance * 2, distance * 2, distance * 2

        # go through the surrounding bins including the bin the cell is in
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of cells for the current bin
                    bin_count = bins_help[x + i][y + j][z + k]

                    # go through the bin
                    for l in range(bin_count):
                        # get the index of the current potential nearest cell
                        current = bins[x + i][y + j][z + k][l]

                        # get the magnitude of the distance vector between the cells
                        mag = np.linalg.norm(locations[current] - locations[focus])

                        # check to see if the current cell is within the search radius and not the same cell
                        if mag <= distance and focus != current:
                            # if the current cell is differentiated
                            if if_diff[current]:
                                # if it's closer than the last cell, update the distance and index
                                if mag < nearest_diff_dist:
                                    nearest_diff_index = current
                                    nearest_diff_dist = mag

                            # if the current cell is gata6 high
                            elif gata6[current] > nanog[current]:
                                # if it's closer than the last cell, update the distance and index
                                if mag < nearest_gata6_dist:
                                    nearest_gata6_index = current
                                    nearest_gata6_dist = mag

                            # if the current cell is nanog high
                            elif gata6[current] < nanog[current]:
                                # if it's closer than the last cell, update the distance and index
                                if mag < nearest_nanog_dist:
                                    nearest_nanog_index = current
                                    nearest_nanog_dist = mag

        # update the arrays
        nearest_gata6[focus] = nearest_gata6_index
        nearest_nanog[focus] = nearest_nanog_index
        nearest_diff[focus] = nearest_diff_index

    return nearest_gata6, nearest_nanog, nearest_diff


@jit(nopython=True, cache=True)
def update_diffusion_jit(base, steps, diffuse_dt, last_dt, diffuse_const, spat_res2):
    """ A just-in-time compiled function for update_diffusion()
        that performs the actual diffusion calculation.
    """
    # holder the following constant for faster computation, given that dx and dy match
    a = diffuse_dt * diffuse_const / spat_res2
    b = 1 - 4 * a

    # finite difference to solve laplacian diffusion equation, currently 2D
    for i in range(steps):
        # on the last step apply smaller diffuse dt if step dt doesn't divide nicely
        if i == steps - 1:
            a = last_dt * diffuse_const / spat_res2
            b = 1 - 4 * a

        # set the initial conditions by reflecting the edges of the gradient
        base[:, 0] = base[:, 1]
        base[:, -1] = base[:, -2]
        base[0, :] = base[1, :]
        base[-1, :] = base[-2, :]

        # get the morphogen addition for the diffusion points, based on other points and hold this
        temp = a * (base[2:, 1:-1] + base[:-2, 1:-1] + base[1:-1, 2:] + base[1:-1, :-2])

        # get the diffusion loss for the diffusion points
        base[1:-1, 1:-1] *= b

        # add morphogen change from the temporary array
        base[1:-1, 1:-1] += temp

    # return the gradient back without the edges
    return base[1:-1, 1:-1]


@cuda.jit(device=True)
def magnitude(vector_1, vector_2):
    """ A just-in-time compiled cuda kernel device function
        for getting the distance between two vectors.
    """
    # loop over the axes add the squared difference
    total = 0
    for i in range(0, 3):
        total += (vector_1[i] - vector_2[i]) ** 2

    # return the sqrt of the total
    return total ** 0.5


def normal_vector(vector):
    """ Returns the normalized vector, sadly this does not
        exist in NumPy.
    """
    # get the magnitude of the vector
    mag = np.linalg.norm(vector)

    # if magnitude is 0 return zero vector, otherwise divide by the magnitude
    if mag == 0:
        return np.zeros(3)
    else:
        return vector / mag


def commandline_param(flag, dtype):
    """ Returns the value for option passed at the
        command line.
    """
    # get list of command line arguments
    args = sys.argv

    # go through the arguments
    for i in range(len(args)):
        # if argument matches flag
        if args[i] == flag:
            # try to return value of
            try:
                return dtype(args[i + 1])
            # otherwise raise error
            except IndexError:
                raise Exception(f"No value for option: {args[i]}")

    # return NoneType if no value passed
    return None


def template_param(path, line_number, dtype):
    """ Gets the parameter as a string from the lines of the
        template file. Used for Simulation instance variables.
    """
    # make an attribute with name as template file path and value as a list of the file lines (reduces file opening)
    if not hasattr(template_param, path):
        with open(path) as file:
            template_param.path = file.readlines()

    # get the right line based on the line numbers not Python indexing
    line = template_param.path[line_number - 1]

    # find the indices of the pipe characters
    begin = line.find("|")
    end = line.find("|", begin + 1)

    # raise error if not a pair of pipe characters
    if begin == -1 or end == -1:
        raise Exception("Please use pipe characters to specify template file parameters. Example: | (value) |")

    # return a slice of the line that is the string representation of the parameter and remove any whitespace
    parameter = line[(begin + 1):end].strip()

    # convert the parameter from string to desired data type
    if dtype == str:
        pass
    elif dtype == tuple or dtype == list or dtype == dict:
        # tuple() list() dict() will not produce desired result, use eval() instead
        parameter = eval(parameter)
    elif dtype == bool:
        # handle potential inputs for booleans
        if parameter in ["True", "true", "T", "t", "1"]:
            parameter = True
        elif parameter in ["False", "false", "F", "f", "0"]:
            parameter = False
        else:
            raise Exception("Invalid value for bool type")
    else:
        # for float and int type
        parameter = dtype(parameter)

    # get the parameter by removing the pipe characters and any whitespace
    return parameter
