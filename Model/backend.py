import numpy as np
import time
import os
import sys
import yaml
import shutil
import re
import igraph
from numba import jit, cuda, prange
from functools import wraps


class Graph(igraph.Graph):
    """ This class extends the Graph class from iGraph adding
        instance variables for the bin/bucket sort algorithm.
    """
    def __init__(self, *args, **kwargs):
        # call the origin constructor from iGraph
        super().__init__(*args, **kwargs)

        # these variables are used in the bin/bucket sort for finding neighbors (values change frequently)
        self.max_neighbors = 1    # the current number of neighbors that can be stored in a holder array
        self.max_agents = 1    # the current number of agents that can be stored in a bin

    def num_neighbors(self, index):
        """ Returns the number of neighbors for the index.
        """
        return len(self.neighbors(index))


@jit(nopython=True, cache=True)
def assign_bins_jit(number_agents, bin_locations, bins, bins_help, max_agents):
    """ This just-in-time compiled method performs the actual
        calculations for the assign_bins() method.
    """
    for index in range(number_agents):
        # get the indices of the bin location
        x, y, z = bin_locations[index]

        # get the place in the bin to put the agent index
        place = bins_help[x][y][z]

        # if there is room in the bin, place the agent's index
        if place < max_agents:
            bins[x][y][z][place] = index

        # update the number of agents that should be in a bin (regardless of if they're placed there)
        bins_help[x][y][z] += 1

    return bins, bins_help


@cuda.jit(device=True)
def magnitude(vector_1, vector_2):
    """ This just-in-time compiled CUDA kernel is a device
        function for calculating the distance between vectors.
    """
    total = 0
    for i in range(0, 3):
        total += (vector_1[i] - vector_2[i]) ** 2
    return total ** 0.5


@cuda.jit
def get_neighbors_gpu(locations, bin_locations, bins, bins_help, distance, edges, if_edge, edge_count, max_neighbors):
    """ This just-in-time compiled CUDA kernel performs the actual
        calculations for the get_neighbors() method.
    """
    # get the agent index in the array
    index = cuda.grid(1)

    # double check that the index is within bounds
    if focus < bin_locations.shape[0]:
        # get the starting index for writing edges to the holder array
        start = index * max_neighbors[0]

        # hold the total amount of edges for the agent
        agent_edge_count = 0

        # get the indices of the bin location
        x, y, z = bin_locations[focus]

        # go through the 9 bins that could all potential neighbors
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of agents for the current bin
                    bin_count = bins_help[x + i][y + j][z + k]

                    # go through the current bin determining if an agent is a neighbor
                    for l in range(bin_count):
                        # get the index of the current potential neighbor
                        current = bins[x + i][y + j][z + k][l]

                        # check to see if the agent is a neighbor and prevent duplicates with index condition
                        if magnitude(locations[focus], locations[current]) <= distance[0] and index < current:
                            # if there is room, add the edge
                            if agent_edge_count < max_neighbors[0]:
                                # get the index for the edge
                                edge_index = start + agent_edge_count

                                # update the edge array and identify that this edge exists
                                edges[edge_index][0] = index
                                edges[edge_index][1] = current
                                if_edge[index] = 1

                            # increase the count of edges for an agent
                            agent_edge_count += 1

        # update the array with number of edges for the agent
        edge_count[focus] = agent_edge_count


@jit(nopython=True, parallel=True, cache=True)
def get_neighbors_cpu(number_agents, locations, bin_locations, bins, bins_help, distance, edges, if_edge, edge_count,
                      max_neighbors):
    """ This just-in-time compiled method performs the actual
        calculations for the get_neighbors() method.
    """
    for index in prange(number_agents):
        # get the starting index for writing edges to the holder array
        start = focus * max_neighbors

        # hold the total amount of edges for the agent
        agent_edge_count = 0

        # get the indices of the bin location
        x, y, z = bin_locations[focus]

        # go through the 9 bins that could all potential neighbors
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of agents for the current bin
                    bin_count = bins_help[x + i][y + j][z + k]

                    # go through the current bin determining if an agent is a neighbor
                    for l in range(bin_count):
                        # get the index of the current potential neighbor
                        current = bins[x + i][y + j][z + k][l]

                        # check to see if the agent is a neighbor and prevent duplicates with index condition
                        if np.linalg.norm(locations[current] - locations[focus]) <= distance and focus < current:
                            # if there is room, add the edge
                            if agent_edge_count < max_neighbors[0]:
                                # get the index for the edge
                                edge_index = start + agent_edge_count

                                # update the edge array and identify that this edge exists
                                edges[edge_index][0] = index
                                edges[edge_index][1] = current
                                if_edge[index] = 1

                            # increase the count of edges for an agent
                            agent_edge_count += 1

        # update the array with number of edges for the agent
        edge_count[focus] = agent_edge_count

    return edges, if_edge, edge_count


def check_direct(path):
    """ Makes sure directory exists.
    """
    if not os.path.isdir(path):
        os.mkdir(path)


def progress_bar(progress, maximum):
    """ Makes a progress bar to show progress of output.
    """
    # length of the bar
    length = 60

    # calculate bar and percent
    progress += 1    # start at 1 not 0
    fill = int(length * progress / maximum)
    bar = '#' * fill + '.' * (length - fill)
    percent = int(100 * progress / maximum)

    # output the progress bar
    print(f"\r[{bar}] {percent}%", end="")


def normal_vector(vector):
    """ Normalizes the vector.
    """
    # get the magnitude of the vector
    mag = np.linalg.norm(vector)

    # if magnitude is 0 return zero vector, otherwise divide by the magnitude
    if mag == 0:
        return np.zeros(3)
    else:
        return vector / mag


def record_time(function):
    """ This is a decorator used to time individual methods.
    """
    @wraps(function)
    def wrap(simulation, *args, **kwargs):    # args and kwargs are for additional arguments
        # call the method and get the start/end time
        start = time.perf_counter()
        function(simulation, *args, **kwargs)
        end = time.perf_counter()

        # add the method time to the dictionary holding these times
        simulation.method_times[function.__name__] = end - start

    return wrap


# ---------------------------------------- helper methods for user-interface ------------------------------------------
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


def template_params(path):
    """ Return parameters as dict from YAML template file.
    """
    with open(path, "r") as file:
        return yaml.safe_load(file)


def output_dir():
    """ Read the output path from paths.yaml and if this directory
        does not exist yet, make it.
    """
    # get file separator
    separator = os.path.sep

    # open the file and load the keys
    with open("paths.yaml", "r") as file:
        keys = yaml.safe_load(file)

    # get output_path key
    output_path = keys["output_path"]

    # keep running until output directory exists
    while not os.path.isdir(output_path):
        # prompt user input
        print("\nSimulation output directory: \"" + output_path + "\" does not exist!")
        user = input("Do you want to make this directory? If \"n\", you can specify the correct path (y/n): ")
        print()

        # if not making this directory
        if user == "n":
            # get new path to output directory
            output_path = input("Correct path (absolute) to output directory: ")

            # update paths.yaml file with new output directory path
            keys["output_path"] = output_path
            with open("paths.yaml", "w") as file:
                keys = yaml.dump(keys, file)

        # if yes, make the directory
        elif user == "y":
            os.makedirs(output_path)
            break

        else:
            print("Either type \"y\" or \"n\"")

    # if path doesn't end with separator, add it
    if output_path[-1] != separator:
        output_path += separator

    # return correct path to output directory
    return output_path


def get_name_mode():
    """ This function will get the name and mode for the simulation
        either from the command line or a text-based UI.
    """
    # try to get the name and mode from the command line
    name = commandline_param("-n", str)
    mode = commandline_param("-m", int)

    # if the name variable has not been initialized by the command-line, run the text-based UI to get it
    if name is None:
        while True:
            # prompt for the name
            name = input("What is the \"name\" of the simulation? Type \"help\" for more information: ")

            # keep running if "help" is typed
            if name == "help":
                print("\nType the name of the simulation (not a path).\n")
            else:
                break

    # if the mode variable has not been initialized by the command-line, run the text-based UI to get it
    if mode is None:
        while True:
            # prompt for the mode
            mode = input("What is the \"mode\" of the simulation? Type \"help\" for more information: ")
            print()

            # keep running if "help" is typed
            if mode == "help":
                print("Here are the following modes:\n0: New simulation\n1: Continuation of past simulation\n"
                      "2: Turn simulation images to video\n3: Zip previous simulation\n")
            else:
                try:
                    # get the mode as an integer make sure mode exists, break the loop if it does
                    mode = int(mode)
                    break

                # if not an integer
                except ValueError:
                    print("Input: \"mode\" should be an integer.\n")

    # return the simulation name and mode
    return name, mode


def check_new_sim(name, output_path):
    """ Check that a new simulation can be made. """
    # get file separator
    separator = os.path.sep

    while True:
        # see if the directory exists
        if os.path.isdir(output_path + name):
            # get user input for overwriting previous simulation
            print("Simulation already exists with name: " + name)
            user = input("Would you like to overwrite that simulation? (y/n): ")
            print()

            # if no overwrite, get new simulation name
            if user == "n":
                name = input("New name: ")
                print()

            # overwrite by deleting all files/folders in previous directory
            elif user == "y":
                # clear current directory to prevent another possible future errors
                files = os.listdir(output_path + name)
                for file in files:
                    # path to each file/folder
                    path = output_path + name + separator + file

                    # delete the file/folder
                    if os.path.isfile(path):
                        os.remove(path)
                    else:
                        shutil.rmtree(path)
                break
            else:
                # inputs should either be "y" or "n"
                print("Either type \"y\" or \"n\"")
        else:
            # if does not exist, make directory
            os.mkdir(output_path + name)
            break


def check_previous_sim(name, output_path):
    """ Makes sure that a previous simulation exists. """
    while True:
        # if the directory exists, break loop
        if os.path.isdir(output_path + name):
            break
        else:
            # try to get correct name
            print("No directory exists with name/path: " + output_path + name)
            name = input("Please type the correct name of the simulation or type \"exit\" to exit: ")
            print()
            if name == "exit":
                exit()


def get_final_step():
    """ Gets the new last step of the simulation if using continuation
        mode.
    """
    # try get step number from commandline
    final_step = commandline_param("-fs", int)

    # if no value, then run UI until found
    if final_step is None:
        while True:
            # prompt for the final step number
            final_step = input("What is the final step of this continued simulation? Type \"help\" for more"
                               " information: ")
            print()

            # keep running if "help" is typed
            if final_step == "help":
                print("Enter the new step number that will be the last step of the simulation.\n")
            else:
                try:
                    # get the final step as an integer, break the loop if conversion is successful
                    final_step = int(final_step)
                    break

                # if not an integer
                except ValueError:
                    print("Input: \"final step\" should be an integer.\n")

    return final_step
