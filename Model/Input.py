import numpy as np
import random as r
import os
import shutil
import platform

from Model import Gradient
from Model import Simulation
from Model import Cell


def Setup():
    """ Looks at all of the setup files and turns them into
        instances of the simulation class
    """
    # keep track of the file separator to use
    if platform.system() == "Windows":
        # windows
        separator = "\\"
    else:
        # linux/unix
        separator = "/"

    # holds all of the instances of the simulation class
    simulations = []

    # opens the txt file of paths for input/output files
    locations = open(os.getcwd() + separator + "Locations.txt", "r")
    locations_list = locations.readlines()
    input_path = locations_list[1].strip()
    output_path = locations_list[4].strip()

    files = os.listdir(input_path)

    # loops over all of the files in the director and turns the file lines into a list
    for file in files:
        setup_file = open(input_path + separator + file, "r")
        setup_list = setup_file.readlines()
        parameters = []

        # looks at every third line, the ones with parameters
        for i in range(1, len(setup_list), 3):
            parameters.append(setup_list[i][2:-3])

        # organizes all of the parameters and converts them from strings to their desired type
        _path = check_name(output_path, parameters[0], separator)
        _parallel = eval(parameters[1])
        _end_time = float(parameters[2])
        _time_step = float(parameters[3])
        _num_GATA6 = int(parameters[4])
        _num_NANOG = int(parameters[5])
        _stochastic = bool(parameters[6])
        _size = eval(parameters[7])
        _functions = eval(parameters[8])
        _pluri_div_thresh = float(parameters[9])
        _diff_div_thresh = float(parameters[10])
        _pluri_to_diff = float(parameters[11])
        _diff_surround_value = int(parameters[12])
        _death_threshold = float(parameters[13])
        _move_time_step = float(parameters[14])
        _move_max_time = float(parameters[15])
        _spring_constant = float(parameters[16])
        _friction = float(parameters[17])
        _energy_kept = float(parameters[18])
        _neighbor_distance = float(parameters[19])
        _mass = float(parameters[20])
        _gradients = eval(parameters[21])
        _density = float(parameters[22])
        _n = int(parameters[23])
        _quality = int(parameters[24])
        _group = int(parameters[25])
        _speed = int(parameters[26])

        # initializes simulation class which holds all information about the simulation
        simulation = Simulation.Simulation(_path, _end_time, _time_step, _pluri_div_thresh, _diff_div_thresh,
                                           _pluri_to_diff, _size, _diff_surround_value, _functions, _parallel,
                                           _death_threshold, _move_time_step, _move_max_time, _spring_constant,
                                           _friction, _energy_kept, _neighbor_distance, _density, _n, _quality,
                                           _group, _speed)

        # copies the setup file to the new directory of the simulation
        shutil.copy(input_path + separator + file, simulation.path)

        # loops over the gradients and adds them to the simulation
        for i in range(len(_gradients)):

            # initializes the gradient class
            new_gradient = Gradient.Gradient(_gradients[i][0], _size, int(_gradients[i][1]), _parallel)

            # adds the gradient object
            simulation.gradients = np.append(simulation.gradients, new_gradient)


        # loops over all cells and creates a stem cell object for each one with given parameters
        for i in range(_num_NANOG + _num_GATA6):

            location = np.array([r.random() * _size[0], r.random() * _size[1], 0.0])
            if _size[2] != 1:
                location[2] = r.random() * _size[2]

            state = "Pluripotent"
            motion = True
            mass = _mass

            if i < _num_NANOG:
                if _stochastic:
                    booleans = np.array([r.randint(0, 1), r.randint(0, 1), 0, 1])
                else:
                    booleans = np.array([0, 0, 0, 1])
            else:
                if _stochastic:
                    booleans = np.array([r.randint(0, 1), r.randint(0, 1), 1, 0])
                else:
                    booleans = np.array([0, 0, 1, 0])

            diff_counter = _pluri_to_diff * r.random()
            division_counter = _pluri_div_thresh * r.random()
            death_counter = _death_threshold * r.random()
            velocity = np.array([0.0, 0.0, 0.0], np.float32)

            # creates instance of Cell class
            cell_obj = Cell.Cell(location, motion, velocity, mass, booleans, state, diff_counter, division_counter,
                                 death_counter)

            # adds object to simulation instance
            simulation.add_cell(cell_obj)

        # adds simulation to the list
        simulations.append(simulation)

    # returns the list of simulations
    return simulations

def check_name(path, name, separator):
    """ Renames the file if another simulation
        has the same name
    """
    while True:
        try:
            os.mkdir(path + name)
            break
        except:
            print("Simulation with identical name: " + str(name))
            user = input("Would you like to overwrite that simulation? (y/n): ")
            if user == "n":
                name = input("New name: ")
            if user == "y":
                files = os.listdir(path + name)
                for file in files:
                    os.remove(path + name + separator + file)
                break
    return path + name + separator