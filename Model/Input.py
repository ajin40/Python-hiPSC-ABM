import numpy as np
import random as r
import os
import shutil
import platform

import Extracellular
import Simulation
import Cell


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

    # list of all potential input files
    files = os.listdir(input_path)

    # loops over all of the files in the director and turns the file lines into a list
    for file in files:
        parameters_file = open(input_path + separator + file, "r")
        parameters = parameters_file.readlines()

        # looks at certain lines of the template file and converts them from strings to their desired type
        # general parameters
        _path = check_name(output_path, parameters[8][3:-4], separator)
        _parallel = eval(parameters[11][2:-3])
        _size = eval(parameters[14][2:-3])
        _resolution = eval(parameters[17][2:-3])
        _num_GATA6 = int(parameters[20][2:-3])
        _num_NANOG = int(parameters[23][2:-3])
        _functions = eval(parameters[26][2:-3])
        _num_states = int(parameters[29][2:-3])

        # timing
        _end_time = float(parameters[35][2:-3])
        _time_step = float(parameters[38][2:-3])
        _pluri_div_thresh = float(parameters[41][2:-3])
        _diff_div_thresh = float(parameters[44][2:-3])
        _pluri_to_diff = float(parameters[47][2:-3])
        _death_threshold = float(parameters[50][2:-3])

        # intercellular
        _neighbor_distance = float(parameters[56][2:-3])

        # extracellular
        _extracellular = eval(parameters[63][2:-3])

        # collision handling
        _move_max_time = float(parameters[69][2:-3])
        _move_time_step = float(parameters[72][2:-3])
        _spring_constant = float(parameters[75][2:-3])
        _friction = float(parameters[78][2:-3])
        _energy_kept = float(parameters[81][2:-3])
        _mass = float(parameters[84][2:-3])
        _density = float(parameters[87][2:-3])

        # miscellaneous/experimental
        _diff_surround_value = int(parameters[93][2:-3])
        _quality = eval(parameters[96][2:-3])
        _group = int(parameters[99][2:-3])
        _speed = float(parameters[102][2:-3])
        _max_radius = float(parameters[105][2:-3])
        _stochastic = bool(parameters[108][2:-3])
        _slices = int(parameters[111][2:-3])

        # initializes simulation class which holds all information about the simulation
        simulation = Simulation.Simulation(_path, _end_time, _time_step, _pluri_div_thresh, _diff_div_thresh,
                                           _pluri_to_diff, _size, _diff_surround_value, _functions, _parallel,
                                           _death_threshold, _move_time_step, _move_max_time, _spring_constant,
                                           _friction, _energy_kept, _neighbor_distance, _density, _num_states,
                                           _quality, _group, _speed, _max_radius, _slices)

        # copies the setup file to the new directory of the simulation
        shutil.copy(input_path + separator + file, simulation.path)

        # loops over the gradients and adds them to the simulation
        for i in range(len(_extracellular)):

            # initializes the extracellular class
            new_extracellular = Extracellular.Extracellular(_size, _resolution, _extracellular[i][0],
                                                            _extracellular[i][1], _extracellular[i][2], _parallel)

            # adds the Extracellular object
            simulation.extracellular = np.append(simulation.extracellular, new_extracellular)


        # loops over all cells and creates a stem cell object for each one with given parameters
        for i in range(_num_NANOG + _num_GATA6):

            # gives random location in environment
            location = np.array([r.random() * _size[0], r.random() * _size[1], r.random() * _size[2]])

            # start the cells as Pluripotent, moving, and preset mass
            state = "Pluripotent"
            motion = True
            mass = _mass

            # give the cells starting states for the finite dynamical system
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

            # randomize the starting counters for differentiation, division, and death
            diff_counter = _pluri_to_diff * r.random()
            division_counter = _pluri_div_thresh * r.random()
            death_counter = _death_threshold * r.random()

            # start with no velocity vector will be updated as the cell is in motion
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
    # keeps the loop running until one condition is met
    while True:
        # if the path does not already exist, make that directory and break out of the loop
        try:
            os.mkdir(path + name)
            break

        # prompt to either rename or overwrite
        except:
            print("Simulation with identical name: " + name)
            user = input("Would you like to overwrite that simulation? (y/n): ")
            if user == "n":
                name = input("New name: ")
            if user == "y":
                # clear current directory to prevent another possible future errors
                files = os.listdir(path + name)
                for file in files:
                    os.remove(path + name + separator + file)
                break

    # updated path and name if need be
    return path + name + separator