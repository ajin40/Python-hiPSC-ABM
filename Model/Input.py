import numpy as np
import random as r
import os
import shutil
import platform

from Model import Extracellular
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

    # list of all potential input files
    files = os.listdir(input_path)

    # loops over all of the files in the director and turns the file lines into a list
    for file in files:
        parameters_file = open(input_path + separator + file, "r")
        parameters = parameters_file.readlines()

        # looks at certain lines of the template file and converts them from strings to their desired type
        _path = check_name(output_path, parameters[1][2:-3], separator)
        _parallel = eval(parameters[4][2:-3])
        _end_time = float(parameters[7][2:-3])
        _time_step = float(parameters[10][2:-3])
        _num_GATA6 = int(parameters[13][2:-3])
        _num_NANOG = int(parameters[16][2:-3])
        _stochastic = bool(parameters[19][2:-3])
        _size = eval(parameters[22][2:-3])
        _functions = eval(parameters[25][2:-3])
        _pluri_div_thresh = float(parameters[28][2:-3])
        _diff_div_thresh = float(parameters[31][2:-3])
        _pluri_to_diff = float(parameters[34][2:-3])
        _diff_surround_value = int(parameters[37][2:-3])
        _death_threshold = float(parameters[40][2:-3])
        _move_time_step = float(parameters[43][2:-3])
        _move_max_time = float(parameters[46][2:-3])
        _spring_constant = float(parameters[49][2:-3])
        _friction = float(parameters[52][2:-3])
        _energy_kept = float(parameters[55][2:-3])
        _neighbor_distance = float(parameters[58][2:-3])
        _mass = float(parameters[61][2:-3])
        _gradients = eval(parameters[64][2:-3])
        _density = float(parameters[67][2:-3])
        _num_states = int(parameters[70][2:-3])
        _quality = int(parameters[73][2:-3])
        _group = int(parameters[76][2:-3])
        _speed = int(parameters[79][2:-3])
        _max_radius = float(parameters[82][2:-3])
        _dx = float(parameters[85][2:-3])
        _dy = float(parameters[88][2:-3])
        _dz = float(parameters[91][2:-3])
        _diffuse_const = float(parameters[94][2:-3])
        _avg_initial = float(parameters[97][2:-3])

        # initializes simulation class which holds all information about the simulation
        simulation = Simulation.Simulation(_path, _end_time, _time_step, _pluri_div_thresh, _diff_div_thresh,
                                           _pluri_to_diff, _size, _diff_surround_value, _functions, _parallel,
                                           _death_threshold, _move_time_step, _move_max_time, _spring_constant,
                                           _friction, _energy_kept, _neighbor_distance, _density, _num_states,
                                           _quality, _group, _speed, _max_radius)

        # copies the setup file to the new directory of the simulation
        shutil.copy(input_path + separator + file, simulation.path)

        # loops over the gradients and adds them to the simulation
        for i in range(len(_gradients)):

            # initializes the extracellular class
            new_extracellular = Extracellular.Extracellular(_size, _dx, _dy, _dz, _diffuse_const, _avg_initial, _parallel)

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
            print("Simulation with identical name: " + str(name))
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