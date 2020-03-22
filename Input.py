import numpy as np
import random as r
import os
import shutil
import platform
import Gradient
import Simulation
import Cell


def Setup():
    """ Looks at all of the setup files and turns them into
        instances of the simulation class
    """

    # which file separator to use
    if platform.system() == "Windows":
        # windows
        sep = "\\"
    else:
        # linux/unix
        sep = "/"

    # holds all of the instances of the simulation class
    simulations = []

    # opens the directory of the setup files
    files = os.listdir(os.getcwd() + sep + "Setup_files")

    # loops over all of the files in the directory
    for file in files:

        # opens the files and turns the lines into a list
        setup_file = open(os.getcwd() + sep + "Setup_files" + sep + file, "r")
        setup_list = setup_file.readlines()
        parameters = []

        # looks at every third line, the ones with parameters
        for i in range(len(setup_list)):
            if i % 3 == 1:
                parameters.append(setup_list[i][2:-3])

        # organizes all of the parameters and converts them from strings to their desired type
        _name = str(parameters[0])
        _path = str(parameters[1])
        _parallel = eval(parameters[2])
        _end_time = float(parameters[3])
        _time_step = float(parameters[4])
        _num_GATA6 = int(parameters[5])
        _num_NANOG = int(parameters[6])
        _stochastic = bool(parameters[7])
        _size = eval(parameters[8])
        _functions = eval(parameters[9])
        _pluri_div_thresh = float(parameters[10])
        _diff_div_thresh = float(parameters[11])
        _pluri_to_diff = float(parameters[12])
        _diff_surround_value = int(parameters[13])
        _death_threshold = int(parameters[14])
        _move_time_step = float(parameters[15])
        _move_max_time = float(parameters[16])
        _spring_constant = float(parameters[17])
        _friction = float(parameters[18])
        _energy_kept = float(parameters[19])
        _neighbor_distance = float(parameters[20])
        _mass = float(parameters[21])
        _radius = float(parameters[22])
        _gradients = eval(parameters[23])
        _three_D = eval(parameters[24])
        _density = float(parameters[25])
        _n = int(parameters[26])
        _sep = sep

        # initializes simulation class which holds all information about the simulation
        simulation = Simulation.Simulation(_name, _path, _end_time, _time_step, _pluri_div_thresh, _diff_div_thresh,
                                           _pluri_to_diff, _size, _diff_surround_value, _functions, _parallel,
                                           _death_threshold, _move_time_step, _move_max_time, _spring_constant,
                                           _friction, _energy_kept, _neighbor_distance, _three_D, _density, _n, _sep)

        # checks to see if the simulation name is desired and valid
        check_name(simulation)

        # copies the setup file to the new directory for each instance of simulation
        shutil.copy(os.getcwd() + "/Setup_files/" + file, simulation.path + simulation.sep + simulation.name +
                    simulation.sep)

        # loops over the gradients and adds them to the simulation
        for i in range(len(_gradients)):

            # initializes the gradient class
            gradient_obj = Gradient.Gradient(_gradients[i][0], _size, int(_gradients[i][1]), _parallel)

            # adds the gradient object
            simulation.add_gradient(gradient_obj)

        # loops over all NANOG_high cells and creates a stem cell object for each one with given parameters
        for i in range(_num_NANOG):

            # random location on grid
            if _three_D:
                location = np.array([r.random() * _size[0], r.random() * _size[1], r.random() * _size[2]])
            else:
                location = np.array([r.random() * _size[0], r.random() * _size[1], 0.0])


            # initially Pluripotent
            state = "Pluripotent"

            # initially Moving
            motion = True

            # initial mass
            mass = _mass

            # staring boolean values
            if _stochastic:
                booleans = np.array([r.randint(0, 1), r.randint(0, 1), 0, 1])
            else:
                booleans = np.array([0, 0, 0, 1])

            # initial radius
            radius = _radius

            # gives random initial differentiation timer
            diff_timer = _pluri_to_diff * r.random() * 0.5

            # gives random initial division timer
            division_timer = _pluri_div_thresh * r.random()

            # gives random initial death timer
            death_timer = _death_threshold * r.random()

            velocity = np.array([0.0, 0.0, 0.0], np.float32)

            # creates instance of Cell class
            sim_obj = Cell.Cell(location, motion, velocity, mass, radius, booleans, state, diff_timer,
                                division_timer, death_timer)

            # adds object to simulation instance
            simulation.add_cell(sim_obj)

        # loops over all GATA6_high cells and creates a stem cell object for each one with given parameters
        for i in range(_num_GATA6):

            # random location on grid
            if _three_D:
                location = np.array([r.random() * _size[0], r.random() * _size[1], r.random() * _size[2]])
            else:
                location = np.array([r.random() * _size[0], r.random() * _size[1], 0.0])

            # initially Pluripotent
            state = "Pluripotent"

            # initially Moving
            motion = True

            # initial mass
            mass = _mass

            # staring boolean values
            if _stochastic:
                booleans = np.array([r.randint(0, 1), r.randint(0, 1), 1, 0])
            else:
                booleans = np.array([0, 0, 1, 0])

            # initial radius
            radius = _radius

            # gives random initial differentiation timer
            diff_timer = _pluri_to_diff * r.random() * 0.5

            # gives random initial division timer
            division_timer = _pluri_div_thresh * r.random()

            # gives random initial death timer
            death_timer = _death_threshold * r.random()

            velocity = np.array([0.0, 0.0, 0.0], np.float32)

            # creates instance of Cell class
            sim_obj = Cell.Cell(location, motion, velocity, mass, radius, booleans, state, diff_timer,
                                division_timer, death_timer)

            # adds object to simulation instance
            simulation.add_cell(sim_obj)

        # adds simulation to the list
        simulations.append(simulation)

    # returns the list of simulations
    return simulations


def check_name(simulation):
    """ Renames the file if another simulation
        has the same name
    """
    while True:
        try:
            os.mkdir(simulation.path + simulation.sep + simulation.name)
            break
        except OSError:
            print("Simulation with identical name")
            user = input("Would you like to overwrite the that simulation? (y/n): ")
            if user == "n":
                simulation.name = input("New name: ")
            if user == "y":
                shutil.rmtree(simulation.path + simulation.sep + simulation.name)
                os.mkdir(simulation.path + simulation.sep + simulation.name)
                break