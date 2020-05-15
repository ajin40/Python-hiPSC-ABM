import numpy as np
import random as r
import os
import shutil
import platform
import csv

import Extracellular
import Simulation
import Cell
import Output

"""
Input.py consists of two functions both of which are used to get the simulation started. If new functionality
is added to the model and subsequently the template file, that can be implemented here. 
"""

def setup(location_of_templates):
    """ Looks at all of the setup files and turns them into
        instances of the simulation class, which contain
        multiple instances of the cell each representing
        a single cell.
    """
    # keep track of the file separator to use
    if platform.system() == "Windows":
        # windows
        separator = "\\"
    else:
        # linux/unix/mac
        separator = "/"

    # holds all of the instances of the simulation class
    simulations = []

    # location of the template file used for initial parameters
    files = os.listdir(location_of_templates)

    # loops over all of the files in the director and turns the file lines into a list
    for file in files:
        # let user know that all files in the Setup_files directory should be .txt
        if not file.endswith(".txt"):
            print("All files in Setup_files should be template files used for simulation setup.")

        # open the template file
        parameters_file = open(location_of_templates + separator + file, "r")
        parameters = parameters_file.readlines()

        # looks at certain lines of the template file and converts them from strings to their desired type
        # general parameters
        _name = parameters[8][2:-3]
        _output_direct = parameters[11][2:-3]
        _parallel = eval(parameters[14][2:-3])
        _size = eval(parameters[17][2:-3])
        _resolution = eval(parameters[20][2:-3])
        _num_GATA6 = int(parameters[23][2:-3])
        _num_NANOG = int(parameters[26][2:-3])
        _functions = eval(parameters[29][2:-3])
        _num_states = int(parameters[32][2:-3])

        # modes
        _output_images = eval(parameters[38][2:-3])
        _output_csvs = eval(parameters[41][2:-3])
        _continuation = eval(parameters[44][2:-3])
        _csv_to_images = eval(parameters[47][2:-3])

        # timing
        _beginning_step = int(parameters[54][2:-3])
        _total_steps = int(parameters[57][2:-3])
        _time_step_value = float(parameters[60][2:-3])
        _boolean_thresh = int(parameters[63][2:-3])
        _pluri_div_thresh = int(parameters[66][2:-3])
        _pluri_to_diff = int(parameters[69][2:-3])
        _diff_div_thresh = int(parameters[72][2:-3])
        _death_thresh = int(parameters[75][2:-3])

        # intercellular
        _neighbor_distance = float(parameters[81][2:-3])

        # extracellular
        _extracellular = eval(parameters[88][2:-3])

        # movement
        _move_time_step = float(parameters[94][2:-3])
        _radius = float(parameters[97][2:-3])
        _adhesion_const = float(parameters[100][2:-3])
        _viscosity = float(parameters[103][2:-3])
        _motility_force = float(parameters[106][2:-3])
        _division_force = float(parameters[109][2:-3])

        # imaging
        _image_quality = eval(parameters[115][2:-3])
        _slices = int(parameters[118][2:-3])
        _background_color = eval(parameters[121][2:-3])
        _bound_color = eval(parameters[124][2:-3])
        _color_mode = eval(parameters[127][2:-3])
        _pluri_color = eval(parameters[130][2:-3])
        _diff_color = eval(parameters[133][2:-3])
        _pluri_gata6_high_color = eval(parameters[136][2:-3])
        _pluri_nanog_high_color = eval(parameters[139][2:-3])
        _pluri_both_high_color = eval(parameters[142][2:-3])

        # miscellaneous/experimental
        _diff_surround = int(parameters[148][2:-3])
        _stochastic = bool(parameters[151][2:-3])
        _group = int(parameters[154][2:-3])
        _lonely_cell = int(parameters[157][2:-3])
        _contact_inhibit = int(parameters[160][2:-3])
        _guye_move = bool(parameters[163][2:-3])
        _dox_step = int(parameters[166][2:-3])
        _max_radius = float(parameters[169][2:-3])
        _move_thresh = int(parameters[172][2:-3])
        _guye_radius = float(parameters[175][2:-3])
        _guye_force = float(parameters[178][2:-3])

        # if it's not a continuation, copy the template file and check to see if the name is valid
        if not _continuation and not _csv_to_images:
            _path, _name = check_name(_output_direct, _name, separator)
            _path += separator
            shutil.copy(location_of_templates + separator + file, _path)

        else:
            _path = _output_direct + separator + _name + separator

        # initializes simulation class which holds all information about the simulation
        simulation = Simulation.Simulation(_name, _path, _parallel, _size, _resolution, _num_states, _functions,
                                           _neighbor_distance, _time_step_value, _beginning_step, _total_steps,
                                           _move_time_step, _pluri_div_thresh, _pluri_to_diff, _diff_div_thresh,
                                           _boolean_thresh, _death_thresh, _diff_surround, _adhesion_const, _viscosity,
                                           _group, _slices, _image_quality, _background_color, _bound_color,
                                           _color_mode, _pluri_color, _diff_color, _pluri_gata6_high_color,
                                           _pluri_nanog_high_color, _pluri_both_high_color, _lonely_cell,
                                           _contact_inhibit, _guye_move, _motility_force, _dox_step, _max_radius,
                                           _division_force, _move_thresh, _output_images, _output_csvs, _guye_radius,
                                           _guye_force)

        # loops over the gradients and adds them to the simulation
        for i in range(len(_extracellular)):

            # initializes the extracellular class
            new_extracellular = Extracellular.Extracellular(_size, _resolution, _extracellular[i][0],
                                                            _extracellular[i][1], _extracellular[i][2], _parallel)
            # adds the Extracellular object
            simulation.extracellular = np.append(simulation.extracellular, [new_extracellular])

        # if this is a continuation of previous simulation, we need to look at previous .csv for cell info
        if _continuation:
            # gets the previous .csv file by subtracting the beginning step by 1
            previous = simulation.path + simulation.name + "_values_" + str(simulation.beginning_step - 1) + ".csv"

            # calls the following function to add instances of the cell class to the simulation instance
            csv_to_simulation(simulation, previous)

        # check if this will be turning an output of .csv files into images and a video
        elif _csv_to_images:
            print("Turning CSVs into images...")
            # get the files from the specified directory
            list_of_files = os.listdir(simulation.path)
            length_of_direct = len(list_of_files)

            # loop over all CSVs avoiding the first file which is a copy of the template
            for i in range(1, length_of_direct):
                # gets the previous .csv file by subtracting the beginning step by 1
                current_csv_name = simulation.path + simulation.name + "_values_" + str(i) + ".csv"

                # calls the following function to add instances of the cell class to the simulation instance
                csv_to_simulation(simulation, current_csv_name)

                # updates the instance variables as they aren't updated by anything else
                simulation.beginning_step = 1
                simulation.current_step = i

                # saves a snapshot of the simulation
                Output.save_file(simulation)

                # clears the array for the next round of images
                simulation.cells = np.array([], dtype=np.object)

            # turns the images into a video
            Output.image_to_video(simulation)

            # exits out as the conversion from .csv to images/video is done
            exit()

        # if this simulation is not a continuation
        else:
            # loops over all cells and creates a stem cell object for each one with given parameters
            for i in range(_num_NANOG + _num_GATA6):
                # gives random location in the space
                _location = np.array([r.random() * _size[0], r.random() * _size[1], r.random() * _size[2]])

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
                _diff_counter = r.randint(0, _pluri_to_diff)
                _div_counter = r.randint(0, _pluri_div_thresh)
                _death_counter = r.randint(0, _death_thresh)
                _boolean_counter = r.randint(0, _boolean_thresh)

                radius = simulation.min_radius + simulation.pluri_growth * _div_counter

                # creates instance of Cell class
                cell = Cell.Cell(_location, radius, True, booleans, "Pluripotent", _diff_counter, _div_counter,
                                 _death_counter, _boolean_counter)

                # adds object to simulation instance
                simulation.add_cell(cell)

        # adds simulation to the list
        simulations.append(simulation)

    # returns the list of simulations
    return simulations


def csv_to_simulation(simulation, file_name):
    """ Turns rows of a into instances of the cell
        class and adds them to the simulation class
    """
    # opens the csv and lists the rows
    csv_file = open(file_name, "r", newline="")
    csv_rows = list(csv.reader(csv_file))

    # loop over all rows in the csv, each line represents a cell
    for row in csv_rows[1:]:
        # get the parameters from each row in the csv
        location_x, location_y, location_z = eval(row[0]), eval(row[1]), eval(row[2])
        location = np.array([location_x, location_y, location_z])
        radius = eval(row[3])
        motion = eval(row[4])
        fgfr, erk, gata6, nanog = eval(row[5]), eval(row[6]), eval(row[7]), eval(row[8])
        booleans = np.array([fgfr, erk, gata6, nanog])
        state = row[9]
        diff_counter = eval(row[10])
        div_counter = eval(row[11])
        death_counter = eval(row[12])
        boolean_counter = eval(row[13])

        # create instance of cell class based on the following parameters taken from the last csv
        cell = Cell.Cell(location, radius, motion, booleans, state, diff_counter, div_counter,
                         death_counter, boolean_counter)

        # adds object to simulation instance
        simulation.add_cell(cell)


def check_name(output_path, name, separator):
    """ Renames the file if another simulation
        has the same name
    """
    # keeps the loop running until one condition is met
    while True:
        # if the path does not already exist, make that directory and break out of the loop
        try:
            os.mkdir(output_path + separator + name)
            break

        # prompt to either rename or overwrite
        except OSError:
            print("Simulation with identical name: " + name)
            user = input("Would you like to overwrite that simulation? (y/n): ")
            if user == "n":
                name = input("New name: ")
            if user == "y":
                # clear current directory to prevent another possible future errors
                files = os.listdir(output_path + separator + name)
                for file in files:
                    os.remove(output_path + separator + name + separator + file)
                break

    # updated path and name if need be
    return output_path + separator + name, name