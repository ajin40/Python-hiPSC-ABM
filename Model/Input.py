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

def setup(template_location):
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

    # open the .txt file that contains the initial parameters
    with open(template_location, "r") as template_file:
        lines = template_file.readlines()

    # looks at certain lines of the template file and converts them from strings to their desired type
    # the sections correspond to the sections in the template file

    # general parameters
    _name = lines[8][2:-3]
    _output_direct = lines[11][2:-3]
    _parallel = eval(lines[14][2:-3])
    _size = eval(lines[17][2:-3])
    _size = np.array([_size[0], _size[1], _size[2]])
    _resolution = eval(lines[20][2:-3])
    _num_GATA6 = int(lines[23][2:-3])
    _num_NANOG = int(lines[26][2:-3])
    _functions = eval(lines[29][2:-3])
    _num_states = int(lines[32][2:-3])

    # modes
    _output_images = eval(lines[38][2:-3])
    _output_csvs = eval(lines[41][2:-3])
    _continuation = eval(lines[44][2:-3])
    _csv_to_images = eval(lines[47][2:-3])
    _images_to_video = eval(lines[50][2:-3])

    # timing
    _beginning_step = int(lines[57][2:-3])
    _total_steps = int(lines[60][2:-3])
    _time_step_value = float(lines[63][2:-3])
    _boolean_thresh = int(lines[66][2:-3])
    _pluri_div_thresh = int(lines[69][2:-3])
    _pluri_to_diff = int(lines[72][2:-3])
    _diff_div_thresh = int(lines[75][2:-3])
    _death_thresh = int(lines[78][2:-3])

    # intercellular
    _neighbor_distance = float(lines[84][2:-3])

    # extracellular
    _extracellular = eval(lines[91][2:-3])

    # movement
    _move_time_step = float(lines[97][2:-3])
    _radius = float(lines[100][2:-3])
    _adhesion_const = float(lines[103][2:-3])
    _viscosity = float(lines[106][2:-3])
    _motility_force = float(lines[109][2:-3])
    _division_force = float(lines[112][2:-3])

    # imaging
    _image_quality = eval(lines[118][2:-3])
    _slices = int(lines[121][2:-3])
    _background_color = eval(lines[124][2:-3])
    _bound_color = eval(lines[127][2:-3])
    _color_mode = eval(lines[130][2:-3])
    _pluri_color = eval(lines[133][2:-3])
    _diff_color = eval(lines[136][2:-3])
    _pluri_gata6_high_color = eval(lines[139][2:-3])
    _pluri_nanog_high_color = eval(lines[142][2:-3])
    _pluri_both_high_color = eval(lines[145][2:-3])

    # miscellaneous/experimental
    _diff_surround = int(lines[151][2:-3])
    _stochastic = bool(lines[154][2:-3])
    _group = int(lines[157][2:-3])
    _lonely_cell = int(lines[160][2:-3])
    _contact_inhibit = int(lines[163][2:-3])
    _guye_move = bool(lines[166][2:-3])
    _dox_step = int(lines[169][2:-3])
    _max_radius = float(lines[172][2:-3])
    _move_thresh = int(lines[175][2:-3])
    _guye_radius = float(lines[178][2:-3])
    _guye_force = float(lines[181][2:-3])

    # if the mode is not a continuation and not turning CSVs into images
    if not _continuation and not _csv_to_images and not _images_to_video:
        # use the check_name function to make sure there aren't any duplicate simulations
        _path, _name = check_name(_output_direct, _name, separator)

        # copy the template file to the new directory of the simulation
        shutil.copy(template_location, _path)

    # otherwise the path becomes the directory specific to the desired simulation
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

    # Adds the initial concentration amounts to the space for each instance of the extracellular class    (base)
    # simulation.initialize_diffusion()

#######################################################################################################################
# Continuation Mode

    # if this is a continuation of previous simulation, we need to look at previous .csv for cell info
    if _continuation:
        # gets the previous .csv file by subtracting the beginning step by 1
        previous = simulation.path + simulation.name + "_values_" + str(simulation.beginning_step - 1) + ".csv"

        # calls the following function to add instances of the cell class to the simulation instance
        csv_to_simulation(simulation, previous)

#######################################################################################################################
# Convert CSVs to Images Mode

    # check if this will be turning an output of .csv files into images
    elif _csv_to_images:
        print("Turning CSVs into images...")

        # use list comprehension to get all .csv files in the directory
        csv_list = [file for file in os.listdir(simulation.path) if file.endswith(".csv")]

        # loop over all CSVs avoiding the first file which is a copy of the template
        for i in range(simulation.beginning_step, len(csv_list) + 1):
            # calls the following function to add instances of the cell class to the simulation instance
            csv_to_simulation(simulation, simulation.path + csv_list[i-1])

            # updates the instance variables as they aren't updated by anything else
            simulation.current_step = i

            # saves a snapshot of the simulation
            Output.save_file(simulation)

            # clears the array for the next round of images
            simulation.cells = np.array([], dtype=np.object)

        # turns the images into a video
        Output.image_to_video(simulation)

        # exits out as the conversion from .csv to images/video is done
        exit()

#######################################################################################################################
# Images to Video Mode

    # check if this will be turning an output of .csv files into images and a video
    elif _images_to_video:
        print("Turning images into a video...")

        # use list comprehension to get all .png files in the directory
        image_list = [file for file in os.listdir(simulation.path) if file.endswith(".png")]

        # update how many images will be made into a video
        simulation.image_counter = len(image_list)

        # turns the images into a video
        Output.image_to_video(simulation)

        # exits out as the conversion from images to video is done
        exit()

#######################################################################################################################
# Normal Mode

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

    # returns the list of simulations
    return simulation


def csv_to_simulation(simulation, csv_file):
    """ Turns rows of a into instances of the cell
        class and adds them to the simulation class
    """
    # opens the csv and lists the rows
    with open(csv_file, "r", newline="") as csv_file:
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
    return output_path + separator + name + separator, name