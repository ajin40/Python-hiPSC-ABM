import numpy as np
import random as r
import os
import shutil
import platform
import csv
import cv2

import Extracellular
import Simulation
import Output


def setup(template_location):
    """ Reads the template file, determining the
        mode in which the simulation shall be run
        and creates a Simulation object corresponding
        to desired initial parameters
    """
    # keep track of the file separator to use
    if platform.system() == "Windows":
        # windows
        separator = "\\"
    else:
        # not windows
        separator = "/"

    # open the .txt template file that contains the initial parameters
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
    _num_GATA6 = int(lines[20][2:-3])
    _num_NANOG = int(lines[23][2:-3])

    # finite dynamical system
    _functions = eval(lines[29][2:-3])
    _num_fds_states = int(lines[32][2:-3])

    # modes
    _output_images = eval(lines[38][2:-3])
    _output_csvs = eval(lines[41][2:-3])
    _continuation = eval(lines[44][2:-3])
    _csv_to_images = eval(lines[47][2:-3])
    _images_to_video = eval(lines[50][2:-3])

    # timing
    _beginning_step = int(lines[57][2:-3])
    _end_step = int(lines[60][2:-3])
    _time_step_value = float(lines[63][2:-3])
    _fds_thresh = int(lines[66][2:-3])
    _pluri_div_thresh = int(lines[69][2:-3])
    _pluri_to_diff = int(lines[72][2:-3])
    _diff_div_thresh = int(lines[75][2:-3])
    _death_thresh = int(lines[78][2:-3])

    # intercellular
    _neighbor_distance = float(lines[84][2:-3])
    _guye_distance = float(lines[87][2:-3])
    _jkr_distance = float(lines[90][2:-3])
    _lonely_cell = int(lines[93][2:-3])
    _contact_inhibit = int(lines[96][2:-3])
    _move_thresh = int(lines[99][2:-3])
    _diff_surround = int(lines[102][2:-3])

    # extracellular
    _extracellular = eval(lines[109][2:-3])
    _resolution = eval(lines[112][2:-3])

    # movement/physical
    _move_time_step = float(lines[118][2:-3])
    _radius = float(lines[121][2:-3])
    _adhesion_const = float(lines[124][2:-3])
    _viscosity = float(lines[127][2:-3])
    _motility_force = float(lines[130][2:-3])
    _max_radius = float(lines[133][2:-3])

    # imaging
    _image_quality = eval(lines[139][2:-3])
    _fps = float(lines[142][2:-3])
    _background_color = eval(lines[145][2:-3])
    _bound_color = eval(lines[148][2:-3])
    _color_mode = eval(lines[151][2:-3])
    _pluri_color = eval(lines[154][2:-3])
    _diff_color = eval(lines[157][2:-3])
    _pluri_gata6_high_color = eval(lines[160][2:-3])
    _pluri_nanog_high_color = eval(lines[163][2:-3])
    _pluri_both_high_color = eval(lines[166][2:-3])

    # miscellaneous/experimental
    _dox_step = int(lines[172][2:-3])
    _stochastic = bool(lines[175][2:-3])
    _group = int(lines[178][2:-3])
    _guye_move = bool(lines[181][2:-3])

    # check that the name and path from the template are valid
    _path, _name = check_name(_output_direct, _name, separator, _continuation, _csv_to_images, _images_to_video,
                              template_location)

    # initializes simulation class which holds all information about the simulation
    simulation = Simulation.Simulation(_path, _name, _parallel, _size, _resolution, _num_fds_states, _functions,
                                       _neighbor_distance, _guye_distance, _jkr_distance, _lonely_cell,
                                       _contact_inhibit, _move_thresh, _time_step_value, _beginning_step, _end_step,
                                       _move_time_step, _dox_step, _pluri_div_thresh, _pluri_to_diff, _diff_div_thresh,
                                       _fds_thresh, _death_thresh, _diff_surround, _adhesion_const, _viscosity, _group,
                                       _output_csvs, _output_images, _image_quality, _fps, _background_color,
                                       _bound_color, _color_mode, _pluri_color, _diff_color, _pluri_gata6_high_color,
                                       _pluri_nanog_high_color, _pluri_both_high_color, _guye_move, _motility_force,
                                       _max_radius)

    # loops over the gradients and adds them to the simulation
    for i in range(len(_extracellular)):
        # initializes the extracellular class
        new_extracellular = Extracellular.Extracellular(_size, _resolution, _extracellular[i][0],
                                                        _extracellular[i][1], _extracellular[i][2], _parallel)
        # adds the Extracellular object
        simulation.extracellular = np.append(simulation.extracellular, [new_extracellular])

    # decide which mode the simulation is intended to be run in
    # continue a previous simulation
    if _continuation:
        continue_mode(simulation)

    # turn a collection of csvs into images and a video
    elif _csv_to_images:
        csv_to_image_mode(simulation)

    # turn a collection of images into a video
    elif _images_to_video:
        images_to_video_mode(simulation)

    # starts a new simulation
    else:
        # loops over all cells and creates a stem cell object for each one with given parameters
        for i in range(_num_NANOG + _num_GATA6):
            # gives random location in the space
            location = np.array([r.random() * _size[0], r.random() * _size[1], r.random() * _size[2]])

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
            diff_counter = r.randint(0, _pluri_to_diff)
            div_counter = r.randint(0, _pluri_div_thresh)
            death_counter = r.randint(0, _death_thresh)
            bool_counter = r.randint(0, _fds_thresh)

            # get the radius based on the randomized growth
            radius = simulation.min_radius + simulation.pluri_growth * div_counter

            # set up the forces and closest differentiated cell with empty values
            motility_force = np.zeros(3, dtype=float)
            jkr_force = np.zeros(3, dtype=float)
            nearest_gata6 = np.nan
            nearest_nanog = np.nan
            nearest_diff = np.nan

            # adds object to simulation instance
            simulation.add_cell(location, radius, True, booleans, "Pluripotent", diff_counter, div_counter,
                                death_counter, bool_counter, motility_force, jkr_force, nearest_gata6, nearest_nanog,
                                nearest_diff)

    # return the modified simulation instance
    return simulation


def continue_mode(simulation):
    """ This is used for when a simulation ended
        early or had an error. It will read the previous
        csv restoring that information to the simulation.
    """
    # gets the previous .csv file by subtracting the beginning step by 1
    previous = simulation.path + "_values_" + str(simulation.beginning_step - 1) + ".csv"

    # calls the following function to add instances of the cell class to the simulation instance
    csv_to_simulation(simulation, previous)


def csv_to_image_mode(simulation):
    """ This is used for turning a collection of
        csvs into images. If you want to change the
        colors of the cells or increase the resolution
        of the images, that can be done here.
    """
    print("Turning CSVs into images...")

    # create the video file
    Output.initialize_video(simulation)

    # loop over all csvs defined in the template file
    for i in range(simulation.beginning_step, simulation.end_step + 1):
        # updates the instance variables as they aren't updated by anything else
        simulation.current_step = i

        # calls the following function to add instances of the cell class to the simulation instance
        csv_to_simulation(simulation, simulation.path + "_values_" + str(int(i)) + ".csv")

        # saves a snapshot of the simulation
        Output.step_image(simulation)

        # clears the array for the next round of images
        simulation.cell_locations = np.empty((0, 3), dtype=float)
        simulation.cell_radii = np.empty((0, 1), dtype=float)
        simulation.cell_motion = np.empty((0, 1), dtype=bool)
        simulation.cell_fds = np.empty((0, 4), dtype=float)
        simulation.cell_states = np.empty((0, 1), dtype=str)
        simulation.cell_diff_counter = np.empty((0, 1), dtype=int)
        simulation.cell_div_counter = np.empty((0, 1), dtype=int)
        simulation.cell_death_counter = np.empty((0, 1), dtype=int)
        simulation.cell_fds_counter = np.empty((0, 1), dtype=int)
        simulation.cell_motility_force = np.empty((0, 3), dtype=float)
        simulation.cell_jkr_force = np.empty((0, 3), dtype=float)
        simulation.cell_closest_diff = np.empty((0, 1), dtype=int)

        # clear the number of cells holder
        simulation.number_cells = 0

    # turns the images into a video
    Output.finish_files(simulation)

    # exits out as the conversion from .csv to images/video is done
    exit()


def images_to_video_mode(simulation):
    """ If a simulation ends early or has an
        error, this will take the images up until
        it ended and turn them into a video.
    """
    print("Turning images into a video...")

    # create the video file
    Output.initialize_video(simulation)

    # loop over all images defined in the template file
    for i in range(simulation.beginning_step, simulation.end_step + 1):
        # calls the following function to add instances of the cell class to the simulation instance
        csv_to_simulation(simulation, simulation.path + "_values_" + str(int(i)) + ".csv")

        # read the image and write it to the video file
        image_name = "_image_" + str(int(i)) + "_slice_" + str(0) + ".png"
        image = cv2.imread(simulation.path + image_name)
        simulation.video_object.write(image)

    # exits out as the conversion from images to video is done
    exit()


def csv_to_simulation(simulation, csv_file):
    """ Revalues the array holders for cell values
        based on the outputs of the csv files.
    """
    # opens the csv and lists the rows
    with open(csv_file, "r", newline="") as csv_file:
        csv_rows = list(csv.reader(csv_file))[1:]

    # updates the total amount of cells
    simulation.number_cells = len(csv_rows)

    # turn all of the rows into a matrix
    cell_data = np.column_stack(csv_rows)

    # each row of the matrix will correspond to a cell value holder. the 2D arrays must be handled differently
    simulation.cell_locations = cell_data[0:3, :].transpose().astype(float)
    simulation.cell_radii = cell_data[3].astype(float)
    simulation.cell_motion = cell_data[4] == "True"
    simulation.cell_fds = cell_data[5:9, :].transpose().astype(float).astype(int)
    simulation.cell_states = cell_data[9].astype(str)
    simulation.cell_diff_counter = cell_data[10].astype(int)
    simulation.cell_div_counter = cell_data[11].astype(int)
    simulation.cell_death_counter = cell_data[12].astype(int)
    simulation.cell_fds_counter = cell_data[12].astype(int)
    simulation.cell_motility_force = np.zeros((simulation.number_cells, 3), dtype=float)
    simulation.cell_jkr_force = np.zeros((simulation.number_cells, 3), dtype=float)
    simulation.cell_closest_diff = np.empty((simulation.number_cells, 3), dtype=None)


def check_name(output_direct, name, separator, continuation, csv_to_images, images_to_video, template_location):
    """ Renames the file if another simulation
        has the same name
    """
    # this will look for an existing directory
    if continuation or csv_to_images or images_to_video:
        # keeps the loop running until one condition is met
        while True:
            # see if the directory exists
            try:
                os.path.isdir(output_direct + separator + name)
                break

            # if not prompt to change name or end the simulation
            except OSError:
                print("No directory exists with name/path: " + output_direct + separator + name)
                user = input("Would you like to continue? (y/n): ")
                if user == "n":
                    exit()
                elif user == "y":
                    output_direct = input("What is the correct path? Don't include simulation name. (type new path): ")

    else:
        # keeps the loop running until one condition is met
        while True:
            # if the path does not already exist, make that directory and break out of the loop
            try:
                os.mkdir(output_direct + separator + name)
                break

            # prompt to either rename or overwrite
            except OSError:
                print("Simulation with identical name: " + name)
                user = input("Would you like to overwrite that simulation? (y/n): ")
                if user == "n":
                    name = input("New name: ")
                elif user == "y":
                    # clear current directory to prevent another possible future errors
                    files = os.listdir(output_direct + separator + name)
                    for file in files:
                        os.remove(output_direct + separator + name + separator + file)
                    break

        # copy the template to the directory
        shutil.copy(template_location, output_direct + separator + name)

    # updated path and name if need be
    return output_direct + separator + name + separator, name
