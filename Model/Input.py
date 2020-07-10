import random as r
import numpy as np
import csv
import cv2
import os
import platform
import shutil

import Output


def setup_simulation(simulation):
    """ Reads the template file, determining the
        mode in which the simulation shall be run
        and creates a Simulation object corresponding
        to desired initial parameters
    """
    # decide which mode the simulation is intended to be run in
    # continue a previous simulation
    if simulation.continuation:
        continue_mode(simulation)

    # turn a collection of csvs into images and a video
    elif simulation.csv_to_images:
        csv_to_image_mode(simulation)

    # turn a collection of images into a video
    elif simulation.images_to_video:
        images_to_video_mode(simulation)

    # starts a new simulation
    else:
        # opens the data csv file and the video file as these will be continually modified.
        Output.initialize_csv(simulation)
        Output.initialize_video(simulation)

        # loops over all cells and creates a stem cell object for each one with given parameters
        for i in range(simulation.num_NANOG + simulation.num_GATA6):
            # gives random location in the space
            location = np.array([r.random() * simulation.size[0],
                                 r.random() * simulation.size[1],
                                 r.random() * simulation.size[2]])

            # give the cells starting states for the finite dynamical system
            if i < simulation.num_NANOG:
                if simulation.stochastic:
                    booleans = np.array([r.randint(0, 1), r.randint(0, 1), 0, 1])
                else:
                    booleans = np.array([0, 0, 0, 1])
            else:
                if simulation.stochastic:
                    booleans = np.array([r.randint(0, 1), r.randint(0, 1), 1, 0])
                else:
                    booleans = np.array([0, 0, 1, 0])

            # randomize the starting counters for differentiation, division, and death
            diff_counter = r.randint(0, simulation.pluri_to_diff)
            div_counter = r.randint(0, simulation.pluri_div_thresh)
            death_counter = r.randint(0, simulation.death_thresh)
            bool_counter = r.randint(0, simulation.fds_thresh)

            # get the radius based on the randomized growth
            radius = simulation.min_radius + simulation.pluri_growth * div_counter

            # set up the forces and closest differentiated cell with empty values
            motility_force = np.zeros(3, dtype=float)
            jkr_force = np.zeros(3, dtype=float)
            nearest_gata6 = np.nan
            nearest_nanog = np.nan
            nearest_diff = np.nan
            highest_fgf4 = np.array([np.nan, np.nan, np.nan])

            # adds object to simulation instance
            simulation.add_cell(location, radius, True, booleans, "Pluripotent", diff_counter, div_counter,
                                death_counter, bool_counter, motility_force, jkr_force, nearest_gata6, nearest_nanog,
                                nearest_diff, highest_fgf4)

    # return the modified simulation instance
    return simulation


def continue_mode(simulation):
    """ This is used for when a simulation ended
        early or had an error. It will read the previous
        csv restoring that information to the simulation.
    """
    # gets the previous .csv file by subtracting the beginning step by 1
    previous_file = simulation.path + simulation.name + "_values_" + str(simulation.beginning_step - 1) + ".csv"

    # calls the following function to add instances of the cell class to the simulation instance
    csv_to_simulation(simulation, previous_file)

    # create a CSV file used to hold information about run time, number of cells, memory, and various other statistics
    data_path = simulation.path + simulation.name + "_data.csv"

    # open the file, appending to it rather than writing
    file_object = open(data_path, "a", newline="")
    simulation.csv_object = csv.writer(file_object)

    # add all of the previous images to a new video object no append exists
    Output.initialize_video(simulation)
    for i in range(1, simulation.beginning_step):
        image = cv2.imread(simulation.path + simulation.name + "_image_" + str(i) + ".png")
        simulation.video_object.write(image)


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
    # opens the csv and create a list of the rows
    with open(csv_file, "r", newline="") as csv_file:
        # skip the first row as this is a header
        csv_rows = list(csv.reader(csv_file))[1:]

    # updates the number of cells and adds that amount of vertices to the graphs
    simulation.number_cells = len(csv_rows)
    simulation.neighbor_graph.add_vertices(simulation.number_cells)
    simulation.jkr_graph.add_vertices(simulation.number_cells)

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
    simulation.cell_nearest_gata6 = np.empty((simulation.number_cells, 3), dtype=None)
    simulation.cell_nearest_nanog = np.empty((simulation.number_cells, 3), dtype=None)
    simulation.cell_nearest_diff = np.empty((simulation.number_cells, 3), dtype=None)


def check_name(simulation, template_location):
    """ Renames the file if another simulation
        has the same name or checks to make
        sure such a simulation exists
    """
    # keep track of the file separator to use
    if platform.system() == "Windows":
        # windows
        separator = "\\"
    else:
        # not windows
        separator = "/"

    # this will look for an existing directory
    if simulation.continuation or simulation.csv_to_images or simulation.images_to_video:
        # keeps the loop running until one condition is met
        while True:
            # see if the directory exists
            if os.path.isdir(simulation.output_direct + separator + simulation.name):
                break

            # if not prompt to change name or end the simulation
            else:
                print("No directory exists with name/path: " + simulation.output_direct + separator + simulation.name)
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
                os.mkdir(simulation.output_direct + separator + simulation.name)
                break

            # prompt to either rename or overwrite
            except OSError:
                print("Simulation with identical name: " + simulation.name)
                user = input("Would you like to overwrite that simulation? (y/n): ")
                if user == "n":
                    simulation.name = input("New name: ")
                elif user == "y":
                    # clear current directory to prevent another possible future errors
                    files = os.listdir(simulation.output_direct + separator + simulation.name)
                    for file in files:
                        os.remove(simulation.output_direct + separator + simulation.name + separator + file)
                    break

        # copy the template to the directory
        shutil.copy(template_location, simulation.output_direct + separator + simulation.name)

    return simulation.output_direct + separator + simulation.name + separator
