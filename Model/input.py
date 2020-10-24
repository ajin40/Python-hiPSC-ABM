import numpy as np
import csv
import os
import platform
import sys
import pickle
import shutil
import natsort

import output
import functions
import parameters


def setup():
    """ controls which mode of the model is run and
        sets up the model accordingly
    """
    # keep track of the file separator to use
    if platform.system() == "Windows":
        separator = "\\"  # windows
    else:
        separator = "/"  # not windows

    # open the path.txt file containing the absolute locations for the template file directory and output directory
    with open('paths.txt', 'r') as file:
        lines = file.readlines()

    # get the paths to the directory of template files and the output directory, exit() if paths don't exist
    templates_path = lines[7].strip()
    if not os.path.isdir(templates_path):
        print("Path: " + templates_path + " to templates directory does not exist. Please use absolute path.")
        exit()
    else:
        # add separator to end if none is given
        if templates_path[-1] != "\\" or templates_path[-1] != "/":
            templates_path += separator

    output_path = lines[13].strip()
    if not os.path.isdir(output_path):
        print("Path: " + output_path + " to output directory does not exist. Please use absolute path.")
        exit()
    else:
        # add separator to end if none is given
        if output_path[-1] != "\\" or output_path[-1] != "/":
            output_path += separator

    # if no command line attributes besides the file, run the mini gui
    if len(sys.argv) == 1:
        # get the name of the simulation
        while True:
            name = input("What is the \"name\" of the simulation? Type \"help\" for more information: ")
            if name == "help":
                print("Type the name of the simulation without quotes and not as a path.\n")
            else:
                break

        # get the mode of the simulation
        while True:
            mode = input("What is the \"mode\" of the simulation? Type \"help\" for more information: ")
            if mode == "help":
                print("\nHere are the following modes:")
                print("new simulation: 0")
                print("continuation of past simulation: 1")
                print("turn simulation images to video: 2")
                print("turn simulation csvs to images/video: 3\n")
            else:
                try:
                    mode = int(mode)
                    break
                except ValueError:
                    print("\"mode\" should be an integer")

    # if both the name and the mode are provided, turn the mode into a integer
    elif len(sys.argv) == 3:
        name, mode = sys.argv[1], int(sys.argv[2])

    # if anything else
    else:
        print("See documentation for running a simulation via command line arguments. No arguments\n"
              "will prompt for the name of the simulation and the mode in which it should be run.")
        exit()

    # check the name of the simulation and create a path to simulation output directory
    name, output_path, path = check_name(name, output_path, separator, mode)

    # run the model normally
    if mode == 0:
        # create instance of Simulation class
        simulation = parameters.Simulation(templates_path, name, path, mode, separator)

        # create directory within the simulation output directory for template files and copy them there
        new_template_direct = simulation.path + simulation.name + "_templates"
        os.mkdir(new_template_direct)
        for template_name in os.listdir(templates_path):
            shutil.copy(templates_path + template_name, new_template_direct)

        # make a directories for outputting images, csvs, gradients, etc.
        output.initialize_outputs(simulation)

        # places all of the diffusion points into bins so that the model can use a bin sorting method to when
        # determining highest/lowest concentrations of the extracellular gradient(s) only needed when beginning
        # a new simulation
        functions.setup_diffusion_bins(simulation)

    # continue a past simulation
    elif mode == 1:
        # get the new end step of the simulation
        end_step = int(input("What is the final step of this continuation? "))

        # open the temporary pickled simulation and update the beginning step and the end step
        with open(path + name + "_temp.pkl", "rb") as temp_file:
            simulation = pickle.load(temp_file)
            simulation.beginning_step = simulation.current_step + 1
            simulation.end_step = end_step

        # make sure the proper output directories exist
        output.initialize_outputs(simulation)

    # images to video mode
    elif mode == 2:
        # create instance of Simulation class used to get imaging information
        simulation = parameters.Simulation(templates_path, name, path, mode, separator)

        # get video using function from output.py
        output.create_video(simulation)

        # exits out as the conversion from images to video is done
        exit()

    # turn a collection of csvs into images and a video
    elif mode == 3:
        # create simulation instance
        simulation = parameters.Simulation(templates_path, name, path, mode, separator)

        # list the csv files in the values directory and sort them naturally
        csv_list = os.listdir(simulation.values_path)
        csv_list = natsort.natsorted(csv_list)

        # list the gradient files in the values directory and sort them naturally
        gradients_list = os.listdir(simulation.gradients_path)
        gradients_list = natsort.natsorted(gradients_list)

        # make sure the proper output directories for imaging exist
        output.initialize_outputs(simulation)

        # loop over all csvs defined in the template file
        for i in range(len(csv_list)):
            # updates the instance variables as they aren't updated by anything else
            simulation.current_step = i + 1

            # get the fgf4 values based on the saved numpy array
            simulation.fgf4_values = np.load(simulation.gradients_path + gradients_list[i])

            # opens the csv and create a list of the rows
            with open(simulation.values_path + csv_list[i], newline="") as file:
                # skip the first row as this is a header
                csv_rows = list(csv.reader(file))[1:]

            # updates the number of cells and adds that amount of vertices to the graphs
            simulation.number_cells = len(csv_rows)

            # turn all of the rows into a matrix
            cell_data = np.column_stack(csv_rows)

            # each row of the matrix will correspond to a cell value holder. the 2D arrays must be handled differently
            simulation.cell_locations = cell_data[0:3, :].transpose().astype(float)
            simulation.cell_radii = cell_data[3].astype(float)
            simulation.cell_motion = cell_data[4] == "True"
            simulation.cell_fds = cell_data[5:9, :].transpose().astype(float).astype(int)
            simulation.cell_states = cell_data[9].astype(str)

            # create image of simulation space
            output.step_image(simulation)

        # get a video of the images
        output.create_video(simulation)

        # exits out as the conversion from .csv to images/video is done
        exit()

    else:
        print("Incorrect mode")
        exit()

    # return the modified simulation instance
    return simulation


def check_name(name, output_path, separator, mode):
    """ renames the file if another simulation has the same name
        or checks to make sure such a simulation exists
    """
    # for a new simulation
    if mode == 0:
        # keeps the loop running until one condition is met
        while True:
            # if the path does not already exist, make that directory and break out of the loop
            try:
                os.mkdir(output_path + name)
                break

            # prompt to either rename or overwrite
            except OSError:
                print("Simulation with identical name: " + name)
                user = input("Would you like to overwrite that simulation? (y/n): ")
                if user == "n":
                    name = input("New name: ")
                elif user == "y":
                    # clear current directory to prevent another possible future errors
                    files = os.listdir(output_path + name)
                    for file in files:
                        path = output_path + name + separator + file
                        # remove file
                        if os.path.isfile(path):
                            os.remove(path)
                        # remove directory
                        else:
                            shutil.rmtree(path)
                    break
                else:
                    print("Either type ""y"" or ""n""")

    # this will look for an existing directory for modes other than 0
    else:
        # keeps the loop running until one condition is met
        while True:
            # see if the directory exists
            if os.path.isdir(output_path + name):
                break

            # if not prompt to change name or end the simulation
            else:
                print("No directory exists with name/path: " + output_path + name)
                user = input("Would you like to continue? (y/n): ")
                if user == "n":
                    exit()
                elif user == "y":
                    print(output_path)
                    user = input("Is the above path correct? (y/n): ")
                    if user == "n":
                        output_path = input("Type correct path:")
                    print(name)
                    user = input("Is the above name correct? (y/n): ")
                    if user == "n":
                        name = input("Type correct name:")
                else:
                    pass

    # return the updated name, directory, and path
    return name, output_path, output_path + name + separator
