import numpy as np
import csv
import os
import platform
import sys
import pickle
import shutil
import natsort
import getopt

import output
import parameters


def setup():
    """ reads files for the starting the model and then
        determine which simulation mode to use
    """
    # keep track of the file separator to use
    if platform.system() == "Windows":
        separator = "\\"  # Windows
    else:
        separator = "/"  # not Windows

    # open the paths.txt file containing the locations of the template files
    with open('paths.txt', 'r') as file:
        lines = file.readlines()

    # get the path to the template file directory and make sure the model can use it
    templates_path = lines[7].strip()
    templates_path = check_path(templates_path, separator)

    # get the path to the output directory and make sure the model can use it
    output_path = lines[13].strip()
    output_path = check_path(output_path, separator)

    # hold the possible modes for the model, used to check that mode exists
    possible_modes = [0, 1, 2, 3, 4, 5]

    # get any command line options for the model
    inputs, other_args = getopt.getopt(sys.argv[1:], "n:m:")

    # go through the inputs getting the options
    for opt, arg in inputs:
        # if the "-n" name option, set the variable name
        if opt == "-n":
            name = arg

        # if the "-m" mode option, set the variable mode
        elif opt == "-m":
            mode = int(arg)

        # if some other option
        else:
            print("Unknown option: ", opt)

    # if the name variable has not been initialized, run the text-based GUI
    try:
        name
    except NameError:
        # get the name of the simulation
        while True:
            name = input("What is the \"name\" of the simulation? Type \"help\" for more information: ")
            if name == "help":
                print("Type the name of the simulation without quotes and not as a path.\n")
            else:
                break

    # if the mode variable has not been initialized, run the text-based GUI
    try:
        mode
    except NameError:
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
                    # get the mode as an integer
                    mode = int(mode)

                    # make sure mode exists, break the loop if it does
                    if mode in possible_modes:
                        break
                    else:
                        print("Mode does not exist. See possible modes: " + str(possible_modes))

                # if not an integer
                except ValueError:
                    print("\"mode\" should be an integer")

    # check the name of the simulation based on the mode
    name, output_path, path = check_name(name, output_path, separator, mode)

    # new simulation
    if mode == 0:
        # create instance of Simulation class
        simulation = parameters.Simulation(templates_path, name, path, mode, separator)

        # copy model files and template parameters
        shutil.copytree(os.getcwd(), path + "Model_copy")

        # make a directories for outputting images, csvs, gradients, etc.
        output.initialize_outputs(simulation)

    # continuation of previous simulation
    elif mode == 1:
        # open the temporary pickled simulation and update the beginning step and the end step
        with open(path + name + "_temp.pkl", "rb") as temp_file:
            simulation = pickle.load(temp_file)
            simulation.beginning_step = simulation.current_step + 1

        # make sure the proper output directories exist
        output.initialize_outputs(simulation)

    # images to video
    elif mode == 2:
        # create instance of Simulation class used to get imaging information
        simulation = parameters.Simulation(templates_path, name, path, mode, separator)

        # get video using function from output.py
        output.create_video(simulation)

        # exits out as the conversion from images to video is done
        exit()

    # CSVs to images/video
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

    # zip a simulation directory
    elif mode == 4:
        print("Compressing: " + name)
        shutil.make_archive(path[:-1], 'zip', path)
        print("Done!")
        exit()

    # unzip a simulation directory
    elif mode == 5:
        print("Unpacking: " + name)
        shutil.unpack_archive(path[:-1] + ".zip", output_path)

    # return the simulation based on the simulation mode
    return simulation


def check_path(path, separator):
    """ checks the path to make sure the directory
        exists and adds the necessary separator
    """
    if not os.path.isdir(path):
        # raise error if directory doesn't exist
        raise Exception("Path: " + path + " to templates directory does not exist.")
    else:
        # if path doesn't end with separator, add one
        if path[-1] != separator:
            path += separator

    return path


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
                print("Simulation with identical name: " + '\033[31m' + name + '\033[0m')
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
                print("No directory exists with name/path: " + '\033[31m' + output_path + name + '\033[0m')
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
