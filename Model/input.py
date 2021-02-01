import os
import sys
import pickle
import shutil
import getopt

import output
import parameters
import run


def start():
    """ Based on the desired mode in which the model is to be
        run, this will control that setup.
    """
    # get the path separator and the absolute path to the template file directory
    separator = os.path.sep
    templates_path = os.path.abspath("templates") + separator

    # get the path to the directory where simulations are outputted and the name/mode for the simulation
    output_path = output_dir(separator)
    possible_modes = [0, 1, 2, 3, 4]    # hold possible model modes
    name, mode = get_namemode(output_path, separator, possible_modes)

    # create path to simulation directory and make Paths object for storing important paths
    main_path = output_path + name + separator
    paths = output.Paths(name, main_path, templates_path, separator)

    # -------------------------- new simulation ---------------------------
    if mode == 0:
        # copy model files to simulation output, ignore pycache files
        copy_name = main_path + name + "_copy"
        shutil.copytree(os.getcwd(), copy_name, ignore=shutil.ignore_patterns("__pycache__"))

        # create Simulation object
        simulation = parameters.Simulation(paths, name)

        # add cell arrays to Simulation object and run the model
        run.setup_cells(simulation)
        run.steps(simulation)

    # ---------------- continuation of previous simulation ----------------
    elif mode == 1:
        # load previous Simulation object instead of creating new Simulation object
        file_name = main_path + name + "_temp" + ".pkl"
        with open(file_name, "rb") as file:
            simulation = pickle.load(file)

        # update the following instance variables
        simulation.paths = paths  # change paths object for cross platform compatibility
        simulation.beginning_step = simulation.current_step + 1    # start one step later
        simulation.end_step = int(input("What is the final step of this continued simulation? "))

        # run the model
        run.steps(simulation)

    # ------------------------- images to video ---------------------------
    elif mode == 2:
        # create Simulation object used to get imaging and path information
        simulation = parameters.Simulation(paths, name)

        # make the video
        output.create_video(simulation)

    # --------------------- zip a simulation directory --------------------
    elif mode == 3:
        # print statement and remove the separator of the path to the simulation directory
        print("Compressing: " + name)
        simulation_dir = main_path[:-1]

        # zip a copy of the directory and save it to the output directory
        shutil.make_archive(simulation_dir, 'zip', root_dir=output_path, base_dir=str(name))
        print("Done!")

    # ----------------- extract a simulation directory zip ----------------
    elif mode == 4:
        # print statement and get name for .zip file
        print("Extracting: " + name)
        zip_file = main_path[:-1] + ".zip"

        # unpack the directory into the output directory
        shutil.unpack_archive(zip_file, output_path)
        print("Done!")


def output_dir(separator):
    """ Get the path to the output directory. If this directory
        does not exist yet make it and update the paths.txt file.
    """
    # read the paths.txt file which should be the path to the output directory
    with open("paths.txt", "r") as file:
        lines = file.readlines()
    output_path = lines[14].strip()   # remove whitespace

    # keep running until output directory exists
    while not os.path.isdir(output_path):
        # prompt user input
        print("No directory: \"" + output_path + "\" for outputting simulations exists!")
        user = input("Would you still like to use this directory? If yes, then that directory will be created, "
                     "otherwise you can specify another path. (y/n): ")

        # if not making this directory
        if user == "n":
            # get new path to output directory
            output_path = input("Correct path (absolute) to output directory: ")

            # update paths.txt file with new output directory path
            with open("paths.txt", "w") as file:
                lines[14] = output_path + "\n"
                file.writelines(lines)

        # if yes, make the directory
        elif user == "y":
            os.mkdir(output_path)
            break

        else:
            print('Either type "y" or "n"')

    # if path doesn't end with separator, add it
    if output_path[-1] != separator:
        output_path += separator

    return output_path


def get_namemode(output_path, separator, possible_modes):
    """ This function will get the name and mode for the simulation
        either from the command line or a text-based GUI.
    """
    # get any command-line options for the model, "n:m:" allows for using -n and -m
    options, args = getopt.getopt(sys.argv[1:], "n:m:")  # first argument is "python" so avoid that

    # go through the inputs, getting the options
    for option, value in options:
        # if the "-n" name option, set the name variable
        if option == "-n":
            name = value

        # if the "-m" mode option, set the mode variable
        elif option == "-m":
            mode = int(value)  # turn from string to int

        # if some other option, raise error
        else:
            raise Exception("Unknown command-line option: " + option)

    # if the name variable has not been initialized by the command-line, run the text-based GUI to get it
    if 'name' not in locals():
        while True:
            # prompt for the name
            name = input("What is the \"name\" of the simulation? Type \"help\" for more information: ")

            # keep running if "help" is typed
            if name == "help":
                print("Type the name of the simulation (not a path)\n")
            else:
                break

    # if the mode variable has not been initialized by the command-line, run the text-based GUI to get it
    if 'mode' not in locals() or mode not in possible_modes:
        while True:
            # prompt for the mode
            mode = input("What is the \"mode\" of the simulation? Type \"help\" for more information: ")

            # keep running if "help" is typed
            if mode == "help":
                print("\nHere are the following modes:\n0: New simulation\n1: Continuation of past simulation\n"
                      "2: Turn simulation images to video\n3: Zip previous simulation\n4: Unzip a simulation file\n"
                      "5: Calculate persistent homology\n")
            else:
                try:
                    # get the mode as an integer make sure mode exists, break the loop if it does
                    mode = int(mode)
                    if mode in possible_modes:
                        break
                    else:
                        print("Mode does not exist. See possible modes: " + str(possible_modes))

                # if not an integer
                except ValueError:
                    print("\"mode\" should be an integer")

    # check that the simulation name is valid based on the mode
    while True:
        # if a new simulation
        if mode == 0:
            # see if the directory exists
            if os.path.isdir(output_path + name):
                # get user input for overwriting previous simulation
                print("Simulation already exists with name: " + name)
                user = input("Would you like to overwrite that simulation? (y/n): ")

                # if no overwrite, get new simulation name
                if user == "n":
                    name = input("New name: ")

                # overwrite by deleting all files/folders in previous directory
                elif user == "y":
                    # clear current directory to prevent another possible future errors
                    files = os.listdir(output_path + name)
                    for file in files:
                        # path to each file/folder
                        path = output_path + name + separator + file

                        # delete the file/folder
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    break
                else:
                    # inputs should either be "y" or "n"
                    print('Either type "y" or "n"')
            else:
                # if does not exist, make directory
                os.mkdir(output_path + name)
                break

        # extract simulation zip mode
        elif mode == 4:
            if os.path.isdir(output_path + name):
                raise Exception(name + " already exists in " + output_path)
            elif os.path.isfile(output_path + name + ".zip"):
                break
            else:
                raise Exception(name + ".zip does not exist in " + output_path)

        # previous simulation output directory modes
        else:
            # if the directory exists, break loop
            if os.path.isdir(output_path + name):
                break

            else:
                print("No directory exists with name/path: " + output_path + name)
                name = input("Please type the correct name of the simulation or type \"exit\" to exit: ")
                if name == "exit":
                    exit()

    return name, mode


def get_parameter(path, line_number, dtype):
    """ Gets the parameter as a string from the lines of the
        template file. Used for Simulation instance variables.
    """
    # make an attribute with name as template file path and value as a list of the file lines (reduces file opening)
    if not hasattr(get_parameter, path):
        with open(path) as file:
            get_parameter.path = file.readlines()

    # get the right line based on the line numbers not Python indexing
    line = get_parameter.path[line_number - 1]

    # find the indices of the pipe characters
    begin = line.find("|")
    end = line.find("|", begin + 1)

    # raise error if not a pair of pipe characters
    if begin == -1 or end == -1:
        raise Exception("Please use pipe characters to specify template file parameters. Example: | (value) |")

    # return a slice of the line that is the string representation of the parameter and remove any whitespace
    parameter = line[(begin + 1):end].strip()

    # convert the parameter from string to desired data type
    if dtype == str:
        pass
    elif dtype == tuple or dtype == list or dtype == dict:
        # tuple() list() dict() will not produce desired result, use eval() instead
        parameter = eval(parameter)
    elif dtype == bool:
        # handle potential inputs for booleans
        if parameter in ["True", "true", "T", "t", "1"]:
            parameter = True
        elif parameter in ["False", "false", "F", "f", "0"]:
            parameter = False
        else:
            raise Exception("Invalid value for bool type")
    else:
        # for float and int type
        parameter = dtype(parameter)

    # get the parameter by removing the pipe characters and any whitespace
    return parameter
