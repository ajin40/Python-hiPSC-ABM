import os
import sys
import pickle
import shutil
import getopt

import output
import parameters


def setup():
    """ reads files for the starting the model and then
        determine which simulation mode to use
    """
    # get the path separator for the OS and open the paths.txt file containing the locations of the template files
    separator = os.path.sep
    with open('paths.txt', 'r') as file:
        lines = file.readlines()

    # get the path to the template file directory which is used to provide some initial simulation parameters
    templates_path = lines[7].strip()
    templates_path = check_path(templates_path, separator)    # and make sure the model can use the path

    # get the path to the output directory which is where the simulation directory is outputted to
    output_path = lines[13].strip()
    output_path = check_path(output_path, separator)    # and make sure the model can use the path

    # get any command line options for the model, "n:m:" allows for the -n and -m options
    options, args = getopt.getopt(sys.argv[1:], "n:m:")    # first argument is "python" so avoid that

    # go through the inputs getting the options
    for option, value in options:
        # if the "-n" name option, set the variable name
        if option == "-n":
            name = value

        # if the "-m" mode option, set the variable mode
        elif option == "-m":
            mode = int(value)    # turn from string to int

        # if some other option
        else:
            print("Unknown option: ", option)

    # if the name variable has not been initialized, run the text-based GUI to get it
    try:
        name
    except NameError:
        # run loop until broken
        while True:
            name = input("What is the \"name\" of the simulation? Type \"help\" for more information: ")
            if name == "help":
                print("Type the name of the simulation without quotes and not as a path.\n")
            else:
                break

    # hold the possible modes for the model, used to check that mode exists
    possible_modes = [0, 1, 2, 3]

    # if the mode variable has not been initialized, run the text-based GUI to get it
    try:
        mode
    except NameError:
        # run loop until broken
        while True:
            mode = input("What is the \"mode\" of the simulation? Type \"help\" for more information: ")
            if mode == "help":
                print("\nHere are the following modes:")
                print("new simulation: 0")
                print("continuation of past simulation: 1")
                print("turn simulation images to video: 2")
                print("turn simulation into zip: 3\n")
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
    name, path = check_name(name, output_path, separator, mode)

    # create paths object with holds any important paths
    paths = output.Paths(name, path, templates_path, separator)

    # new simulation
    if mode == 0:
        # create instance of Simulation class
        simulation = parameters.Simulation(paths, name, mode)

        # copy model files and template parameters
        shutil.copytree(os.getcwd(), path + name + "_copy")

    # continuation of previous simulation
    elif mode == 1:
        # get the new end step of the simulation
        end_step = int(input("What is the final step of this continued simulation? "))

        # open the last pickled simulation and update the beginning step
        with open(path + name + "_temp" + ".pkl", "rb") as temp_file:
            simulation = pickle.load(temp_file)
            simulation.beginning_step = simulation.current_step + 1
            simulation.end_step = end_step
            simulation.mode = mode

    # images to video
    elif mode == 2:
        # create instance of Simulation class used to get imaging information
        simulation = parameters.Simulation(paths, name, mode)

        # get video using function from output.py
        output.create_video(simulation)

        # exits out as the conversion from images to video is done
        exit()

    # zip a simulation directory
    elif mode == 3:
        print("Compressing: " + name)
        shutil.make_archive(path[:-1], 'zip', path)
        print("Done!")
        exit()

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
    return name, output_path + name + separator
