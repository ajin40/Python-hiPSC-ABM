import os
import sys
import pickle
import shutil
import getopt

import output
import parameters


def setup():
    """ Takes parameters from files, the command line,
        and/or a text-based GUI to start the model.
    """
    # get the path separator for the OS and open the paths.txt file containing the locations of the template files
    separator = os.path.sep
    with open('paths.txt', 'r') as file:
        lines = file.readlines()

    # get the path to the template file directory which is used to provide some initial simulation parameters
    templates_path = os.path.abspath("templates") + separator

    # get the path to the output directory where each simulation directory is created and make sure the
    # output directory path exists and has separator at the end
    output_path = lines[7].strip()
    if not os.path.isdir(output_path):
        # raise error if directory doesn't exist
        raise Exception("Path: " + output_path + " to templates directory does not exist.")
    else:
        # if path doesn't end with separator, add one
        if output_path[-1] != separator:
            output_path += separator

    # get any command-line options for the model, "n:m:" allows for the -n and -m options
    options, args = getopt.getopt(sys.argv[1:], "n:m:")  # first argument is "python" so avoid that

    # go through the inputs getting the options
    for option, value in options:
        # if the "-n" name option, set the variable name
        if option == "-n":
            name = value

        # if the "-m" mode option, set the variable mode
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

    # hold the possible modes for the model, used to check that a particular mode exists
    possible_modes = [0, 1, 2, 3]

    # if the mode variable has not been initialized by the command-line, run the text-based GUI to get it
    if 'mode' not in locals() or mode not in possible_modes:
        while True:
            # prompt for the mode
            mode = input("What is the \"mode\" of the simulation? Type \"help\" for more information: ")

            # keep running if "help" is typed
            if mode == "help":
                print("\nHere are the following modes:\n0: New simulation\n1: Continuation of past simulation\n"
                      "2: Turn simulation images to video\n3: Zip previous simulation\n")
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

    # check the name of the simulation based on the mode and return a path to the simulation directory
    name, path = check_name(name, output_path, separator, mode)

    # create a paths object that holds any important paths
    paths = output.Paths(name, path, templates_path, separator)

    # -------------- new simulation ---------------------------
    if mode == 0:
        # create instance of Simulation class
        simulation = parameters.Simulation(paths, name, mode)

        # copy model files and template parameters
        shutil.copytree(os.getcwd(), path + name + "_copy")

    # -------------- continuation of previous simulation ---------------------------
    elif mode == 1:
        # get the new end step of the simulation
        end_step = int(input("What is the final step of this continued simulation? "))

        # open the last pickled simulation and update the beginning step
        with open(path + name + "_temp" + ".pkl", "rb") as temp_file:
            simulation = pickle.load(temp_file)
            simulation.beginning_step = simulation.current_step + 1
            simulation.end_step = end_step
            simulation.mode = mode

    # -------------- images to video ---------------------------
    elif mode == 2:
        # create instance of Simulation class used to get imaging information
        simulation = parameters.Simulation(paths, name, mode)

        # get video using function from output.py
        output.create_video(simulation)

        # exits out as the conversion from images to video is done
        exit()

    # -------------- zip a simulation directory --------------
    elif mode == 3:
        print("Compressing: " + name)

        # remove separator of the path to the simulation directory
        simulation_path = path[:-1]

        # zip a copy of the directory and save it to the same simulation directory
        shutil.make_archive(simulation_path, 'zip', path)
        print("Done!")
        exit()

    # return the simulation based on the simulation mode
    return simulation


def check_name(name, output_path, separator, mode):
    """ renames the file if another simulation has the same name
        or checks to make sure such a simulation exists
    """
    # keeps the loop running
    while True:
        if mode == 0:
            # try to make a new simulation directory
            try:
                os.mkdir(output_path + name)
                break

            # if simulation directory with same name already exists
            except OSError:
                print("Simulation with identical name: " + '\033[31m' + name + '\033[0m')

                # get user input for overwriting previous simulation
                user = input("Would you like to overwrite that simulation? (y/n): ")

                # if no overwrite, get new simulation name
                if user == "n":
                    name = input("New name: ")

                # overwrite by deleting all files in previous directory
                elif user == "y":
                    # clear current directory to prevent another possible future errors
                    files = os.listdir(output_path + name)
                    for file in files:
                        # path to each file
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

        else:
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
                    user = input("Is the above path to the output directory correct? (y/n): ")
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
