import os
import sys
import pickle
import shutil
import getopt
import psutil

from backend import commandline_param
import outputs
import parameters


def start():
    """ Based on the desired mode in which the model is to be
        run, this start and setup the model.
    """
    # get the path separator and the absolute path to the template file directory
    separator = os.path.sep
    templates_path = os.path.abspath("templates") + separator

    # get the path to the directory where simulations are outputted and the name/mode for the simulation
    output_path = output_dir(separator)
    possible_modes = [0, 1, 2, 3]    # hold possible model modes
    name, mode, final_step = start_params(output_path, separator, possible_modes)

    # create path to simulation directory and make Paths object for storing important paths
    main_path = output_path + name + separator
    paths = backend.Paths(name, main_path, templates_path, separator)

    # -------------------------- new simulation ---------------------------
    if mode == 0:
        # copy model files to simulation output, ignore pycache files
        copy_name = main_path + name + "_copy"
        shutil.copytree(os.getcwd(), copy_name, ignore=shutil.ignore_patterns("__pycache__"))

        # create Simulation object
        simulation = parameters.Simulation(paths, name)

        # add cell arrays to Simulation object and run the model
        simulation.agent_initials()
        simulation.steps()

    # ---------------- continuation of previous simulation ----------------
    elif mode == 1:
        # load previous Simulation object instead of creating new Simulation object
        file_name = main_path + name + "_temp" + ".pkl"
        with open(file_name, "rb") as file:
            simulation = pickle.load(file)

        # update the following instance variables
        simulation.paths = paths  # change paths object for cross platform compatibility
        simulation.beginning_step = simulation.current_step + 1    # start one step later
        simulation.end_step = final_step    # update final step

        # run the model
        simulation.steps()

    # ------------------------- images to video ---------------------------
    elif mode == 2:
        # create Simulation object used to get imaging and path information
        simulation = parameters.Simulation(paths, name)

        # make the video
        simulation.create_video()

    # --------------------- zip a simulation directory --------------------
    elif mode == 3:
        # print statement and remove the separator of the path to the simulation directory
        print('Compressing "' + name + '" simulation...')
        simulation_dir = main_path[:-1]

        # zip a copy of the directory and save it to the output directory
        shutil.make_archive(simulation_dir, 'zip', root_dir=output_path, base_dir=str(name))
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
        print("Simulation output directory: \"" + output_path + "\" does not exist!")
        user = input('Do you want to make this directory? If "n" you can specify the correct path (y/n): ')

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
            os.makedirs(output_path)
            break

        else:
            print('Either type "y" or "n"')

    # if path doesn't end with separator, add it
    if output_path[-1] != separator:
        output_path += separator

    return output_path


def start_params(output_path, separator, possible_modes):
    """ This function will get the name and mode for the simulation
        either from the command line or a text-based GUI.
    """
    # try to get the name and mode from the command line
    name = commandline_param("-n", str)
    mode = commandline_param("-m", int)

    # if the name variable has not been initialized by the command-line, run the text-based UI to get it
    if name is None:
        while True:
            # prompt for the name
            name = input("What is the \"name\" of the simulation? Type \"help\" for more information: ")

            # keep running if "help" is typed
            if name == "help":
                print("\nType the name of the simulation (not a path).\n")
            else:
                break

    # if the mode variable has not been initialized by the command-line, run the text-based UI to get it
    if mode is None or mode not in possible_modes:
        while True:
            # prompt for the mode
            mode = input("What is the \"mode\" of the simulation? Type \"help\" for more information: ")
            print()

            # keep running if "help" is typed
            if mode == "help":
                print("Here are the following modes:\n0: New simulation\n1: Continuation of past simulation\n"
                      "2: Turn simulation images to video\n3: Zip previous simulation\n")
            else:
                try:
                    # get the mode as an integer make sure mode exists, break the loop if it does
                    mode = int(mode)
                    if mode in possible_modes:
                        break
                    else:
                        print("Mode does not exist, see possible modes: " + str(possible_modes) + "\n")

                # if not an integer
                except ValueError:
                    print("Input: \"mode\" should be an integer.\n")

    # if the final_step variable has not been initialized by the command-line, run the text-based GUI to get it
    if 'final_step' not in locals():
        # if continuation mode
        if mode == 1:
            while True:
                # prompt for the final step number
                final_step = input("What is the final step of this continued simulation? Type \"help\" for more"
                                   " information: ")
                print()

                # keep running if "help" is typed
                if final_step == "help":
                    print("Enter the new step number that will be the last step of the simulation.\n")
                else:
                    try:
                        # get the final step as an integer, break the loop if conversion is successful
                        final_step = int(final_step)
                        break

                    # if not an integer
                    except ValueError:
                        print("Input: \"final step\" should be an integer.\n")

        # if not continuation mode, give final_step default value
        else:
            final_step = None

    # check that the simulation name is valid based on the mode
    while True:
        # if a new simulation
        if mode == 0:
            # see if the directory exists
            if os.path.isdir(output_path + name):
                # get user input for overwriting previous simulation
                print("Simulation already exists with name: " + name)
                user = input("Would you like to overwrite that simulation? (y/n): ")
                print()

                # if no overwrite, get new simulation name
                if user == "n":
                    name = input("New name: ")
                    print()

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

        # previous simulation output directory modes
        else:
            # if the directory exists, break loop
            if os.path.isdir(output_path + name):
                break

            else:
                print("No directory exists with name/path: " + output_path + name)
                name = input("Please type the correct name of the simulation or type \"exit\" to exit: ")
                print()
                if name == "exit":
                    exit()

    return name, mode, final_step


# Only start the model if this file is being run directly.
if __name__ == "__main__":
    # get process (run.py) and set priority to high
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)

    # start the model
    start()
