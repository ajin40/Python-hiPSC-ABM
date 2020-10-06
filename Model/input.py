import random as r
import numpy as np
import csv
import os
import platform
import sys
import pickle
import igraph
import shutil
import natsort

import output
import functions


# used to hold all values necessary to the simulation based on the various modes
class Simulation:
    def __init__(self, templates_path, name, path, mode, separator):
        self.name = name    # name of the simulation, used to name files
        self.path = path    # path to the main simulation directory

        # these directories fall under the main simulation directory
        self.images_path = path + name + "_images" + separator    # path to image directory
        self.values_path = path + name + "_values" + separator    # path to step csv directory
        self.gradients_path = path + name + "_gradients" + separator    # path to gradient directory
        self.tda_path = path + name + "_tda" + separator    # path to TDA directory

        # ------------- general template file -------------------------
        # open the .txt file and get a list of the lines
        with open(templates_path + "general.txt") as general_file:
            general = general_file.readlines()

        # create instance variables based on template parameters
        self.parallel = eval(general[4][2:-3])
        self.end_step = int(general[7][2:-3])
        self.number_cells = int(general[10][2:-3])
        self.size = np.array(eval(general[13][2:-3]))

        # ------------- imaging template file -------------------------
        # open the .txt file and get a list of the lines
        with open(templates_path + "imaging.txt") as imaging_file:
            imaging = imaging_file.readlines()

        # create instance variables based on template parameters
        self.output_images = eval(imaging[4][2:-3])
        self.image_quality = int(imaging[8][2:-3])
        self.fps = float(imaging[11][2:-3])
        self.color_mode = eval(imaging[15][2:-3])
        self.output_gradient = eval(imaging[18][2:-3])

        # ------------- experimental template file -------------------------
        # open the .txt file and get a list of the lines
        with open(templates_path + "experimental.txt") as experimental_file:
            experimental = experimental_file.readlines()

        # create instance variables based on template parameters
        self.pluri_div_thresh = int(experimental[4][2:-3])
        self.diff_div_thresh = int(experimental[7][2:-3])
        self.pluri_to_diff = int(experimental[10][2:-3])
        self.death_thresh = int(experimental[13][2:-3])
        self.fds_thresh = int(experimental[16][2:-3])
        self.fgf4_thresh = int(experimental[19][2:-3])
        self.lonely_cell = int(experimental[22][2:-3])
        self.group = int(experimental[25][2:-3])
        self.guye_move = eval(experimental[28][2:-3])
        self.eunbi_move = eval(experimental[31][2:-3])
        self.max_concentration = float(experimental[34][2:-3])
        self.fgf4_move = eval(experimental[37][2:-3])
        self.output_tda = eval(experimental[40][2:-3])
        self.dox_value = float(experimental[43][2:-3])
        self.field = int(experimental[46][2:-3])

        # the following instance variables are only necessary for a new simulation. these include arrays for
        # storing the cell values, graphs for cell neighbors, and miscellaneous values.
        if mode == 0:
            # start the simulation at step 1
            self.beginning_step = 1

            # these arrays hold all values of the cells, each index corresponds to a cell.
            self.cell_locations = np.empty((self.number_cells, 3), dtype=float)
            self.cell_radii = np.empty(self.number_cells, dtype=float)
            self.cell_motion = np.empty(self.number_cells, dtype=bool)
            self.cell_fds = np.empty((self.number_cells, 4), dtype=float)
            self.cell_states = np.empty(self.number_cells, dtype='<U14')
            self.cell_diff_counter = np.empty(self.number_cells, dtype=int)
            self.cell_div_counter = np.empty(self.number_cells, dtype=int)
            self.cell_death_counter = np.empty(self.number_cells, dtype=int)
            self.cell_fds_counter = np.empty(self.number_cells, dtype=int)
            self.cell_motility_force = np.empty((self.number_cells, 3), dtype=float)
            self.cell_jkr_force = np.empty((self.number_cells, 3), dtype=float)
            self.cell_nearest_gata6 = np.empty(self.number_cells)
            self.cell_nearest_nanog = np.empty(self.number_cells)
            self.cell_nearest_diff = np.empty(self.number_cells)
            self.cell_highest_fgf4 = np.empty((self.number_cells, 3))
            self.cell_nearest_cluster = np.empty(self.number_cells)
            self.cell_dox_value = np.empty(self.number_cells, dtype=float)
            self.cell_rotation = np.empty(self.number_cells, dtype=float)

            # the names of the cell arrays should be in this list as this will be used to delete and add cells
            # automatically without the need to update add/delete functions
            self.cell_array_names = ["cell_locations", "cell_radii", "cell_motion", "cell_fds", "cell_states",
                                     "cell_diff_counter", "cell_div_counter", "cell_death_counter", "cell_fds_counter",
                                     "cell_motility_force", "cell_jkr_force", "cell_nearest_gata6",
                                     "cell_nearest_nanog", "cell_nearest_diff", "cell_highest_fgf4",
                                     "cell_nearest_cluster", "cell_dox_value", "cell_rotation"]

            # holds all indices of cells that will divide at a current step or be removed at that step
            self.cells_to_divide, self.cells_to_remove = np.array([], dtype=int), np.array([], dtype=int)

            # create graphs used to all cells and their neighbors, initialize them with the number of cells
            self.neighbor_graph, self.jkr_graph = igraph.Graph(self.number_cells), igraph.Graph(self.number_cells)

            # min and max radius lengths are used to calculate linear growth of the radius over time in 2D
            self.max_radius = 0.000005
            self.min_radius = self.max_radius / 2 ** 0.5
            self.pluri_growth = (self.max_radius - self.min_radius) / self.pluri_div_thresh
            self.diff_growth = (self.max_radius - self.min_radius) / self.diff_div_thresh

            # the spatial resolution of the space
            self.spat_res = 0.00001
            self.spat_res2 = self.spat_res ** 2

            # the diffusion constant for the molecule gradients and the radius of search for diffusion points
            self.diffuse = 0.0000000000357
            self.diffuse_radius = self.spat_res * 0.707106781187

            # the temporal resolution of the diffusion 60 seconds, which is considered stable for above parameters
            self.diffuse_dt = 0.7

            # calculate the size of the array holding the diffusion points
            self.gradient_size = np.ceil(self.size / self.spat_res).astype(int) + np.ones(3, dtype=int)

            # create a primary array for the diffusion points and one to add in a step-wise fashion
            self.fgf4_values = np.zeros(self.gradient_size)
            self.fgf4_alt = np.zeros(self.gradient_size)

            # much like the cell arrays add any gradient names to list this so that a diffusion function can
            # act on them automatically
            self.extracellular_names = ["fgf4_values", "fgf4_alt"]

            # the time in seconds for an entire step and the incremental movement time
            self.step_dt = 1800
            self.move_dt = 200

            # used to hold the run times of functions
            self.function_times = dict()


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
        simulation = Simulation(templates_path, name, path, mode, separator)

        # create directory within the simulation output directory for template files and copy them there
        new_template_direct = simulation.path + simulation.name + "_templates"
        os.mkdir(new_template_direct)
        for template_name in os.listdir(templates_path):
            shutil.copy(templates_path + template_name, new_template_direct)

        # go through all cell arrays giving initial parameters of the cells
        for i in range(simulation.number_cells):
            n = simulation.field - 1
            div_counter = r.randint(0, simulation.pluri_div_thresh)
            simulation.cell_locations[i] = np.array([r.random() * simulation.size[0],
                                                     r.random() * simulation.size[1],
                                                     r.random() * simulation.size[2]])
            simulation.cell_radii[i] = simulation.min_radius + simulation.pluri_growth * div_counter
            simulation.cell_motion[i] = True
            simulation.cell_fds[i] = np.array([r.randint(0, n), r.randint(0, n), 0, r.randint(1, n)])
            simulation.cell_states[i] = "Pluripotent"
            simulation.cell_diff_counter[i] = r.randint(0, simulation.pluri_to_diff)
            simulation.cell_div_counter[i] = div_counter
            simulation.cell_death_counter[i] = r.randint(0, simulation.death_thresh)
            simulation.cell_fds_counter[i] = r.randint(0, simulation.fds_thresh)
            simulation.cell_motility_force[i] = np.zeros(3, dtype=float)
            simulation.cell_jkr_force[i] = np.zeros(3, dtype=float)
            simulation.cell_nearest_gata6[i] = np.nan
            simulation.cell_nearest_nanog[i] = np.nan
            simulation.cell_nearest_diff[i] = np.nan
            simulation.cell_highest_fgf4[i] = np.array([np.nan, np.nan, np.nan])
            simulation.cell_nearest_cluster[i] = np.nan
            simulation.cell_dox_value[i] = r.random()

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
        simulation = Simulation(templates_path, name, path, mode, separator)

        # get video using function from output.py
        output.create_video(simulation)

        # exits out as the conversion from images to video is done
        exit()

    # turn a collection of csvs into images and a video
    elif mode == 3:
        # create simulation instance
        simulation = Simulation(templates_path, name, path, mode, separator)

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
