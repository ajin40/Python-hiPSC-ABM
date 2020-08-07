import random as r
import numpy as np
import csv
import cv2
import os
import platform
import shutil
import sys
import webbrowser

import output
import igraph


# used to hold all values necessary to the simulation as it moves from one step to the next
class Simulation:
    def __init__(self, template_location):
        # open the .txt template file that contains the initial parameters
        with open(template_location) as template_file:
            lines = template_file.readlines()

        # the following lines correspond to lines of template file, so anything implemented to the template file
        # can be added to the class initialization.
        # general parameters
        self.name = lines[8][2:-3]  # the name of the simulation, used to name files in the output directory
        self.output_direct = lines[11][2:-3]   # the output directory where simulation is placed
        self.parallel = eval(lines[14][2:-3]) # whether the model is using parallel GPU processing for certain functions
        self.size = np.array([0.001, 0.001, 0.0])  # the dimensions of the space (in meters) the cells reside in
        self.num_GATA6 = int(lines[20][2:-3])   # the number of GATA6 high cells to begin the simulation
        self.num_NANOG = int(lines[23][2:-3])   # the number of NANOG high cells to being the simulation

        # modes
        self.output_csvs = eval(lines[41][2:-3])  # whether or not to produce csvs with cell information
        self.output_images = eval(lines[38][2:-3])   # whether or not to produce images
        self.continuation = eval(lines[44][2:-3])   # continuation of a previous simulation
        self.csv_to_images = eval(lines[47][2:-3])  # turn a collection of csvs to images
        self.images_to_video = eval(lines[50][2:-3])    # turn a collection of images into a video

        # timing
        self.end_step = int(lines[60][2:-3])   # the last step of a simulation
        self.time_step_value = 1800
        self.fds_thresh = int(lines[66][2:-3])  # the threshold (in steps) for updating the finite dynamical system
        self.pluri_div_thresh = int(lines[69][2:-3])  # the division threshold (in steps) of a pluripotent cell
        self.pluri_to_diff = int(lines[72][2:-3])  # the differentiation threshold (in steps) of a pluripotent cell
        self.diff_div_thresh = int(lines[75][2:-3])  # the division threshold (in steps) of a differentiated cell
        self.death_thresh = int(lines[78][2:-3])  # the death threshold (in steps) for cell death

        # intercellular
        self.lonely_cell = int(lines[93][2:-3])  # if the number of neighbors is below this threshold, a cell is lonely
        self.contact_inhibit = int(lines[96][2:-3])  # if the number of neighbors is below this threshold, no inhibition
        self.move_thresh = int(lines[99][2:-3])  # if the number of neighbors is above this threshold, inhibit motion
        self.diff_surround = int(lines[102][2:-3])  # the number of diff cells needed to help induce differentiation

        # imaging
        self.output_images = eval(lines[38][2:-3])   # whether or not to produce images
        self.image_quality = eval(lines[135][2:-3])    # the output image/video dimensions in pixels
        self.fps = float(lines[138][2:-3])   # the frames per second of the video produced
        self.background_color = eval(lines[141][2:-3])    # the background space color
        self.color_mode = eval(lines[144][2:-3])   # used to vary which method of coloring used
        self.output_gradient = eval(lines[189][2:-3])   # output an image of the extracellular gradient

        # miscellaneous/experimental
        self.stochastic = eval(lines[168][2:-3])    # if initial fds variables are stochastic
        self.group = int(lines[171][2:-3])   # the number of cells introduced into or removed from the space at once
        self.guye_move = eval(lines[174][2:-3])    # whether or not to use the Guye method of cell motility
        self.diffuse_radius = float(lines[177][2:-3])   # the radius of search of diffusion points
        self.max_fgf4 = float(lines[180][2:-3])  # the maximum amount of fgf4 at a diffusion point
        self.eunbi_move = eval(lines[183][2:-3])    # use Eunbi's model for movement
        self.fgf4_move = eval(lines[186][2:-3])     # use FGF4 concentrations for NANOG high movements

        # check that the name and path from the template are valid
        self.path = check_name(self, template_location)

        # holds the current number of cells, step, and time when a step started (used for tracking efficiency)
        self.number_cells = 0
        self.current_step = self.beginning_step
        self.step_start = float()

        # these arrays hold all values of the cells, each index corresponds to a cell.
        self.cell_locations = np.empty((0, 3), dtype=float)    # holds every cell's location vector in the space
        self.cell_radii = np.empty((0, 1), dtype=float)     # holds every cell's radius
        self.cell_motion = np.empty((0, 1), dtype=bool)    # holds every cell's boolean for being in motion or not
        self.cell_fds = np.empty((0, 4), dtype=float)    # holds every cell's values for the fds, currently 4
        self.cell_states = np.empty((0, 1), dtype='<U14')    # holds every cell's state pluripotent or differentiated
        self.cell_diff_counter = np.empty((0, 1), dtype=int)    # holds every cell's differentiation counter
        self.cell_div_counter = np.empty((0, 1), dtype=int)    # holds every cell's division counter
        self.cell_death_counter = np.empty((0, 1), dtype=int)    # holds every cell's death counter
        self.cell_fds_counter = np.empty((0, 1), dtype=int)    # holds every cell's finite dynamical system counter
        self.cell_motility_force = np.empty((0, 3), dtype=float)    # holds every cell's motility force vector
        self.cell_jkr_force = np.empty((0, 3), dtype=float)    # holds every cell's JKR force vector
        self.cell_nearest_gata6 = np.empty((0, 1))  # holds index of nearest gata6 high neighbor
        self.cell_nearest_nanog = np.empty((0, 1))  # holds index of nearest nanog high neighbor
        self.cell_nearest_diff = np.empty((0, 1))  # holds index of nearest differentiated neighbor
        self.cell_highest_fgf4 = np.empty((0, 3))  # holds the location of highest fgf4 point

        self.cell_array_names = ["cell_locations", "cell_radii", "cell_motion", "cell_fds", "cell_states",
                                 "cell_diff_counter", "cell_div_counter", "cell_death_counter", "cell_fds_counter",
                                 "cell_motility_force", "cell_jkr_force", "cell_nearest_gata6", "cell_nearest_nanog",
                                 "cell_nearest_diff", "cell_highest_fgf4"]

        # holds the run time for key functions to track efficiency. each step these are outputted to the CSV file.
        self.update_diffusion_time = float()
        self.check_neighbors_time = float()
        self.nearest_time = float()
        self.cell_motility_time = float()
        self.cell_update_time = float()
        self.update_queue_time = float()
        self.handle_movement_time = float()
        self.jkr_neighbors_time = float()
        self.get_forces_time = float()
        self.apply_forces_time = float()

        # extracellular
        self.dx = 0.00001
        self.dy = 0.00001
        self.dz = 1

        # movement/physical
        self.move_time_step = 200
        self.max_radius = 0.000005

        # neighbor graph is used to locate cells that are in close proximity, while the JKR graph holds adhesion bonds
        # between cells that are either currently overlapping or still maintain an adhesive bond
        self.neighbor_graph, self.jkr_graph = igraph.Graph(), igraph.Graph()

        # squaring the approximation of the differential
        self.dx2, self.dy2, self.dz2 = self.dx ** 2, self.dy ** 2, self.dz ** 2

        # get the time step value for diffusion updates depending on whether 2D or 3D
        if self.size[2] == 0:
            self.dt = (self.dx2 * self.dy2) / (2 * self.diffuse * (self.dx2 + self.dy2))
        else:
            self.dt = (self.dx2 * self.dy2 * self.dz2) / (2 * self.diffuse * (self.dx2 + self.dy2 + self.dz2))

        # the points at which the diffusion values are calculated
        x_steps = (int(self.size[0] / self.dx) + 1)
        y_steps = (int(self.size[1] / self.dy) + 1)
        z_steps = (int(self.size[2] / self.dz) + 1)
        self.fgf4_values = np.zeros((x_steps, y_steps, z_steps))
        self.extracellular_names = ["fgf4_values"]

        # holds all indices of cells that will divide at a current step or be removed at that step
        self.cells_to_divide, self.cells_to_remove = np.empty((0, 1), dtype=int), np.empty((0, 1), dtype=int)

        # min and max radius lengths are used to calculate linear growth of the radius over time in 2D
        self.min_radius = self.max_radius / 2 ** 0.5
        self.pluri_growth = (self.max_radius - self.min_radius) / self.pluri_div_thresh
        self.diff_growth = (self.max_radius - self.min_radius) / self.diff_div_thresh

        # the csv and video objects that will be updated each step
        self.csv_object = object()
        self.video_object = object()

        # given all of the above parameters, run the corresponding mode
        setup_simulation(self)

    def add_cell(self, location, radius, motion, fds, state, diff_counter, div_counter, death_counter, fds_counter,
                 motility_force, jkr_force, nearest_gata6, nearest_nanog, nearest_diff, highest_fgf4):
        """ Adds each of the new cell's values to
            the array holders, graphs, and total
            number of cells.
        """
        # adds the cell to the arrays holding the cell values, the 2D arrays have to be handled a bit differently as
        # axis=0 has to be provided and the appended array should also be of the same shape with additional brackets
        self.cell_locations = np.append(self.cell_locations, [location], axis=0)
        self.cell_radii = np.append(self.cell_radii, radius)
        self.cell_motion = np.append(self.cell_motion, motion)
        self.cell_fds = np.append(self.cell_fds, [fds], axis=0)
        self.cell_states = np.append(self.cell_states, state)
        self.cell_diff_counter = np.append(self.cell_diff_counter, diff_counter)
        self.cell_div_counter = np.append(self.cell_div_counter, div_counter)
        self.cell_death_counter = np.append(self.cell_death_counter, death_counter)
        self.cell_fds_counter = np.append(self.cell_fds_counter, fds_counter)
        self.cell_motility_force = np.append(self.cell_motility_force, [motility_force], axis=0)
        self.cell_jkr_force = np.append(self.cell_jkr_force, [jkr_force], axis=0)
        self.cell_nearest_gata6 = np.append(self.cell_nearest_gata6, nearest_gata6)
        self.cell_nearest_nanog = np.append(self.cell_nearest_nanog, nearest_nanog)
        self.cell_nearest_diff = np.append(self.cell_nearest_diff, nearest_diff)
        self.cell_highest_fgf4 = np.append(self.cell_highest_fgf4, [highest_fgf4], axis=0)

        # add it to the following graphs, this is done implicitly by increasing the length of the vertex list by
        # one, which the indices directly correspond to the cell holder arrays
        self.neighbor_graph.add_vertex()
        self.jkr_graph.add_vertex()

        # revalue the total number of cells
        self.number_cells += 1


def setup():
    """ controls which mode of the model is run and
        sets up the model accordingly
    """
    # if no command line attributes, run the mini gui to get the name and the mode
    if len(sys.argv) == 1:
        # prompt the user
        print("Type the name and the mode of the simulation when prompted")
        response = input("More for information on modes, type ""modes"" or for more help, type help \n"
                         "press any key to continue")

        # if response is help
        if response == "help":
            print("see documentation/README for more information")

        # if response is modes
        elif response == "modes" or "Modes":
            print("normal simulation: 0")
            print("continuation of past simulation: 1")
            print("images to video: 2")
            print("csvs to images/video: 3")

        else:
            pass

        # get the name and the mode of the simulation
        name = input("What is the name of the simulation?")
        mode = input("What is the mode of the simulation?")

    # if both the name and the mode are provided, do nothing
    elif len(sys.argv) == 3:
        name, mode = sys.argv[1], sys.argv[2]

    # if anything else
    else:
        if input("Type ""help"" for more information about beginning a simulation") == "help" or "Help":
            print("After python Run.py, include the name of the simulation followed by the mode or \n"
                  "simply no arguments for a small GUI")
        # anything else, open NetLogo
        else:
            webbrowser.open('https://ccl.northwestern.edu/netlogo/')
        exit()

    # open the path.txt file containing information about template locations and output directory
    with open('paths.txt', 'r') as file:
        lines = file.readlines()

    templates_path = lines[7]
    output_path = lines[13]







    # run the model normally
    if len(sys.argv) == 1:
        return Simulation()

    # continuation mode, where the model will continue a past simulation
    elif sys.argv[1] == 1:
        return Simulation()

    # images to video mode, where the model will turn images of a past simulation into videos
    elif sys.argv[1] == 2:
        pass

    # csvs to images mode, where the model will turn csvs of a past simulation into images and an a video
    elif sys.argv[1] == 3:
        pass

    # if user specifies "help" open NetLogo website
    elif sys.argv[1] == "help":
        webbrowser.open('https://ccl.northwestern.edu/netlogo/')

    else:
        pass

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
        output.initialize_csv(simulation)
        output.initialize_video(simulation)

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
    output.initialize_video(simulation)
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
    output.initialize_video(simulation)

    # loop over all csvs defined in the template file
    for i in range(simulation.beginning_step, simulation.end_step + 1):
        # updates the instance variables as they aren't updated by anything else
        simulation.current_step = i

        # calls the following function to add instances of the cell class to the simulation instance
        csv_to_simulation(simulation, simulation.path + simulation.name + "_values_" + str(int(i)) + ".csv")

        # saves a snapshot of the simulation
        output.step_image(simulation)

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
    output.finish_files(simulation)

    # exits out as the conversion from .csv to images/video is done
    exit()


def images_to_video_mode(simulation):
    """ If a simulation ends early or has an
        error, this will take the images up until
        it ended and turn them into a video.
    """
    print("Turning images into a video...")

    # create the video file
    output.initialize_video(simulation)

    # loop over all images defined in the template file
    for i in range(simulation.beginning_step, simulation.end_step + 1):
        # read the image and write it to the video file
        image_name = "_image_" + str(int(i)) + ".png"
        image = cv2.imread(simulation.path + simulation.name + image_name)
        simulation.video_object.write(image)

    # exits out as the conversion from images to video is done
    exit()


def csv_to_simulation(simulation, csv_file):
    """ Revalues the array holders for cell values
        based on the outputs of the csv files.
    """
    # opens the csv and create a list of the rows
    with open(csv_file, newline="") as csv_file:
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
    simulation.cell_nearest_gata6 = np.empty((simulation.number_cells, 3))
    simulation.cell_nearest_nanog = np.empty((simulation.number_cells, 3))
    simulation.cell_nearest_diff = np.empty((simulation.number_cells, 3))
    simulation.cell_highest_fgf4 = np.empty((simulation.number_cells, 3))


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
                    simulation.output_direct = input("What is the correct path? Don't include simulation name."
                                                     " (type new path): ")

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
