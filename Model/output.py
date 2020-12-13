import cv2
import csv
import time
import memory_profiler
import numpy as np
import pickle
import natsort
import os
import math

import backend


class Paths:
    """ This object contains the paths to the multiple output
        directories for the simulation.
    """
    def __init__(self, name, main, templates, separator):
        # the main directory of the simulation output
        self.main = main

        # the directory to the template .txt file
        self.templates = templates

        # these directories are sub-directories under the main simulation directory
        self.images = main + name + "_images" + separator    # the images output directory
        self.values = main + name + "_values" + separator    # the cell array values output directory
        self.gradients = main + name + "_gradients" + separator    # the gradients output directory
        self.tda = main + name + "_tda" + separator    # the topological data analysis output directory


def step_outputs(simulation):
    """ Calls multiple functions that each output some sort of
        file relating to the simulation at a particular step.
    """
    # information about the cells/environment at current step
    step_image(simulation)
    step_values(simulation)
    step_gradients(simulation)
    step_tda(simulation)

    # create a temporary pickle of the Simulation object
    temporary(simulation)

    # number of cells, memory, step time, and individual methods times
    simulation_data(simulation)


@backend.record_time
def step_image(simulation):
    """ Creates an image representation of the space in which
        the cells reside including the extracellular gradient.
        Note OpenCV uses BGR instead of RGB.
    """
    # only continue if outputting images
    if simulation.output_images:
        # make sure directory exists
        if not os.path.isdir(simulation.paths.images):
            os.mkdir(simulation.paths.images)

        # get the size of the array used for imaging in addition to the scaling factor
        pixels = simulation.image_quality
        scale = pixels/simulation.size[0]
        x_size = pixels
        y_size = math.ceil(scale * simulation.size[1])

        # create the cell space background image
        image = np.zeros((y_size, x_size, 3), dtype=np.uint8)

        # create the gradient image
        if simulation.output_fgf4_image:
            # normalize the concentration values and multiple by 255 to create grayscale image
            grad_image = simulation.fgf4_values[:, :, 0] * (255 / simulation.max_concentration)
            grad_image = grad_image.astype(np.uint8)

            # recolor the grayscale image into a colormap and resize to match the cell space array
            grad_image = cv2.applyColorMap(grad_image, cv2.COLORMAP_OCEAN)
            grad_image = cv2.resize(grad_image, (y_size, x_size), interpolation=cv2.INTER_NEAREST)

            # flip and rotate to turn go from (y, x) to (x, y) so that origin is top, left to match OpenCV locations
            grad_image = cv2.rotate(grad_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            grad_image = cv2.flip(grad_image, 0)

        # go through all of the cells
        for index in range(simulation.number_cells):
            x = math.ceil(simulation.locations[index][0] * scale)    # the x-coordinate
            y = math.ceil(simulation.locations[index][1] * scale)    # the y-coordinate
            point = (x, y)                                           # the x, y point
            major = math.ceil(simulation.radii[index] * scale)       # the major axis length
            minor = math.ceil(simulation.radii[index] * scale)       # the minor axis length
            rotation = 0                                             # the rotation of the ellipse (zero for now)

            # color the cells according to the mode
            if simulation.color_mode:
                # if the cell is differentiated, color red
                if simulation.states[index] == "Differentiated":
                    color = (0, 0, 230)

                # if the cell is gata6 high and nanog low, color white
                elif simulation.GATA6[index] > simulation.NANOG[index]:
                    color = (255, 255, 255)

                # if anything else, color green
                else:
                    color = (32, 252, 22)

            # False yields coloring based on the finite dynamical system
            else:
                # if the cell is differentiated, color red
                if simulation.states[index] == "Differentiated":
                    color = (0, 0, 230)

                # if the cell is gata6 high and nanog low, color white
                elif simulation.GATA6[index] > simulation.NANOG[index]:
                    color = (255, 255, 255)

                # if the cell is both gata6 high and nanog high, color yellow
                elif simulation.GATA6[index] == simulation.NANOG[index] == simulation.field - 1:
                    color = (30, 255, 255)

                # if the cell is both gata6 low and nanog low, color blue
                elif simulation.GATA6[index] == simulation.NANOG[index] == 0:
                    color = (255, 50, 50)

                # if anything else, color green
                else:
                    color = (32, 252, 22)

            # draw the cell and a black outline to distinguish overlapping cells
            image = cv2.ellipse(image, point, (major, minor), rotation, 0, 360, color, -1)
            image = cv2.ellipse(image, point, (major, minor), rotation, 0, 360, (0, 0, 0), 1)

            # draw the outline of the cell on the gradient image
            if simulation.output_fgf4_image:
                grad_image = cv2.ellipse(grad_image, point, (major, minor), rotation, 0, 360, (255, 255, 255), 1)

        # combine the to images side by side if including gradient, gradient will be on the right
        if simulation.output_fgf4_image:
            image = np.concatenate((image, grad_image), axis=1)

        # flip the image so that origin goes from top, left to bottom, left
        image = cv2.flip(image, 0)

        # save the image as a PNG
        image_path = simulation.paths.images + simulation.name + "_image_" + str(int(simulation.current_step)) + ".png"
        cv2.imwrite(image_path, image)


@backend.record_time
def step_values(simulation):
    """ Outputs a CSV file containing information about
        from all cell arrays
    """
    # only continue if outputting cell values
    if simulation.output_values:
        # make sure directory exists
        if not os.path.isdir(simulation.paths.values):
            os.mkdir(simulation.paths.values)

        # get file path
        file_path = simulation.paths.values + simulation.name + "_values_" + str(int(simulation.current_step)) + ".csv"

        # open the file
        with open(file_path, "w", newline="") as new_file:
            # create CSV object
            csv_file = csv.writer(new_file)

            # creat lists for the header and the data of the CSV
            header = list()
            data = list()

            # go through each of the cell arrays
            for array_name in simulation.cell_array_names:
                # get the cell array
                cell_array = simulation.__dict__[array_name]

                # if the array is one dimensional
                if cell_array.ndim == 1:
                    header.append(array_name)    # add the array name to the header
                    cell_array = np.reshape(cell_array, (-1, 1))  # resize array from 1D to 2D
                    data.append(cell_array)    # add the array to the data holder

                # if the array is not one dimensional
                else:
                    # add multiple headers for each slice of the 2D array
                    for i in range(cell_array.shape[1]):
                        header.append(array_name + "[" + str(i) + "]")
                    data.append(cell_array)    # add the array to the data holder

            # create a header as the first row of the CSV
            csv_file.writerow(header)

            # stack the arrays to create rows for the CSV file
            cell_data = np.hstack(data)

            # write the 2D list to the CSV
            csv_file.writerows(cell_data)


@backend.record_time
def step_gradients(simulation):
    """ Saves the gradient arrays as .npy files for
        potential later analysis with python
    """
    # only continue if outputting gradient pickles
    if simulation.output_gradients:
        # make sure directory exists
        if not os.path.isdir(simulation.gradients_path):
            os.mkdir(simulation.gradients_path)

        # go through all gradient arrays
        for gradient_name in simulation.gradient_names:
            # get the name for the file
            name = "_" + gradient_name + "_" + str(simulation.current_step)

            # save the gradient via numpy compression
            np.save(simulation.paths.gradients + simulation.name + name + ".npy", simulation.__dict__[gradient_name])


@backend.record_time
def step_tda(simulation):
    """ Output a CSV similar to the step_csv though this
        contains no header and only key cell info
    """
    # only continue if outputting TDA files
    if simulation.output_tda:
        # make sure directory exists
        if not os.path.isdir(simulation.paths.tda):
            os.mkdir(simulation.paths.tda)

        # get file path
        file_path = simulation.paths.tda + simulation.name + "_tda_" + str(int(simulation.current_step)) + ".csv"

        # open the file
        with open(file_path, "w", newline="") as new_file:
            # create CSV object
            csv_file = csv.writer(new_file)

            # create a temporary array to write cell colors to
            cell_colors = np.empty(simulation.number_cells, dtype="<U14")

            # go through all cells giving the corresponding color
            for i in range(simulation.number_cells):
                if simulation.states[i] == "Differentiated":
                    color = "red"
                elif simulation.GATA6[i] > simulation.NANOG[i]:
                    color = "white"
                elif not simulation.GATA6[i] < simulation.NANOG[i]:
                    color = "green"
                else:
                    color = "other"

                # update color
                cell_colors[i] = color

            # combine the multiple cell arrays into a single 2D list
            cell_data = list(zip(simulation.locations[:, 0], simulation.locations[:, 1], cell_colors))

            # write the 2D list to the CSV
            csv_file.writerows(cell_data)


@backend.record_time
def temporary(simulation):
    """ Pickle a copy of the simulation class that can be used
        to continue a past simulation without losing information
    """
    # get file path
    file_path = simulation.paths.main + simulation.name + "_temp" + ".pkl"

    # open the file and get the object
    with open(file_path, 'wb') as temp_file:
        # use the highest protocol: -1 for pickling the instance
        pickle.dump(simulation, temp_file, -1)


def simulation_data(simulation):
    """ Creates/adds a new line to the running CSV for data about
        the simulation such as memory, step time, number of cells,
        and run time of functions
    """
    # get path to data CSV
    data_path = simulation.paths.main + simulation.name + "_data.csv"

    # open the file
    with open(data_path, "a", newline="") as file_object:
        # create CSV object
        csv_object = csv.writer(file_object)

        # create header if this is the beginning of a new simulation
        if simulation.current_step == 1:
            # header names
            header = ["Step Number", "Number Cells", "Step Time", "Memory (MB)"]

            # header with all the names of the functions with the "record_time" decorator
            functions_header = list(simulation.method_times.keys())

            # merge the headers together and write the row to the CSV
            csv_object.writerow(header + functions_header)

        # calculate the total step time and get the current memory used by the model
        step_time = time.perf_counter() - simulation.step_start
        memory = memory_profiler.memory_usage()[0]

        # write the row with the corresponding values
        columns = [simulation.current_step, simulation.number_cells, step_time, memory]
        function_times = list(simulation.method_times.values())
        csv_object.writerow(columns + function_times)


def create_video(simulation):
    """ Takes all of the step images of a simulation and writes
        them to a new video file.
    """
    # continue if there is an image directory
    if os.path.isdir(simulation.paths.images):
        # get all of the images in the directory
        file_list = os.listdir(simulation.paths.images)

        # continue if image directory has images in it
        image_count = len(file_list)
        if image_count > 0:
            print("\nCreating video...")

            # sort the list naturally so "2, 20, 3, 31..." becomes "2, 3,...,20,...,31"
            file_list = natsort.natsorted(file_list)

            # sample the first image to get the shape of the images
            frame = cv2.imread(simulation.paths.images + file_list[0])
            height, width, channels = frame.shape

            # get the video file path
            video_path = simulation.paths.main + simulation.name + '_video.mp4'

            # create the file object with parameters from simulation and above
            video_object = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), simulation.fps, (width, height))

            # go through sorted image name list, reading and writing each to the video object
            for i in range(image_count):
                image = cv2.imread(simulation.paths.images + file_list[i])
                video_object.write(image)
                backend.progress_bar(i + 1, image_count)

            # close the video file
            video_object.release()

    # print end statement...super important. Don't remove or model won't run!
    print("\n\nThe simulation is finished. May the force be with you.")
