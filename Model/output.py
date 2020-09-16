import matplotlib.pyplot as plt
import matplotlib.collections
import cv2
import csv
import time
import memory_profiler
import numpy as np
import pickle
import natsort
import os
import gc
import math
import random as r

import backend


def initialize_outputs(simulation):
    """ Sets up the simulation data csv and makes directories
        for images, values, and the gradients
    """
    # make directories for the following given initial parameters and if directories already exist
    if not os.path.isdir(simulation.images_path) and simulation.output_images:
        os.mkdir(simulation.images_path)
    if not os.path.isdir(simulation.values_path):
        os.mkdir(simulation.values_path)
    if not os.path.isdir(simulation.gradients_path):
        os.mkdir(simulation.gradients_path)
    if not os.path.isdir(simulation.tda_path) and simulation.output_tda:
        os.mkdir(simulation.tda_path)


def step_outputs(simulation):
    """ Calls multiple functions that each output some sort of
        file relating to the simulation at a particular step
    """
    # information about the cells/environment at current step
    # step_image(simulation)
    step_image_new(simulation)
    # step_csv(simulation)
    # step_gradients(simulation)
    # step_tda(simulation)

    # a temporary pickled file of the simulation, used for continuing past simulations
    temporary(simulation)

    # number of cells, memory, step time, and individual function times
    simulation_data(simulation)


@backend.record_time
def step_image(simulation):
    """ Creates an image representation of the space in which
        the cells reside including the extracellular gradient.
    """
    # continue if images are desired
    if simulation.output_images:
        # create an image of the FGF4 gradient if desired
        if simulation.output_gradient:
            # create a figure with two subplots one for the space image and one for the gradient
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # no gradient image
        else:
            # create a figure with one subplot for the space image
            fig, axes = plt.subplots(1, 1, figsize=(5, 5))
            axes = [axes]

        # get the 2D locations and minor/major axes of the ellipses which are just circles for now
        locations = simulation.cell_locations[:, :2]
        minor = simulation.cell_radii * 2
        major = simulation.cell_radii * 2
        rotate = np.zeros(simulation.number_cells)

        # create an array to write cell colors to
        colors = np.empty((simulation.number_cells, 3), dtype=float)

        # color the cells according to the mode
        if simulation.color_mode:
            for i in range(simulation.number_cells):
                # if the cell is differentiated, color red
                if simulation.cell_states[i] == "Differentiated":
                    colors[i] = [230, 0, 0]

                # if the cell is gata6 high and nanog low, color white
                elif simulation.cell_fds[i][2] * (simulation.cell_fds[i][3] + 1) % 2:
                    colors[i] = [255, 255, 255]

                # if anything else, color green
                else:
                    colors[i] = [22, 252, 32]

        # False yields coloring based on the finite dynamical system
        else:
            for i in range(simulation.number_cells):
                # if the cell is differentiated, color red
                if simulation.cell_states[i] == "Differentiated":
                    colors[i] = [230, 0, 0]

                # if the cell is both gata6 high and nanog high, color yellow
                elif simulation.cell_fds[i][2] * simulation.cell_fds[i][3] % 2:
                    colors[i] = [255, 255, 30]

                # if the cell is both gata6 low and nanog low, color blue
                elif (simulation.cell_fds[i][2] + 1) * (simulation.cell_fds[i][3] + 1) % 2:
                    colors[i] = [50, 50, 255]

                # if the cell is gata6 high and nanog low, color white
                elif simulation.cell_fds[i][2] * (simulation.cell_fds[i][3] + 1) % 2:
                    colors[i] = [255, 255, 255]

                # if anything else, color green
                else:
                    colors[i] = [22, 252, 32]

        # convert colors to floats and turn entries into tuples
        colors /= 255
        colors = tuple(map(tuple, colors))

        # create a collection of ellipses representing the cells for the space image
        ellipses = matplotlib.collections.EllipseCollection(major, minor, rotate, offsets=locations, units='xy',
                                                            facecolors=colors, edgecolors="black", linewidths=0.1,
                                                            transOffset=axes[0].transData)

        # add the ellipses to the plot and perform the following edits to the plot
        axes[0].add_collection(ellipses)
        axes[0].margins(0, 0)
        axes[0].set_xlim(0, simulation.size[0])
        axes[0].set_ylim(0, simulation.size[1])
        axes[0].axis("off")
        axes[0].set_facecolor("k")
        axes[0].add_artist(axes[0].patch)
        axes[0].patch.set_zorder(-1)

        # create the image of the gradient
        if simulation.output_gradient:
            # plot the gradient
            axes[1].imshow(simulation.fgf4_values[:, :, 0], cmap="Blues", interpolation="nearest", origin="lower",
                           extent=[0, simulation.size[0], 0, simulation.size[1]], vmin=0, vmax=simulation.max_fgf4)

            # add outlines of cells to gradient image
            ellipses_ = matplotlib.collections.EllipseCollection(major, minor, rotate, offsets=locations, units='xy',
                                                                 facecolors="none", edgecolors="black", linewidths=0.1,
                                                                 transOffset=axes[1].transData)

            # add the ellipses to the plot and perform the following edits to the plot
            axes[1].add_collection(ellipses_)
            axes[1].margins(0, 0)
            axes[1].axis("off")

        # remove margins
        plt.tight_layout(pad=0)

        # save the image
        image_path = simulation.images_path + simulation.name + "_image_" + str(int(simulation.current_step)) + ".png"
        plt.savefig(image_path, dpi=200)
        plt.close("all")


@backend.record_time
def step_image_new(simulation):
    """ Creates an image representation of the space in which
        the cells reside including the extracellular gradient.
    """
    # get the size of the array used for imaging
    pixels = 2000
    scale = pixels/simulation.size[0]
    x_size = pixels
    y_size = math.ceil(pixels * simulation.size[1] / simulation.size[0])

    # create the array
    image = np.zeros((y_size, x_size, 3), dtype=np.uint8)

    # go through all of the cells
    for i in range(simulation.number_cells):
        # get the following values
        x = math.ceil(simulation.cell_locations[i][0] * scale)
        y = math.ceil(simulation.cell_locations[i][1] * scale)
        point = (x, y)
        major = math.ceil(simulation.cell_radii[i] * scale)
        minor = math.ceil(simulation.cell_radii[i] * scale)
        rotation = r.randint(0, 360)

        # color the cells according to the mode
        if simulation.color_mode:
            # if the cell is differentiated, color red
            if simulation.cell_states[i] == "Differentiated":
                color = (0, 0, 230)

            # if the cell is gata6 high and nanog low, color white
            elif simulation.cell_fds[i][2] * (simulation.cell_fds[i][3] + 1) % 2:
                color = (255, 255, 255)

            # if anything else, color green
            else:
                color = (32, 252, 22)

        # False yields coloring based on the finite dynamical system
        else:
            # if the cell is differentiated, color red
            if simulation.cell_states[i] == "Differentiated":
                color = (0, 0, 230)

            # if the cell is both gata6 high and nanog high, color yellow
            elif simulation.cell_fds[i][2] * simulation.cell_fds[i][3] % 2:
                color = (30, 255, 255)

            # if the cell is both gata6 low and nanog low, color blue
            elif (simulation.cell_fds[i][2] + 1) * (simulation.cell_fds[i][3] + 1) % 2:
                color = (255, 50, 50)

            # if the cell is gata6 high and nanog low, color white
            elif simulation.cell_fds[i][2] * (simulation.cell_fds[i][3] + 1) % 2:
                color = (255, 255, 255)

            # if anything else, color green
            else:
                color = (32, 252, 22)

        # update the array with the cell image
        image = cv2.ellipse(image, point, (major, minor), rotation, 0, 360, color, -1)
        image = cv2.ellipse(image, point, (major, minor), rotation, 0, 360, (0, 0, 0), 1)

    if simulation.output_gradient:

        a = simulation.fgf4_values[:, :, 0] / simulation.max_fgf4

        a *= 255
        b = np.ones((a.shape[0], a.shape[1]), dtype=np.uint8) * 255
        a = b - a
        a = a.astype(np.uint8)
        a = cv2.applyColorMap(a, cv2.COLORMAP_OCEAN)

        a = cv2.resize(a, (x_size, y_size), interpolation=cv2.INTER_NEAREST)

        # go through all of the cells
        for i in range(simulation.number_cells):
            # get the following values
            x = math.ceil(simulation.cell_locations[i][0] * scale)
            y = math.ceil(simulation.cell_locations[i][1] * scale)
            point = (x, y)
            major = math.ceil(simulation.cell_radii[i] * scale)
            minor = math.ceil(simulation.cell_radii[i] * scale)
            rotation = r.randint(0, 360)
            a = cv2.ellipse(a, point, (major, minor), rotation, 0, 360, (75, 75, 75), 1)

        image = np.concatenate((image, a), axis=1)

    image = cv2.flip(image, 0)
    image_path = simulation.images_path + simulation.name + "_image_" + str(int(simulation.current_step)) + ".png"
    cv2.imwrite(image_path, image)


@backend.record_time
def step_csv(simulation):
    """ Outputs a .csv file containing information about
        all cells with each row corresponding to a cell
    """
    # get file path
    file_path = simulation.values_path + simulation.name + "_values_" + str(int(simulation.current_step)) + ".csv"

    # open the file and create csv object
    with open(file_path, "w", newline="") as new_file:
        csv_file = csv.writer(new_file)

        # create a header as the first row of the csv
        csv_file.writerow(['x_location', 'y_location', 'z_location', 'radius', 'motion', 'FGFR', 'ERK', 'GATA6',
                           'NANOG', 'state', 'differentiation_counter', 'division_counter', 'death_counter',
                           'fds_counter'])

        # combine the multiple cell arrays into a single 2D list
        cell_data = list(zip(simulation.cell_locations[:, 0], simulation.cell_locations[:, 1],
                             simulation.cell_locations[:, 2], simulation.cell_radii, simulation.cell_motion,
                             simulation.cell_fds[:, 0], simulation.cell_fds[:, 1], simulation.cell_fds[:, 2],
                             simulation.cell_fds[:, 3], simulation.cell_states, simulation.cell_diff_counter,
                             simulation.cell_div_counter, simulation.cell_death_counter,
                             simulation.cell_fds_counter))

        # write the 2D list to the csv
        csv_file.writerows(cell_data)


@backend.record_time
def step_gradients(simulation):
    """ Saves the gradient arrays as .npy files for use in
        later imaging and/or continuation of previous
        simulations.
    """
    # go through all gradient arrays, skipping the temporary array
    for gradient, temp in simulation.extracellular_names:
        # get the name for the file
        gradient_name = "_" + gradient + "_" + str(simulation.current_step)

        # save the gradient with numpy
        np.save(simulation.gradients_path + simulation.name + gradient_name + ".npy", simulation.__dict__[gradient])


@backend.record_time
def step_tda(simulation):
    """ Output a csv similar to the step_csv though this
        contains no header and only key cell info
    """
    # get file path
    file_path = simulation.tda_path + simulation.name + "_tda_" + str(int(simulation.current_step)) + ".csv"

    # open the file and create csv object
    with open(file_path, "w", newline="") as new_file:
        csv_file = csv.writer(new_file)

        # create an array to write cell colors to
        cell_color = np.empty(simulation.number_cells, dtype="<U14")

        # go through all cells giving the corresponding color
        for i in range(simulation.number_cells):
            if simulation.cell_states[i] == "Differentiated":
                color = "red"
            elif simulation.cell_fds[i][2] and not simulation.cell_fds[i][3]:
                color = "white"
            elif not simulation.cell_fds[i][2] and simulation.cell_fds[i][3]:
                color = "green"
            else:
                color = "other"

            # update color
            cell_color[i] = color

        # combine the multiple cell arrays into a single 2D list
        cell_data = list(zip(simulation.cell_locations[:, 0], simulation.cell_locations[:, 1], cell_color))

        # write the 2D list to the csv
        csv_file.writerows(cell_data)


@backend.record_time
def temporary(simulation):
    """ Pickle a copy of the simulation class that can be used
        to continue a past simulation without losing information.
    """
    # open the file and get the object
    with open(simulation.path + simulation.name + '_temp.pkl', 'wb') as file:
        # use the highest protocol "-1" for pickling the instance
        pickle.dump(simulation, file, -1)


def simulation_data(simulation):
    """ Creates/adds a new line to the running csv for data amount
        the simulation such as memory, step time, number of cells,
        and run time of functions.
    """
    # get path to data csv
    data_path = simulation.path + simulation.name + "_data.csv"

    # open the file and create csv object
    with open(data_path, "a", newline="") as file_object:
        csv_object = csv.writer(file_object)

        # create header if this is the beginning of a new simulation
        if simulation.current_step == 1:
            # add/remove custom elements of the header
            custom_header = ["Step Number", "Number Cells", "Step Time", "Memory (MB)"]

            # header with all the names of the functions with the "record_time" decorator
            functions_header = list(simulation.function_times.keys())

            # add the headers together and write the row to the csv
            csv_object.writerow(custom_header + functions_header)

        # calculate the total step time
        step_time = time.perf_counter() - simulation.step_start

        # run a garbage collect before getting the memory
        gc.collect()
        memory = memory_profiler.memory_usage(max_usage=True)

        # write the row with the corresponding values
        custom = [simulation.current_step, simulation.number_cells, step_time, memory]
        functions = list(simulation.function_times.values())
        csv_object.writerow(custom + functions)


def create_video(simulation):
    """ Takes all of the step images of a simulation and writes
        them to a new video file.
    """
    # continue if there is an image directory
    if os.path.isdir(simulation.images_path):
        # get all of the images in the directory
        file_list = os.listdir(simulation.images_path)

        # continue if image directory has images in it
        if len(file_list) > 0:
            # sort the list naturally so "2, 20, 3, 31..." becomes "2, 3,...,20,...,31"
            file_list = natsort.natsorted(file_list)

            # sample the first image to get the shape of the images
            frame = cv2.imread(simulation.images_path + file_list[0])
            height, width, channels = frame.shape

            # get the video file path
            video_path = simulation.path + simulation.name + '_video.avi'

            # create the file object with parameters from simulation and above
            video_object = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc("M", "J", "P", "G"), simulation.fps,
                                           (width, height))

            # go through sorted image name list reading and writing each to the video object
            for image_file in file_list:
                image = cv2.imread(simulation.images_path + image_file)
                video_object.write(image)

            # close the video file
            video_object.release()

    # print end statement...super important. Don't remove or model won't run!
    print("The simulation is finished. May the force be with you.")
