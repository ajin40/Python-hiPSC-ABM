from PIL import Image, ImageDraw
from matplotlib.cm import ScalarMappable
import cv2
import csv
import time
import memory_profiler
import numpy as np
import pickle
import natsort
import os


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
    step_image(simulation)
    step_csv(simulation)
    step_gradients(simulation)
    step_tda(simulation)

    # number of cells, memory, step time, and individual function times
    simulation_data(simulation)

    # a temporary pickled file of the simulation, used for continuing past simulations
    temporary(simulation)


def step_image(simulation):
    """ Creates an image representation of the space in which
        the cells reside including the extracellular gradient.
    """
    # continue if images are desired
    if simulation.output_images:
        # create a dilation factor to adjust the coordinates to the image resolution
        dilation_x = simulation.image_quality[0] / simulation.size[0]
        dilation_y = simulation.image_quality[1] / simulation.size[1]

        # draws the background of the cell space
        base = Image.new("RGBA", simulation.image_quality, simulation.background_color)
        image = ImageDraw.Draw(base)

        # create an image of the FGF4 gradient if desired
        if simulation.output_gradient:
            # normalize the concentrations of the FGF4 array and reshape the array to 2D as before it's a 3D
            # equivalent of a 2D array
            fgf4_array = simulation.fgf4_values / simulation.max_fgf4
            fgf4_array = np.reshape(fgf4_array, (fgf4_array.shape[0], fgf4_array.shape[1]))

            # create a color map object that is used to turn the fgf4 array into a rgba array
            cmap_object = ScalarMappable(cmap='Blues')
            fgf4_as_rgba = cmap_object.to_rgba(fgf4_array, bytes=True, norm=False)

            # create an image from the fgf4 rgba array
            cmap_base = Image.fromarray(fgf4_as_rgba, mode="RGBA")
            cmap_base = cmap_base.resize(simulation.image_quality, resample=Image.BICUBIC)

            # transpose the image due to the way matplotlib/pillow interpret the array
            cmap_base = cmap_base.transpose(Image.TRANSPOSE)

            # draw the gradient color map
            cmap_image = ImageDraw.Draw(cmap_base)

        # loops over all of the cells, adding their image to the background image
        for i in range(simulation.number_cells):
            # determine the radius based on the dilation
            x_radius = dilation_x * simulation.cell_radii[i]
            y_radius = dilation_y * simulation.cell_radii[i]

            # get location in 2D with the dilation
            x = dilation_x * simulation.cell_locations[i][0]
            y = dilation_y * simulation.cell_locations[i][1]

            # color the cells according to the mode
            if simulation.color_mode:
                # True yields pluripotent/differentiated coloring
                if simulation.cell_states[i] == "Differentiated":
                    color = (230, 0, 0)
                else:
                    color = (22, 252, 32)

            # False yields coloring based on the finite dynamical system
            else:
                # color red if differentiated
                if simulation.cell_states[i] == "Differentiated":
                    color = (230, 0, 0)
                # color yellow if both high
                elif simulation.cell_fds[i][2] == 1 and simulation.cell_fds[i][3] == 1:
                    color = (255, 255, 30)
                # color blue if both low
                elif simulation.cell_fds[i][2] == 0 and simulation.cell_fds[i][3] == 0:
                    color = (50, 50, 255)
                # color white if gata6 high
                elif simulation.cell_fds[i][2] == 1:
                    color = (255, 255, 255)
                # color green if nanog high
                else:
                    color = (22, 252, 32)

            # get the four points that are the max/min along the x and y axes
            membrane_circle = (x - x_radius, y - y_radius, x + x_radius, y + y_radius)

            # draw the cell in the normal image
            image.ellipse(membrane_circle, fill=color, outline="black")

            # if outputting the gradient, draw a small outline of the cell to give an idea of where it without
            # hindering the gradient image
            if simulation.output_gradient:
                cmap_image.ellipse(membrane_circle, outline='gray', width=1)

        # get the image path
        image_path = simulation.images_path + simulation.name + "_image_" + str(int(simulation.current_step)) + ".png"

        # if including the gradient, combine the two images
        if simulation.output_gradient:
            # create a new background to paste to
            background = Image.new('RGBA', (base.width + cmap_base.width, base.height))

            # base both images
            background.paste(base, (0, 0))
            background.paste(cmap_base, (base.width, 0))

            # flip the image so that (0, 0) is bottom left
            background = background.transpose(Image.FLIP_TOP_BOTTOM)

            # save the image
            background.save(image_path, 'PNG')

        # no gradient, save only the cell space image
        else:
            # flip the image so that (0, 0) is bottom left
            base = base.transpose(Image.FLIP_TOP_BOTTOM)

            # save the image
            base.save(image_path, 'PNG')


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


def step_tda(simulation):
    """ Output a csv similar to the step_csv though this
        contains no header and only key cell info
    """
    # get file path
    file_path = simulation.tda_path + simulation.name + "_tda_" + str(int(simulation.current_step)) + ".csv"

    # open the file and create csv object
    with open(file_path, "w", newline="") as new_file:
        csv_file = csv.writer(new_file)

        # combine the multiple cell arrays into a single 2D list
        cell_data = list(zip(simulation.cell_locations[:, 0], simulation.cell_locations[:, 1], simulation.cell_states,
                             simulation.cell_fds[:, 2], simulation.cell_fds[:, 3]))

        # write the 2D list to the csv
        csv_file.writerows(cell_data)


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

        # calculate the total step time and the max memory used
        step_time = time.perf_counter() - simulation.step_start
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
