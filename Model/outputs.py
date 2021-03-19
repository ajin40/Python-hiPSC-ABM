import numpy as np
import cv2
import csv
import time
import psutil
import pickle
import os
import math
import re


class Outputs:
    """ The methods in this class are meant to be inherited by the Simulation
        class so that Simulation objects can called these methods. All output
        methods are placed here to organize the Simulation class a bit.
    """
    @record_time
    def step_image(self, background=(0, 0, 0), origin_bottom=True, fgf4_gradient=False):
        """ Creates an image of the simulation space. Note the imaging library
            OpenCV uses BGR instead of RGB.

            background -> (tuple) The color of the background image as BGR
            origin_bottom -> (bool) Location of origin True -> bottom/left, False -> top/left
            fgf4_gradient -> (bool) If outputting image of FGF4 gradient alongside step image
        """
        # only continue if outputting images
        if self.output_images:
            # get path and make sure directory exists
            directory_path = check_direct(self.paths.images)

            # get the size of the array used for imaging in addition to the scaling factor
            x_size = self.image_quality
            scale = x_size / self.size[0]
            y_size = math.ceil(scale * self.size[1])

            # create the cell space background image and apply background color
            image = np.zeros((y_size, x_size, 3), dtype=np.uint8)
            image[:, :] = background

            # if outputting gradient image, create it
            if fgf4_gradient:
                # normalize the concentration values and multiple by 255
                grad_image = 255 * self.fgf4_values[:, :, 0] / self.max_concentration
                grad_image = grad_image.astype(np.uint8)    # use unsigned int8

                # recolor the grayscale image into a colormap and resize to match the cell space image
                grad_image = cv2.applyColorMap(grad_image, cv2.COLORMAP_OCEAN)
                grad_image = cv2.resize(grad_image, (y_size, x_size), interpolation=cv2.INTER_NEAREST)

                # transpose the array to match the point location of OpenCV: (x, y) with origin top left
                grad_image = cv2.transpose(grad_image)

            # go through all of the cells
            for index in range(self.number_agents):
                # get xy coordinates and the axis lengths
                x, y = int(scale * self.locations[index][0]), int(scale * self.locations[index][1])
                major = int(scale * self.radii[index])
                minor = int(scale * self.radii[index])

                # color the cells according to the mode
                if self.color_mode:
                    if self.states[index] == 1:
                        color = (0, 0, 230)    # red
                    elif self.GATA6[index] >= self.NANOG[index] and self.GATA6[index] != 0:
                        color = (255, 255, 255)    # white
                    else:
                        color = (32, 252, 22)    # green

                # False yields coloring based on the finite dynamical system
                else:
                    if self.states[index] == 1:
                        color = (0, 0, 230)    # red
                    elif self.GATA6[index] > self.NANOG[index]:
                        color = (255, 255, 255)    # white
                    elif self.GATA6[index] == self.NANOG[index] == self.field - 1:
                        color = (30, 255, 255)    # yellow
                    elif self.GATA6[index] == self.NANOG[index] == 0:
                        color = (255, 50, 50)    # blue
                    else:
                        color = (32, 252, 22)    # green

                # draw the cell and a black outline to distinguish overlapping cells
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, color, -1)
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, (0, 0, 0), 1)

                # draw a black outline of the cell on the gradient image
                if fgf4_gradient:
                    grad_image = cv2.ellipse(grad_image, (x, y), (major, minor), 0, 0, 360, (255, 255, 255), 1)

            # if including gradient image, combine the to images side by side with gradient image on the right
            if fgf4_gradient:
                image = np.concatenate((image, grad_image), axis=1)

            # if the origin should be bottom-left flip it, otherwise it will be top-left
            if origin_bottom:
                image = cv2.flip(image, 0)

            # save the image as a PNG
            image_compression = 4    # image compression of png (0: no compression, ..., 9: max compression)
            file_name = f"{self.name}_image_{self.current_step}.png"
            cv2.imwrite(directory_path + file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, image_compression])

    @record_time
    def step_gradients(self):
        """ Saves each of the 2D gradients as a CSV at each step of the
            simulation.
        """
        # only continue if outputting gradient CSVs
        if self.output_gradients:
            # get path and make sure directory exists
            directory_path = check_direct(self.paths.gradients)

            # get the separator and save the following gradient outputs each to separate directories
            separator = self.paths.separator

            # go through all gradient arrays
            for gradient_name in self.gradient_names:
                # get directory to specific gradient
                grad_direct = check_direct(directory_path + separator + gradient_name + separator)

                # get file name, use f-string
                file_name = f"{self.name}_{gradient_name}_{self.current_step}.csv"

                # convert gradient from 3D to 2D array and save it as CSV
                gradient = self.__dict__[gradient_name][:, :, 0]
                np.savetxt(grad_direct + file_name, gradient, delimiter=",")

    @record_time
    def step_tda(self, in_pixels=False):
        """ Create CSV files for Topological Data Analysis (TDA) of different cell
            types. Each type will have its own subdirectory.

            in_pixels -> (bool) If the locations should be in pixels instead of meters
        """
        # only continue if outputting TDA files
        if self.output_tda:
            # get path and make sure directory exists
            directory_path = check_direct(self.paths.tda)

            # get the indices as an array of True/False of gata6 high cells and the non gata6 high cells
            red_indices = self.GATA6 > self.NANOG
            green_indices = np.invert(red_indices)

            # if TDA locations should be based on pixel location
            if in_pixels:
                scale = self.image_quality / self.size[0]
            else:
                scale = 1    # use meters

            # get the locations of the cells
            red_locations = self.locations[red_indices, 0:2] * scale
            green_locations = self.locations[green_indices, 0:2] * scale
            all_locations = self.locations[:, 0:2] * scale

            # get the separator and save the following TDA outputs each to separate directories
            separator = self.paths.separator

            # save all cell locations to a CSV
            all_path = check_direct(directory_path + separator + "all" + separator)
            file_name = f"{self.name}_tda_all_{self.current_step}.csv"
            np.savetxt(all_path + file_name, all_locations, delimiter=",")

            # save only GATA6 high cell locations to CSV
            red_path = check_direct(directory_path + separator + "red" + separator)
            file_name = f"{self.name}_tda_red_{self.current_step}.csv"
            np.savetxt(red_path + file_name, red_locations, delimiter=",")

            # save only non-GATA6 high, pluripotent cells to a CSV
            green_path = check_direct(directory_path + separator + "green" + separator)
            file_name = f"{self.name}_tda_green_{self.current_step}.csv"
            np.savetxt(green_path + file_name, green_locations, delimiter=",")


def record_time(function):
    """ A decorator used to time individual methods.
    """
    @wraps(function)
    def wrap(simulation, *args, **kwargs):  # args and kwargs are for additional arguments
        # get the start/end time and call the method
        start = time.perf_counter()
        function(simulation, *args, **kwargs)
        end = time.perf_counter()

        # add the time to the dictionary holding these times
        simulation.method_times[function.__name__] = end - start

    return wrap


def check_direct(path):
    """ Check directory for simulation outputs.
    """
    # if it doesn't exist make directory
    if not os.path.isdir(path):
        os.mkdir(path)

    # optionally return the path
    return path


def sort_naturally(file_list):
    """ Key for sorting the file list based on the step number.
    """
    return int(re.split('(\d+)', file_list)[-2])


def progress_bar(progress, maximum):
    """ Make a progress bar because progress bars are cool.
    """
    # length of the bar
    length = 60

    # calculate bar and percent
    progress += 1    # start at 1 not 0
    fill = int(length * progress / maximum)
    bar = '#' * fill + '.' * (length - fill)
    percent = int(100 * progress / maximum)

    # output the progress bar
    print(f"\r[{bar}] {percent}%", end="")
