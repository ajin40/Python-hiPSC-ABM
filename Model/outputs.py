import numpy as np
import cv2
import csv
import time
import psutil
import pickle
import os
import math
import re

from backend import record_time


class Paths:
    """ Hold any important paths for a particular simulation. For a continued
        simulation, this will update the Paths object in case the path(s) change.
    """
    def __init__(self, name, main, templates, separator):
        self.main = main    # the path to the main directory for this simulation
        self.templates = templates    # the path to the .txt template directory
        self.separator = separator    # file separator

        # these directories are sub-directories under the main simulation directory
        general = main + name
        self.images = general + "_images" + separator    # the images output directory
        self.values = general + "_values" + separator    # the cell array values output directory
        self.gradients = general + "_gradients" + separator    # the gradients output directory
        self.tda = general + "_tda" + separator    # the topological data analysis output directory


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
            for index in range(self.number_cells):
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
    def step_values(self, arrays=None):
        """ Outputs a CSV file with value from the cell arrays with each
            row corresponding to a particular cell index.

            arrays -> (list) If arrays is None, all cell arrays are outputted otherwise only the
                cell arrays named in the list will be outputted.
        """
        # only continue if outputting cell values
        if self.output_values:
            # if arrays is None automatically output all cell arrays
            if arrays is None:
                cell_array_names = self.cell_array_names

            # otherwise only output arrays specified in list
            else:
                cell_array_names = arrays

            # get path and make sure directory exists
            directory_path = check_direct(self.paths.values)

            # get file name, use f-string
            file_name = f"{self.name}_values_{self.current_step}.csv"

            # open the file
            with open(directory_path + file_name, "w", newline="") as file:
                # create CSV object and the following lists
                csv_file = csv.writer(file)
                header = list()    # header of the CSV (first row)
                data = list()    # holds the cell arrays

                # go through each of the cell arrays
                for array_name in cell_array_names:
                    # get the cell array
                    cell_array = self.__dict__[array_name]

                    # if the array is one dimensional
                    if cell_array.ndim == 1:
                        header.append(array_name)    # add the array name to the header
                        cell_array = np.reshape(cell_array, (-1, 1))  # resize array from 1D to 2D

                    # if the array is not one dimensional
                    else:
                        # create name for column based on slice of array ex. locations[0], locations[1], locations[2]
                        for i in range(cell_array.shape[1]):
                            header.append(array_name + "[" + str(i) + "]")

                    # add the array to the data holder
                    data.append(cell_array)

                # write header as the first row of the CSV
                csv_file.writerow(header)

                # stack the arrays to create rows for the CSV file and save to CSV
                data = np.hstack(data)
                csv_file.writerows(data)

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

    @record_time
    def temp(self):
        """ Pickle the current state of the Simulation object which can be used
            to continue a past simulation without losing information.
        """
        # get file name, use f-string
        file_name = f"{self.name}_temp.pkl"

        # open the file in binary mode
        with open(self.paths.main + file_name, "wb") as file:
            # use the highest protocol: -1 for pickling the instance
            pickle.dump(self, file, -1)

    def data(self):
        """ Creates/adds a new line to the running CSV for data about the simulation
            such as memory, step time, number of cells and method profiling.
        """
        # get file name, use f-string
        file_name = f"{self.name}_data.csv"

        # open the file
        with open(self.paths.main + file_name, "a", newline="") as file_object:
            # create CSV object
            csv_object = csv.writer(file_object)

            # create header if this is the beginning of a new simulation
            if self.current_step == 1:
                # header names
                header = ["Step Number", "Number Cells", "Step Time", "Memory (MB)"]

                # header with all the names of the functions with the "record_time" decorator
                functions_header = list(self.method_times.keys())

                # merge the headers together and write the row to the CSV
                csv_object.writerow(header + functions_header)

            # calculate the total step time and get memory of current python process in megabytes
            step_time = time.perf_counter() - self.step_start
            process = psutil.Process(os.getpid())
            memory = process.memory_info()[0] / 1024 ** 2

            # write the row with the corresponding values
            columns = [self.current_step, self.number_cells, step_time, memory]
            function_times = list(self.method_times.values())
            csv_object.writerow(columns + function_times)

    def create_video(self):
        """ Write all of the step images from a simulation to video file in the
            main simulation directory.
        """
        # continue if there is an image directory
        if os.path.isdir(self.paths.images):
            # get all of the images in the directory and the number of images
            file_list = [file for file in os.listdir(self.paths.images) if file.endswith(".png")]
            image_count = len(file_list)

            # only continue if image directory has images in it
            if image_count > 0:
                print("\nCreating video...")

                # sort the file list so "2, 20, 3, 31..." becomes "2, 3, ..., 20, ..., 31"
                file_list = sorted(file_list, key=sort_naturally)

                # sample the first image to get the dimensions of the image, and then scale the image
                size = cv2.imread(self.paths.images + file_list[0]).shape[0:2]
                scale = self.video_quality / size[1]
                new_size = (self.video_quality, int(scale * size[0]))

                # get the video file path, use f-string
                file_name = f"{self.name}_video.mp4"
                video_path = self.paths.main + file_name

                # create the file object with parameters from simulation and above
                codec = cv2.VideoWriter_fourcc(*"mp4v")
                video_object = cv2.VideoWriter(video_path, codec, self.fps, new_size)

                # go through sorted image list, reading and writing each image to the video object
                for i in range(image_count):
                    image = cv2.imread(self.paths.images + file_list[i])    # read image from directory
                    if size != new_size:
                        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)    # scale down if necessary
                    video_object.write(image)    # write to video
                    progress_bar(i, image_count)    # show progress

                # close the video file
                video_object.release()

        # print end statement...super important...don't remove or model won't run!
        print("\n\nThe simulation is finished. May the force be with you.\n")


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
