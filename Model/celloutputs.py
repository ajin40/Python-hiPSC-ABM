import numpy as np
import cv2
import math

from backend import record_time, check_direct


class CellOutputs:
    """ The methods in this class are meant to be inherited by the CellSimulation
        class so that CellSimulation objects can call these methods.
    """
    @record_time
    def step_image(self, background=(0, 0, 0), origin_bottom=True):
        """ Creates an image of the simulation space. Note the imaging library
            OpenCV uses BGR instead of RGB.
            background -> (tuple) The color of the background image as BGR
            origin_bottom -> (bool) Location of origin True -> bottom/left, False -> top/left
        """
        # only continue if outputting images
        if self.output_images:
            # get path and make sure directory exists
            check_direct(self.images_path)

            # get the size of the array used for imaging in addition to the scaling factor
            x_size = self.image_quality
            scale = x_size / self.size[0]
            y_size = math.ceil(scale * self.size[1])

            # create the cell space background image and apply background color
            image = np.zeros((y_size, x_size, 3), dtype=np.uint8)
            image[:, :] = background

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

            # if the origin should be bottom-left flip it, otherwise it will be top-left
            if origin_bottom:
                image = cv2.flip(image, 0)

            # save the image as a PNG
            image_compression = 4    # image compression of png (0: no compression, ..., 9: max compression)
            file_name = f"{self.name}_image_{self.current_step}.png"
            cv2.imwrite(self.images_path + file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, image_compression])

    @record_time
    def step_gradients(self):
        """ Saves each of the 2D gradients as a CSV at each step of the
            simulation.
        """
        # only continue if outputting gradient CSVs
        if self.output_gradients:
            # get path and make sure directory exists
            check_direct(self.gradients_path)

            # get the separator and save the following gradient outputs each to separate directories
            separator = self.paths.separator

            # go through all gradient arrays
            for gradient_name in self.gradient_names:
                # get directory to specific gradient
                check_direct(self.gradients_path + separator + gradient_name + separator)

                # get file name, use f-string
                file_name = f"{self.name}_{gradient_name}_{self.current_step}.csv"

                # convert gradient from 3D to 2D array and save it as CSV
                gradient = self.__dict__[gradient_name][:, :, 0]
                direct = self.gradients_path + separator + gradient_name + separator
                np.savetxt(direct + file_name, gradient, delimiter=",")

    @record_time
    def step_tda(self, in_pixels=False):
        """ Create CSV files for Topological Data Analysis (TDA) of different cell
            types. Each type will have its own subdirectory.
            in_pixels -> (bool) If the locations should be in pixels instead of meters
        """
        # only continue if outputting TDA files
        if self.output_tda:
            # get path and make sure directory exists
            check_direct(self.tda_path)

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
            separator = self.separator

            # save all cell locations to a CSV
            all_path = self.tda_path + separator + "all" + separator
            check_direct(all_path)
            file_name = f"{self.name}_tda_all_{self.current_step}.csv"
            np.savetxt(all_path + file_name, all_locations, delimiter=",")

            # save only GATA6 high cell locations to CSV
            red_path = self.tda_path + separator + "red" + separator
            check_direct(red_path)
            file_name = f"{self.name}_tda_red_{self.current_step}.csv"
            np.savetxt(red_path + file_name, red_locations, delimiter=",")

            # save only non-GATA6 high, pluripotent cells to a CSV
            green_path = self.tda_path + separator + "green" + separator
            check_direct(green_path)
            file_name = f"{self.name}_tda_green_{self.current_step}.csv"
            np.savetxt(green_path + file_name, green_locations, delimiter=",")
