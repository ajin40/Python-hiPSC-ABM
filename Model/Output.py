from PIL import Image, ImageDraw
import cv2
import csv
import numpy as np


def draw_image(simulation):
    """ Turns the graph into an image at each timestep
    """
    # increases the image counter by 1 each time this is called
    simulation.image_counter += 1

    # thickness of the image slice in the z direction
    thickness = simulation.size[2] / simulation.slices

    # get the dilation of the image for the correctly sizing the image
    dilation_x = simulation.image_quality[0] / simulation.size[0]
    dilation_y = simulation.image_quality[1] / simulation.size[1]

    # bounds of the simulation used for drawing lines
    corner_1 = [0, 0]
    corner_2 = [0, simulation.size[1]]
    corner_3 = [simulation.size[0], simulation.size[1]]
    corner_4 = [simulation.size[0], 0]
    bounds = np.array([corner_1, corner_2, corner_3, corner_4])

    # starting z value for slice location
    lower_slice = 0
    upper_slice = thickness

    for i in range(simulation.slices):

        # draws the background of the image
        base = Image.new("RGB", simulation.image_quality[0:2], simulation.background_color)
        image = ImageDraw.Draw(base)

        # loops over all of the cells and draws the nucleus and radius
        for j in range(len(simulation.cells)):
            location = simulation.cells[j].location
            radius = simulation.cells[j].radius

            # determine the radius based on the lower bound of the slice
            x_radius_lower = dilation_x * max(radius ** 2 - (location[2] - lower_slice) ** 2, 0.0) ** 0.5
            y_radius_lower = dilation_y * max(radius ** 2 - (location[2] - lower_slice) ** 2, 0.0) ** 0.5

            # determine the radius based on the upper bound of the slice
            x_radius_upper = dilation_x * max(radius ** 2 - (location[2] - upper_slice) ** 2, 0.0) ** 0.5
            y_radius_upper = dilation_y * max(radius ** 2 - (location[2] - upper_slice) ** 2, 0.0) ** 0.5

            # check to see which slice will produce the largest radius
            if x_radius_lower >= x_radius_upper:
                x_radius = x_radius_lower
                y_radius = y_radius_lower
            else:
                x_radius = x_radius_upper
                y_radius = y_radius_upper

            # get location in 2D with image size dilation
            x = dilation_x * location[0]
            y = dilation_y * location[1]

            # coloring of the cells
            if simulation.current_step >= simulation.dox_step:
                if simulation.cells[j].state == "Differentiated":
                    color = simulation.diff_color
                elif simulation.cells[j].booleans[2] == 1 and simulation.cells[j].booleans[3] == 1:
                    color = simulation.pluri_both_high_color
                elif simulation.cells[j].booleans[2] == 1:
                    color = simulation.pluri_gata6_high_color
                elif simulation.cells[j].booleans[3] == 1:
                    color = simulation.pluri_nanog_high_color
                else:
                    color = simulation.pluri_nanog_high_color
            else:
                color = simulation.pluri_nanog_high_color

            # draw the circle representing the cell
            membrane_circle = (x - x_radius, y - y_radius, x + x_radius, y + y_radius)
            image.ellipse(membrane_circle, fill=color)

        # loops over all of the bounds and draws lines to represent the grid
        for j in range(len(bounds)):
            # get the bound sizing
            x0 = dilation_x * bounds[j][0]
            y0 = dilation_y * bounds[j][1]

            # get the bounds as lines
            if j < len(bounds) - 1:
                x1 = dilation_x * bounds[j + 1][0]
                y1 = dilation_y * bounds[j + 1][1]
            else:
                x1 = dilation_x * bounds[0][0]
                y1 = dilation_y * bounds[0][1]

            # draw the lines
            lines = (x0, y0, x1, y1)
            width = int((simulation.image_quality[0] + simulation.image_quality[0]) / 500)
            image.line(lines, fill=simulation.bound_color, width=width)

        # saves the image as a .png
        image_name = "image_" + str(int(simulation.current_step)) + "_slice_" + str(int(i)) + ".png"
        base.save(simulation.path + image_name, 'PNG')

        # moves to the next slice location
        lower_slice += thickness
        upper_slice += thickness


def image_to_video(simulation):
    """ Creates a video out of all the png images at
        the end of the simulation
    """
    image_quality = simulation.image_quality

    # creates a base video file to save to
    video_path = simulation.path + 'network_video.avi'
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc("M", "J", "P", "G"), 1.0, image_quality)

    # loops over all images and writes them to the base video file
    for i in range(1, simulation.image_counter + 1):
        path = simulation.path + "image_" + str(i) + "_slice_0" + ".png"
        image = cv2.imread(path)
        out.write(image)

    # releases the file
    out.release()
    print("Done")

def create_csv(simulation):
    """ Outputs a .csv file of important Cell
        instance variables from each cell
    """
    # opens .csv file
    new_file = open(simulation.path + "network_values_" + str(int(simulation.current_step)) + ".csv", "w", newline="")
    csv_write = csv.writer(new_file)
    csv_write.writerow(['X_position', 'Y_position', 'Z_position', 'Motion', 'Radius', 'FGFR', 'ERK', 'GATA6', 'NANOG',
                        'State', 'Differentiation_counter', 'Division_counter', 'Death_counter', 'Boolean_counter'])

    # each row is a different cell
    for i in range(len(simulation.cells)):
        location_x = round(simulation.cells[i].location[0], 8)
        location_y = round(simulation.cells[i].location[1], 8)
        location_z = round(simulation.cells[i].location[2], 8)
        radius = simulation.cells[i].radius
        motion = simulation.cells[i].motion
        fgfr = simulation.cells[i].booleans[0]
        erk = simulation.cells[i].booleans[1]
        gata = simulation.cells[i].booleans[2]
        nanog = simulation.cells[i].booleans[3]
        state = simulation.cells[i].state
        diff_counter = simulation.cells[i].diff_counter
        div_counter = simulation.cells[i].div_counter
        death_counter = simulation.cells[i].death_counter
        bool_counter = simulation.cells[i].boolean_counter

        # writes the row for the cell
        csv_write.writerow([location_x, location_y, location_z, radius, motion, fgfr, erk, gata, nanog, state,
                            diff_counter, div_counter, death_counter, bool_counter])

def save_file(simulation):
    """ Saves the simulation txt files
        and image files
    """
    # saves the .csv file with all the key information for each cell
    create_csv(simulation)

    # draws the image of the simulation
    draw_image(simulation)