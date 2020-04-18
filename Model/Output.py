from PIL import Image, ImageDraw
import cv2
import csv
import numpy as np


def draw_image(simulation):
    """ Turns the graph into an image at each timestep
    """
    # increases the image counter by 1 each time this is called
    simulation.image_counter += 1

    # get the dilation of the image for the correctly sizing the image
    dilation_x = simulation.quality[0] / simulation.size[0]
    dilation_y = simulation.quality[1] / simulation.size[1]

    # draws the background of the image
    base = Image.new("RGB", simulation.quality[0:2], color=(255, 255, 255))
    image = ImageDraw.Draw(base)

    # bounds of the simulation used for drawing lines
    bounds = np.array([[0, 0],[0, simulation.size[1]],[simulation.size[0],simulation.size[1]],[simulation.size[0], 0]])

    # loops over all of the cells and draws the nucleus and radius
    for i in range(len(simulation.cells)):
        # get location in 2D
        x = dilation_x * simulation.cells[i].location[0]
        y = dilation_y * simulation.cells[i].location[1]

        # gets membrane sizing
        membrane_x = dilation_x * simulation.cells[i].radius
        membrane_y = dilation_y * simulation.cells[i].radius

        # coloring of the cells
        if simulation.cells[i].state == "Pluripotent":
            color = 'green'
        else:
            color = 'red'

        # draw the circle representing the cell
        membrane_circle = (x - membrane_x, y - membrane_y, x + membrane_x, y + membrane_y)
        image.ellipse(membrane_circle, outline="black", fill=color)

    # loops over all of the bounds and draws lines to represent the grid
    for i in range(len(bounds)):
        # get the bound sizing
        x0 = dilation_x * bounds[i][0]
        y0 = dilation_y * bounds[i][1]

        # get the bounds as lines
        if i < len(bounds) - 1:
            x1 = dilation_x * bounds[i + 1][0]
            y1 = dilation_y * bounds[i + 1][1]
        else:
            x1 = dilation_x * bounds[0][0]
            y1 = dilation_y * bounds[0][1]

        # draw the lines
        lines = (x0, y0, x1, y1)
        color = 'black'
        image.line(lines, fill=color, width=10)

    # saves the image as a .png
    base.save(simulation.path + "network_image_" + str(int(simulation.time_counter)) + ".png", 'PNG')

def image_to_video(simulation):
    """ Creates a video out of all the png images at
        the end of the simulation
    """
    image_quality = simulation.quality
    image_size = (image_quality * 1500, image_quality * 1500)

    # creates a base video file to save to
    video_path = simulation.path + 'network_video.avi'
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc("M","J","P","G"), 1.0, image_size)

    # loops over all images and writes them to the base video file
    for i in range(simulation.image_counter):
        path = simulation.path + 'network_image_' + str(i) + ".png"
        image = cv2.imread(path)
        out.write(image)

    # releases the file
    out.release()

def create_csv(simulation):
    """ Outputs a .csv file of important Cell
        instance variables from each cell
    """
    # opens .csv file
    new_file = open(simulation.path + "network_values_" + str(int(simulation.time_counter)) + ".csv", "w")
    csv_write = csv.writer(new_file)
    csv_write.writerow(['X_position', 'Y_position', 'Z_position', 'X_velocity', 'Y_velocity', 'Z_velocity', 'Motion',
                        'Mass', 'Radius', 'FGFR', 'ERK', 'GATA6', 'NANOG', 'State', 'Differentiation_counter',
                        'Division_counter', 'Death_counter'])

    # each row is a different cell
    for i in range(len(simulation.cells)):
        x_pos = round(simulation.cells[i].location[0], 1)
        y_pos = round(simulation.cells[i].location[1], 1)
        z_pos = round(simulation.cells[i].location[2], 1)
        x_vel = round(simulation.cells[i].velocity[0], 1)
        y_vel = round(simulation.cells[i].velocity[1], 1)
        z_vel = round(simulation.cells[i].velocity[2], 1)
        motion = simulation.cells[i].motion
        mass = simulation.cells[i].mass
        radius = simulation.cells[i].radius
        fgfr = simulation.cells[i].booleans[0]
        erk = simulation.cells[i].booleans[1]
        gata = simulation.cells[i].booleans[2]
        nanog = simulation.cells[i].booleans[3]
        state = simulation.cells[i].state
        diff = round(simulation.cells[i].diff_counter, 1)
        div = round(simulation.cells[i].div_counter, 1)
        death = round(simulation.cells[i].death_counter, 1)

        # writes the row for the cell
        csv_write.writerow([x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, motion, mass, radius, fgfr, erk, gata, nanog,
                            state, diff, div, death])

def save_file(simulation):
    """ Saves the simulation txt files
        and image files
    """
    # saves the .csv file with all the key information for each cell
    create_csv(simulation)

    # draws the image of the simulation
    draw_image(simulation)