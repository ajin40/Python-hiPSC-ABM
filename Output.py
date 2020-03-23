from PIL import Image, ImageDraw
import cv2
import csv
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np


def draw_cell_image(self, network, path):
    """ Turns the graph into an image at each timestep
    """
    # increases the image counter by 1 each time this is called
    self.image_counter += 1

    # get list of all objects/nodes in the simulation
    cells = list(network.nodes)

    # draws the background of the image
    image1 = Image.new("RGB", (1500, 1500), color=(130, 130, 130))
    # image1 = Image.new("RGB", (1500, 1500), color="black")
    draw = ImageDraw.Draw(image1)

    # bounds of the simulation used for drawing patch
    # inherit
    bounds_xy = [[-12, -12], [-12, self.size[1] + 12], [self.size[0] + 12, self.size[1] + 12], [self.size[0] + 12, -12]]

    # loops over all of the cells/nodes and draws a circle with corresponding color
    for i in range(len(cells)):
        node = cells[i]
        x, y, z = node.location
        r1 = 0.3 * node.radius
        r2 = node.radius

        # if node.state == "Pluripotent" or node.state == "Differentiated":
        #     if node.booleans[3] == 0 and node.booleans[2] == 1:
        #         col = (255, 0, 0)
        #     elif node.booleans[3] == 1 and node.booleans[2] == 0:
        #         col = (17, 235, 24)
        #     elif node.booleans[3] == 1 and node.booleans[2] == 1:
        #         col = (245, 213, 7)
        #     else:
        #         col = (60, 0, 255)

        if node.state == "Pluripotent":
            col = 'white'
        else:
            col = 'black'

        out = "black"
        draw.ellipse((x - r1 + 250, y - r1 + 250, x + r1 + 250, y + r1 + 250), outline=out, fill=col)
        draw.ellipse((x - r2 + 250, y - r2 + 250, x + r2 + 250, y + r2 + 250), outline=out)


    # loops over all of the bounds and draws lines to represent the grid
    for i in range(len(bounds_xy)):
        x, y = bounds_xy[i]
        if i < len(bounds_xy) - 1:
            x1, y1 = bounds_xy[i + 1]
        else:
            x1, y1 = bounds_xy[0]
        r = 4
        draw.ellipse((x - r + 250, y - r + 250, x + r + 250, y + r + 250), outline='white', fill='white')
        draw.line((x + 250, y + 250, x1 + 250, y1 + 250), fill='white', width=10)

    # saves the image as a .png
    image1.save(path + ".png", 'PNG')




def image_to_video(self):
    """ Creates a video out of all the png images at
        the end of the simulation
    """
    # gets base path
    base_path = self.path + self.sep + self.name + self.sep

    # image list to hold all image objects
    img_array = []
    # loops over all images created
    for i in range(self.image_counter - 1):
        path = base_path + 'network_image_' + str(i) + ".png"
        img = cv2.imread(path)
        img_array.append(img)

    # output file for the video
    out = cv2.VideoWriter(base_path + 'network_video.avi', cv2.VideoWriter_fourcc("M", "J", "P", "G"), 1.0, (640, 480))

    # adds image to output file
    for i in range(len(img_array)):
        out.write(img_array[i])

    # releases the file
    out.release()


def location_to_text(self, path):
    """ Outputs a txt file of the cell coordinates and the boolean values
    """
    # opens file
    new_file = open(path, "w")

    # initializes csv file
    object_writer = csv.writer(new_file)
    object_writer.writerow(['x_coord', 'y_coord', 'z_coord', 'x_vel', 'y_vel', 'z_vel', 'Motion', 'Mass', 'Radius',
                            'FGFR', 'ERK', 'GATA6', 'NANOG', 'State', 'Diff_Timer', 'Div_Timer', 'Death_Timer'])

    # writes for each cell. Lists the last four boolean values
    for i in range(len(self.cells)):
        x_coord = str(round(self.cells[i].location[0], 1))
        y_coord = str(round(self.cells[i].location[1], 1))
        z_coord = str(round(self.cells[i].location[2], 1))
        x_vel = str(round(self.cells[i].velocity[0], 1))
        y_vel = str(round(self.cells[i].velocity[1], 1))
        z_vel = str(round(self.cells[i].velocity[2], 1))
        motion = str(self.cells[i].motion)
        mass = str(self.cells[i].mass)
        radius = str(self.cells[i].radius)
        fgfr = str(self.cells[i].booleans[0])
        erk = str(self.cells[i].booleans[1])
        gata = str(self.cells[i].booleans[2])
        nanog = str(self.cells[i].booleans[3])
        state = str(self.cells[i].state)
        diff = str(round(self.cells[i].diff_timer, 1))
        div = str(round(self.cells[i].division_timer, 1))
        death = str(round(self.cells[i].death_timer, 1))

        object_writer.writerow([x_coord, y_coord, z_coord, x_vel, y_vel, z_vel, motion, mass, radius, fgfr, erk, gata, nanog,
                                state, diff, div, death])


def save_file(self):
    """ Saves the simulation txt files
        and image files
    """
    # get the base path
    base_path = self.path + self.sep + self.name + self.sep

    # saves the txt file with all the key information
    n2_path = base_path + "network_values" + str(int(self.time_counter)) + ".csv"
    location_to_text(self, n2_path)

    # draws the image of the simulation
    draw_cell_image(self, self.network, base_path + "network_image_" + str(int(self.time_counter)))

    # voronoi(self, base_path + "network_image_" + str(int(self.time_counter)))
    self.image_counter += 1



def voronoi(self, path):

    points = np.empty((0, 2), int)
    for i in range(len(self.cells)):
        points = np.append(points, [self.cells[i].location[:2]], axis=0)

    vor = Voronoi(points)
    fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='green', line_width=2, line_alpha=0.6, point_size=5)

    fig.savefig(path, quality=100)
