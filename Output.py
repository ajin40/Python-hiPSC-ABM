#########################################################
# Name:    Output                                       #
# Author:  Jack Toppen                                  #
# Date:    3/17/20                                      #
#########################################################
from PIL import Image, ImageDraw
import cv2
import csv

"""
This contains all important functions for handling the output
of data from the simulation. Including images, CSV, and video
"""


def draw_cell_image_xy(self, network, path):
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
    bounds_xz = [[-12, -12], [-12, self.size[2] + 12], [self.size[0] + 12, self.size[2] + 12], [self.size[0] + 12, -12]]
    bounds_yz = [[-12, -12], [-12, self.size[2] + 12], [self.size[1] + 12, self.size[2] + 12], [self.size[1] + 12, -12]]

    # loops over all of the cells/nodes and draws a circle with corresponding color
    for i in range(len(cells)):
        node = cells[i]
        x, y, z = node.location
        r = node.nuclear_radius

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
        draw.ellipse((x - r + 250, y - r + 250, x + r + 250, y + r + 250), outline=out, fill=col)


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

def draw_cell_image_xz(self, network, path):
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
    bounds_xz = [[-12, -12], [-12, self.size[2] + 12], [self.size[0] + 12, self.size[2] + 12], [self.size[0] + 12, -12]]
    bounds_yz = [[-12, -12], [-12, self.size[2] + 12], [self.size[1] + 12, self.size[2] + 12], [self.size[1] + 12, -12]]

    # loops over all of the cells/nodes and draws a circle with corresponding color
    for i in range(len(cells)):
        node = cells[i]
        x, y, z = node.location
        r = node.nuclear_radius

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
        draw.ellipse((x - r + 250, z - r + 250, x + r + 250, z + r + 250), outline=out, fill=col)


    # loops over all of the bounds and draws lines to represent the grid
    for i in range(len(bounds_xz)):
        x, y = bounds_xz[i]
        if i < len(bounds_xz) - 1:
            x1, y1 = bounds_xz[i + 1]
        else:
            x1, y1 = bounds_xz[0]
        r = 4
        draw.ellipse((x - r + 250, y - r + 250, x + r + 250, y + r + 250), outline='white', fill='white')
        draw.line((x + 250, y + 250, x1 + 250, y1 + 250), fill='white', width=10)

    # saves the image as a .png
    image1.save(path + ".png", 'PNG')

def draw_cell_image_yz(self, network, path):
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
    bounds_xz = [[-12, -12], [-12, self.size[2] + 12], [self.size[0] + 12, self.size[2] + 12], [self.size[0] + 12, -12]]
    bounds_yz = [[-12, -12], [-12, self.size[2] + 12], [self.size[1] + 12, self.size[2] + 12], [self.size[1] + 12, -12]]

    # loops over all of the cells/nodes and draws a circle with corresponding color
    for i in range(len(cells)):
        node = cells[i]
        x, y, z = node.location
        r = node.nuclear_radius

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
        draw.ellipse((y - r + 250, z - r + 250, y + r + 250, z + r + 250), outline=out, fill=col)


    # loops over all of the bounds and draws lines to represent the grid
    for i in range(len(bounds_yz)):
        x, y = bounds_yz[i]
        if i < len(bounds_yz) - 1:
            x1, y1 = bounds_yz[i + 1]
        else:
            x1, y1 = bounds_yz[0]
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
    for i in range(self.image_counter):
        img = cv2.imread(base_path + 'network_image' + str(i) + ".png")
        img_array.append(img)

    # output file for the video
    out = cv2.VideoWriter(base_path + 'network_video.avi', cv2.VideoWriter_fourcc("M", "J", "P", "G"), 1.0, (1500, 1500))

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
    object_writer.writerow(['x_coord', 'y_coord', 'z_coord', 'Motion', 'Mass', 'Nuclear_Radius', 'Cytoplasm_Radius',
                            'FGFR', 'ERK', 'GATA6', 'NANOG', 'State', 'Diff_Timer', 'Div_Timer', 'Death_Timer'])

    # writes for each cell. Lists the last four boolean values
    for i in range(len(self.cells)):
        x_coord = str(round(self.cells[i].location[0], 1))
        y_coord = str(round(self.cells[i].location[1], 1))
        z_coord = str(round(self.cells[i].location[2], 1))
        motion = str(self.cells[i].motion)
        mass = str(self.cells[i].mass)
        nuclear = str(self.cells[i].nuclear_radius)
        cytoplasm = str(self.cells[i].cytoplasm_radius)
        fgfr = str(self.cells[i].booleans[0])
        erk = str(self.cells[i].booleans[1])
        gata = str(self.cells[i].booleans[2])
        nanog = str(self.cells[i].booleans[3])
        state = str(self.cells[i].state)
        diff = str(round(self.cells[i].diff_timer, 1))
        div = str(round(self.cells[i].division_timer, 1))
        death = str(round(self.cells[i].death_timer, 1))

        object_writer.writerow([x_coord, y_coord, z_coord, motion, mass, nuclear, cytoplasm, fgfr, erk, gata, nanog,
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
    draw_cell_image_xy(self, self.network, base_path + "network_image_xy" + str(int(self.time_counter)))
    draw_cell_image_xz(self, self.network, base_path + "network_image_xz" + str(int(self.time_counter)))
    draw_cell_image_yz(self, self.network, base_path + "network_image_yz" + str(int(self.time_counter)))
