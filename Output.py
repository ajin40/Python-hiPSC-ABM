#########################################################
# Name:    Output                                       #
# Author:  Jack Toppen                                  #
# Date:    3/4/20                                       #
#########################################################
from PIL import Image, ImageDraw
import cv2
import csv


def draw_cell_image(self, network, path):
    """Turns the graph into an image at each timestep
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
    bounds = self.bounds

    # loops over all of the cells/nodes and draws a circle with corresponding color
    for i in range(len(cells)):
        node = cells[i]
        x, y = node.location
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

        if node.state == "Differentiated":
            col = (255, 255, 255)

        if node.state == "Pluripotent":
            col = 'white'

        else:
            col = 'black'

        out = "black"
        draw.ellipse((x - r + 250, y - r + 250, x + r + 250, y + r + 250), outline=out, fill=col)

    # loops over all of the bounds and draws lines to represent the grid
    for i in range(len(bounds)):
        x, y = bounds[i]
        if i < len(bounds) - 1:
            x1, y1 = bounds[i + 1]
        else:
            x1, y1 = bounds[0]
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
    """Outputs a txt file of the cell coordinates and the boolean values
    """
    # opens file
    new_file = open(path, "w")

    object_writer = csv.writer(new_file)
    object_writer.writerow(['ID', 'x_coord', 'y_coord', 'State', 'FGFR', 'ERK', 'GATA6', 'NANOG', 'Motion',
                            'diff_count', 'div_count'])

    for i in range(len(self.objects)):
        ID = str(self.objects[i].ID)
        x_coord = str(round(self.objects[i].location[0], 1))
        y_coord = str(round(self.objects[i].location[1], 1))
        x1 = str(self.objects[i].booleans[0])
        x2 = str(self.objects[i].booleans[1])
        x3 = str(self.objects[i].booleans[2])
        x4 = str(self.objects[i].booleans[3])
        diff = str(round(self.objects[i].diff_timer, 1))
        div = str(round(self.objects[i].division_timer, 1))
        state = str(self.objects[i].state)
        motion = str(self.objects[i].motion)

        object_writer.writerow([ID, x_coord, y_coord, state, x1, x2, x3, x4, motion, diff, div])


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
    draw_cell_image(self, self.network, base_path + "network_image" + str(int(self.time_counter)))