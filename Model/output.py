"""

output.py serves as a way of separating the following functions from the Simulation class. These
functions will prepare the files to store data about the efficiency of the model, information
at each step for all of the cells, images from each step, and a video.

"""
from PIL import Image, ImageDraw
from matplotlib.cm import ScalarMappable
import cv2
import csv
import time
import memory_profiler
import numpy as np
import pickle


def step_image(simulation):
    """ Creates an image representation of the
        space in which the cells reside and
        the extracellular gradient.
    """
    # continue if images are desired
    if simulation.output_images:
        # get the dilation of the image for the correctly sizing the image
        dilation_x = simulation.image_quality[0] / simulation.size[0]
        dilation_y = simulation.image_quality[1] / simulation.size[1]

        # draws the background of the image
        base = Image.new("RGBA", simulation.image_quality[0:2], simulation.background_color)
        image = ImageDraw.Draw(base)

        # get the fgf4 gradient array, resize to a 2D array, and normalize the concentrations
        fgf4_array = simulation.fgf4_values
        fgf4_array = np.reshape(fgf4_array, (fgf4_array.shape[0], fgf4_array.shape[1]))
        fgf4_array /= simulation.max_fgf4

        # create a gradient image if desired
        if simulation.output_gradient:
            # create a color map object that is used to turn the fgf4 array into a rgba array
            cmap_object = ScalarMappable(cmap='Blues')
            fgf4_as_rgba = cmap_object.to_rgba(fgf4_array, bytes=True, norm=False)

            # create an image from the fgf4 rgba array
            cmap_base = Image.fromarray(fgf4_as_rgba, mode="RGBA")
            cmap_base = cmap_base.resize(simulation.image_quality[0:2], resample=Image.BICUBIC)
            cmap_base = cmap_base.transpose(Image.TRANSPOSE)
            cmap_image = ImageDraw.Draw(cmap_base)

        # loops over all of the cells and draws the nucleus and radius
        for i in range(simulation.number_cells):
            location = simulation.cell_locations[i]
            radius = simulation.cell_radii[i]

            # determine the radius based on the pixel dilation
            x_radius = dilation_x * radius
            y_radius = dilation_y * radius

            # get location in 2D with image size dilation
            x = dilation_x * location[0]
            y = dilation_y * location[1]

            # coloring of the cells based on what mode the user selects
            # this mode is just showing pluripotent and differentiated cells
            if simulation.color_mode:
                if simulation.cell_states[i] == "Pluripotent":
                    color = (22, 252, 32)
                else:
                    color = (230, 0, 0)

            # this mode is showing color based on the finite dynamical system
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

            # draw the circle representing the cell to both the normal image and the colormap image
            membrane_circle = (x - x_radius, y - y_radius, x + x_radius, y + y_radius)
            image.ellipse(membrane_circle, fill=color, outline="black")

            # draw on gradient image if desired
            if simulation.output_gradient:
                cmap_image.ellipse(membrane_circle, outline='gray', width=1)

        # get the image path
        image_path = simulation.path + simulation.name + "_image_" + str(int(simulation.current_step)) + ".png"

        # draw on gradient image if desired
        if simulation.output_gradient:
            # paste the images to a new background image
            background = Image.new('RGBA', (base.width + cmap_base.width, base.height))
            background.paste(base, (0, 0))
            background.paste(cmap_base, (base.width, 0))

            # save the image
            background.save(image_path, 'PNG')

        else:
            # save the image
            base.save(image_path, 'PNG')

        # write it to the video object
        add_image = cv2.imread(image_path)
        simulation.video_object.write(add_image)


def step_csv(simulation):
    """ Outputs a .csv file containing information
        about each cell with each row corresponding
        to a cell
    """
    # only create this file if desired
    if simulation.output_csvs:
        # open a new file
        file_path = simulation.path + simulation.name + "_values_" + str(int(simulation.current_step)) + ".csv"
        with open(file_path, "w", newline="") as new_file:
            csv_file = csv.writer(new_file)

            # write the header of the csv
            csv_file.writerow(['X_position', 'Y_position', 'Z_position', 'Radius', 'Motion', 'FGFR', 'ERK', 'GATA6',
                               'NANOG', 'State', 'Differentiation_counter', 'Division_counter', 'Death_counter',
                               'FDS_counter'])

            # turn the cell holder arrays into a list of rows to add to the csv file
            cell_data = list(zip(simulation.cell_locations[:, 0], simulation.cell_locations[:, 1],
                                 simulation.cell_locations[:, 2], simulation.cell_radii, simulation.cell_motion,
                                 simulation.cell_fds[:, 0], simulation.cell_fds[:, 1], simulation.cell_fds[:, 2],
                                 simulation.cell_fds[:, 3], simulation.cell_states, simulation.cell_diff_counter,
                                 simulation.cell_div_counter, simulation.cell_death_counter,
                                 simulation.cell_fds_counter))

            # write the list containing the new csv rows
            csv_file.writerows(cell_data)


def simulation_data(simulation):
    """ Adds a new line to the running csv for
        data amount the simulation such as
        memory, step time, number of cells,
        and various other stats.
    """
    # calculate the step time and the memory
    step_time = time.time() - simulation.step_start
    memory = memory_profiler.memory_usage(max_usage=True)

    # write the row with the corresponding values
    simulation.csv_object.writerow([simulation.current_step, simulation.number_cells, step_time, memory,
                                    simulation.update_diffusion_time, simulation.check_neighbors_time,
                                    simulation.nearest_time, simulation.cell_motility_time,
                                    simulation.cell_update_time, simulation.update_queue_time,
                                    simulation.handle_movement_time, simulation.jkr_neighbors_time,
                                    simulation.get_forces_time, simulation.apply_forces_time])


def initialize_csv(simulation):
    """ Opens a csv file to be written
        to each step with stats
    """
    # create a CSV file used to hold information about run time, number of cells, memory, and various other statistics
    data_path = simulation.path + simulation.name + "_data.csv"

    # open the file and create a csv object and write a header as the first line
    file_object = open(data_path, "w", newline="")
    simulation.csv_object = csv.writer(file_object)
    simulation.csv_object.writerow(["Step Number", "Number Cells", "Step Time", "Memory (MB)", "update_diffusion",
                                    "check_neighbors", "nearest_diff", "cell_motility", "update_cells",
                                    "update_cell_queue", "handle_movement", "jkr_neighbors", "get_forces",
                                    "apply_forces"])


def initialize_video(simulation):
    """ Opens the video file to be written
        to each step with the image produced
    """
    # creates a video file that can be written to each step
    video_path = simulation.path + simulation.name + '_video.avi'

    # determine the appropriate size of the video in pixels
    if simulation.output_gradient:
        image_size = (simulation.image_quality[0] * 2, simulation.image_quality[1])
    else:
        image_size = (simulation.image_quality[0], simulation.image_quality[1])

    simulation.video_object = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc("M", "J", "P", "G"), simulation.fps,
                                              image_size)


def temporary(simulation):
    """ Pickle a copy of the simulation class that can be used
        to continue a past simulation without losing information
    """
    with open(simulation.path + simulation.name + '_temp.pkl', 'wb') as file:
        pickle.dump(simulation, file, -1)


def finish_files(simulation):
    """ Closes any necessary files and
        prints an ending statement
    """
    # close out the running video file
    simulation.video_object.release()

    # end statement
    print("The simulation is finished. May the force be with you.")
