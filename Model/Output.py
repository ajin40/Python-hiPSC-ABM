from PIL import Image, ImageDraw
from matplotlib.cm import ScalarMappable
import cv2
import csv
import time
import memory_profiler
import numpy as np


def step_image(simulation):
    """ Creates an image representation of the
        space in which the cells reside
    """
    if simulation.output_images:
        # get the dilation of the image for the correctly sizing the image
        dilation_x = simulation.image_quality[0] / simulation.size[0]
        dilation_y = simulation.image_quality[1] / simulation.size[1]

        # draws the background of the image
        base = Image.new("RGBA", simulation.image_quality[0:2], simulation.background_color)
        image = ImageDraw.Draw(base)

        # get the fgf4 gradient array, resize to a 2D array, and normalize the concentrations
        fgf4_array = simulation.extracellular[0].diffuse_values
        fgf4_array = np.reshape(fgf4_array, (fgf4_array.shape[0], fgf4_array.shape[1]))
        max_value = np.amax(fgf4_array)
        if max_value != 0:
            fgf4_array *= max_value ** -1

        # create a color map object that is used to turn the fgf4 array into a rgba array
        cmap_object = ScalarMappable(cmap='jet')
        fgf4_as_rgba = cmap_object.to_rgba(fgf4_array, bytes=True)

        # create an image from the fgf4 rgba array
        cmap_base = Image.fromarray(fgf4_as_rgba, mode="RGBA")
        cmap_base = cmap_base.resize(simulation.image_quality[0:2], resample=Image.NEAREST)
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
                    color = simulation.pluri_color
                else:
                    color = simulation.diff_color

            # this mode is showing color based on the finite dynamical system
            else:
                if simulation.cell_states[i] == "Differentiated":
                    color = simulation.diff_color
                elif simulation.cell_fds[i][2] == 1 and simulation.cell_fds[i][3] == 1:
                    color = simulation.pluri_both_high_color
                elif simulation.cell_fds[i][2] == 1:
                    color = simulation.pluri_gata6_high_color
                elif simulation.cell_fds[i][3] == 1:
                    color = simulation.pluri_nanog_high_color
                else:
                    color = simulation.pluri_color

            # draw the circle representing the cell to both the normal image and the colormap image
            membrane_circle = (x - x_radius, y - y_radius, x + x_radius, y + y_radius)
            image.ellipse(membrane_circle, fill=color, outline="black")
            cmap_image.ellipse(membrane_circle, outline='white', width=4)

        # paste the images to a new background image
        background = Image.new('RGBA', (base.width + cmap_base.width, base.height))
        background.paste(base, (0, 0))
        background.paste(cmap_base, (base.width, 0))

        # saves the image as a .png
        image_path = simulation.path + simulation.name + "_image_" + str(int(simulation.current_step)) + ".png"
        background.save(image_path, 'PNG')

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
                               'Boolean_counter'])

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
                                    simulation.nearest_diff_time, simulation.cell_death_time,
                                    simulation.cell_diff_surround_time, simulation.cell_motility_time,
                                    simulation.cell_update_time, simulation.update_queue_time,
                                    simulation.handle_movement_time])


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
                                    "check_neighbors", "nearest_diff", "cell_death", "diff_surround",
                                    "cell_motility", "update_cells", "update_cell_queue", "handle_movement"])


def initialize_video(simulation):
    """ Opens the video file to be written
        to each step with the image produced
    """
    # creates a video file that can be written to each step
    video_path = simulation.path + simulation.name + '_video.avi'
    simulation.video_object = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc("M", "J", "P", "G"), simulation.fps,
                                              (simulation.image_quality[0] * 2, simulation.image_quality[1]))


def finish_files(simulation):
    """ Closes any necessary files and
        prints an ending statement
    """
    # close out the running video file
    simulation.video_object.release()

    # end statement
    print("The simulation is finished. May the force be with you.")
