from PIL import Image, ImageDraw
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

        # bounds of the simulation used for drawing lines
        corner_1 = [0, 0]
        corner_2 = [0, simulation.size[1]]
        corner_3 = [simulation.size[0], simulation.size[1]]
        corner_4 = [simulation.size[0], 0]
        bounds = np.array([corner_1, corner_2, corner_3, corner_4])

        # draws the background of the image
        base = Image.new("RGB", simulation.image_quality[0:2], simulation.background_color)
        image = ImageDraw.Draw(base)

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

            # draw the circle representing the cell
            membrane_circle = (x - x_radius, y - y_radius, x + x_radius, y + y_radius)
            image.ellipse(membrane_circle, fill=color, outline="black")

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

                # draw the lines, width is kinda arbitrary
                lines = (x0, y0, x1, y1)
                width = int((simulation.image_quality[0] + simulation.image_quality[1]) / 500)
                image.line(lines, fill=simulation.bound_color, width=width)

        # saves the image as a .png
        image_path = simulation.path + simulation.name + "_image_" + str(int(simulation.current_step)) + ".png"
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
                                              simulation.image_quality)


def finish_files(simulation):
    """ Closes any necessary files and
        prints an ending statement
    """
    # close out the running video file
    simulation.video_object.release()

    # end statement
    print("The simulation is finished. May the force be with you.")
