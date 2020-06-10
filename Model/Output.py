from PIL import Image, ImageDraw
import cv2
import csv
import time
import memory_profiler
import numpy as np


def step_image(simulation):
    """ Turns the graph into an image at each timestep
    """
    if simulation.output_images:
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
            for j in range(simulation.number_cells):
                location = simulation.cell_locations[j]
                radius = simulation.cell_radii[j]

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

                # coloring of the cells based on what mode the user selects
                if simulation.color_mode:
                    if simulation.cell_states[j] == "Pluripotent":
                        color = simulation.pluri_color
                    else:
                        color = simulation.diff_color

                else:
                    if simulation.cell_states[j] == "Differentiated":
                        color = simulation.diff_color
                    elif simulation.cell_booleans[j][2] == 1 and simulation.cell_booleans[j][3] == 1:
                        color = simulation.pluri_both_high_color
                    elif simulation.cell_booleans[j][2] == 1:
                        color = simulation.pluri_gata6_high_color
                    elif simulation.cell_booleans[j][3] == 1:
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

                # draw the lines
                lines = (x0, y0, x1, y1)
                width = int((simulation.image_quality[0] + simulation.image_quality[0]) / 500)
                image.line(lines, fill=simulation.bound_color, width=width)

            # saves the image as a .png
            image_name = simulation.name + "_image_" + str(int(simulation.current_step))+"_slice_"+str(int(i)) + ".png"
            base.save(simulation.path + image_name, 'PNG')

            # moves to the next slice location
            lower_slice += thickness
            upper_slice += thickness


def image_to_video(simulation):
    """ Creates a video out of all the png images at
        the end of the simulation
    """
    # only make a video if images exist
    if simulation.output_images:
        image_quality = simulation.image_quality

        # creates a base video file to save to
        video_path = simulation.path + simulation.name + '_video.avi'
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc("M", "J", "P", "G"), 1.0, image_quality)

        # loops over all images and writes them to the base video file
        for i in range(simulation.beginning_step, simulation.beginning_step + simulation.image_counter + 1):
            path = simulation.path + simulation.name + "_image_" + str(i) + "_slice_0" + ".png"
            image = cv2.imread(path)
            out.write(image)

        # releases the file
        out.release()
    # tells the use the simulation is over
    print("Done")


def step_csv(simulation):
    """ Outputs a .csv file of important Cell
        instance variables from each cell
    """
    if simulation.output_csvs:
        # opens .csv file
        file_name = simulation.path + simulation.name + "_values_" + str(int(simulation.current_step)) + ".csv"
        new_file = open(file_name, "w", newline="")
        csv_write = csv.writer(new_file)
        csv_write.writerow(['X_position', 'Y_position', 'Z_position', 'Radius', 'Motion', 'FGFR', 'ERK', 'GATA6',
                            'NANOG', 'State', 'Differentiation_counter', 'Division_counter', 'Death_counter',
                            'Boolean_counter'])

        # each row is a different cell
        for i in range(simulation.number_cells):
            location_x = round(simulation.cell_locations[i][0], 8)
            location_y = round(simulation.cell_locations[i][1], 8)
            location_z = round(simulation.cell_locations[i][2], 8)
            radius = simulation.cell_radii[i]
            motion = simulation.cell_motion[i]
            fgfr = simulation.cell_booleans[i][0]
            erk = simulation.cell_booleans[i][1]
            gata = simulation.cell_booleans[i][2]
            nanog = simulation.cell_booleans[i][3]
            state = simulation.cell_states[i]
            diff_counter = simulation.cell_diff_counter[i]
            div_counter = simulation.cell_div_counter[i]
            death_counter = simulation.cell_death_counter[i]
            bool_counter = simulation.cell_bool_counter[i]

            # writes the row for the cell
            csv_write.writerow([location_x, location_y, location_z, radius, motion, fgfr, erk, gata, nanog, state,
                                diff_counter, div_counter, death_counter, bool_counter])


def simulation_data(simulation):
    """ Adds a new line to the running csv for
        data amount the simulation such as
        memory, step time, number of cells,
        and various other stats.
    """
    # opens the file
    with open(simulation.data_path, "a", newline="") as file_object:
        csv_object = csv.writer(file_object)

        # get all of the values necessary to write to the data file
        step = simulation.current_step
        cells = simulation.number_cells
        step_time = time.time() - simulation.step_start
        memory = memory_profiler.memory_usage(max_usage=True)
        ud = simulation.ud_time
        cn = simulation.cn_time
        nd = simulation.nd_time
        cd = simulation.cd_time
        cds = simulation.cds_time
        cm = simulation.cm_time
        uc = simulation.uc_time
        ucq = simulation.ucq_time
        hm = simulation.hm_time

        # write the row with the corresponding values
        csv_object.writerow([step, cells, step_time, memory, ud, cn, nd, cd, cds, cm, uc, ucq, hm])
