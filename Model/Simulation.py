import numpy as np
import igraph

import Input


# used to hold all values necessary to the simulation as it moves from one step to the next
class Simulation:
    def __init__(self, template_location):
        # open the .txt template file that contains the initial parameters
        with open(template_location) as template_file:
            lines = template_file.readlines()

        # the following lines correspond to lines of template file, so anything implemented to the template file
        # can be added to the class initialization.
        # general parameters
        self.name = lines[8][2:-3]  # the name of the simulation, used to name files in the output directory
        self.output_direct = lines[11][2:-3]   # the output directory where simulation is placed
        self.parallel = eval(lines[14][2:-3]) # whether the model is using parallel GPU processing for certain functions
        self.size = np.array(eval(lines[17][2:-3]))  # the dimensions of the space (in meters) the cells reside in
        self.num_GATA6 = int(lines[20][2:-3])   # the number of GATA6 high cells to begin the simulation
        self.num_NANOG = int(lines[23][2:-3])   # the number of NANOG high cells to being the simulation

        # modes
        self.output_csvs = eval(lines[41][2:-3])  # whether or not to produce csvs with cell information
        self.output_images = eval(lines[38][2:-3])   # whether or not to produce images
        self.continuation = eval(lines[44][2:-3])   # continuation of a previous simulation
        self.csv_to_images = eval(lines[47][2:-3])  # turn a collection of csvs to images
        self.images_to_video = eval(lines[50][2:-3])    # turn a collection of images into a video

        # timing
        self.beginning_step = int(lines[57][2:-3])  # the step the simulation starts on, used for certain modes
        self.end_step = int(lines[60][2:-3])   # the last step of a simulation
        self.time_step_value = float(lines[63][2:-3])   # the real-time value of each step
        self.fds_thresh = int(lines[66][2:-3])  # the threshold (in steps) for updating the finite dynamical system
        self.pluri_div_thresh = int(lines[69][2:-3])  # the division threshold (in steps) of a pluripotent cell
        self.pluri_to_diff = int(lines[72][2:-3])  # the differentiation threshold (in steps) of a pluripotent cell
        self.diff_div_thresh = int(lines[75][2:-3])  # the division threshold (in steps) of a differentiated cell
        self.death_thresh = int(lines[78][2:-3])  # the death threshold (in steps) for cell death

        # intercellular
        self.lonely_cell = int(lines[93][2:-3])  # if the number of neighbors is below this threshold, a cell is lonely
        self.contact_inhibit = int(lines[96][2:-3])  # if the number of neighbors is below this threshold, no inhibition
        self.move_thresh = int(lines[99][2:-3])  # if the number of neighbors is above this threshold, inhibit motion
        self.diff_surround = int(lines[102][2:-3])  # the number of diff cells needed to help induce differentiation

        # extracellular
        self.diffuse = float(lines[108][2:-3])  # the diffusion constant
        self.dx = eval(lines[111][2:-3])[0]  # the diffusion resolution along the x-axis
        self.dy = eval(lines[111][2:-3])[1]  # the diffusion resolution along the y-axis
        self.dz = eval(lines[111][2:-3])[2]  # the diffusion resolution along the z-axis

        # movement/physical
        self.move_time_step = float(lines[117][2:-3])
        self.motility_force = float(lines[126][2:-3])   # the active force (in Newtons) of a cell actively moving
        self.max_radius = float(lines[129][2:-3])    # the maximum radius (in meters) of a cell

        # imaging
        self.image_quality = eval(lines[135][2:-3])    # the output image/video dimensions in pixels
        self.fps = float(lines[138][2:-3])   # the frames per second of the video produced
        self.background_color = eval(lines[141][2:-3])    # the background space color
        self.color_mode = eval(lines[144][2:-3])   # used to vary which method of coloring used
        self.pluri_color = eval(lines[147][2:-3])   # color of a pluripotent cell
        self.diff_color = eval(lines[150][2:-3])   # color of a differentiated cell
        self.pluri_gata6_high_color = eval(lines[153][2:-3])    # color of a pluripotent gata6 high cell
        self.pluri_nanog_high_color = eval(lines[156][2:-3])    # color of a pluripotent nanog high cell
        self.pluri_both_high_color = eval(lines[159][2:-3])    # color of a pluripotent gata6/nanog high cell

        # miscellaneous/experimental
        self.stochastic = eval(lines[168][2:-3])    # if initial fds variables are stochastic
        self.group = int(lines[171][2:-3])   # the number of cells introduced into or removed from the space at once
        self.guye_move = eval(lines[174][2:-3])    # whether or not to use the Guye method of cell motility
        self.diffuse_radius = float(lines[177][2:-3])   # the radius of search of diffusion points
        self.max_fgf4 = float(lines[180][2:-3])  # the maximum amount of fgf4 at a diffusion point
        self.eunbi_move = eval(lines[183][2:-3])    # use Eunbi's model for movement
        self.fgf4_move = eval(lines[186][2:-3])     # use FGF4 concentrations for NANOG high movements
        self.output_gradient = eval(lines[189][2:-3])   # output an image of the extracellular gradient

        # check that the name and path from the template are valid
        self.path = Input.check_name(self, template_location)

        # holds the current number of cells, step, and time when a step started (used for tracking efficiency)
        self.number_cells = 0
        self.current_step = self.beginning_step
        self.step_start = float()

        # these arrays hold all values of the cells, each index corresponds to a cell.
        self.cell_locations = np.empty((0, 3), dtype=float)    # holds every cell's location vector in the space
        self.cell_radii = np.empty((0, 1), dtype=float)     # holds every cell's radius
        self.cell_motion = np.empty((0, 1), dtype=bool)    # holds every cell's boolean for being in motion or not
        self.cell_fds = np.empty((0, 4), dtype=float)    # holds every cell's values for the fds, currently 4
        self.cell_states = np.empty((0, 1), dtype='<U14')    # holds every cell's state pluripotent or differentiated
        self.cell_diff_counter = np.empty((0, 1), dtype=int)    # holds every cell's differentiation counter
        self.cell_div_counter = np.empty((0, 1), dtype=int)    # holds every cell's division counter
        self.cell_death_counter = np.empty((0, 1), dtype=int)    # holds every cell's death counter
        self.cell_fds_counter = np.empty((0, 1), dtype=int)    # holds every cell's finite dynamical system counter
        self.cell_motility_force = np.empty((0, 3), dtype=float)    # holds every cell's motility force vector
        self.cell_jkr_force = np.empty((0, 3), dtype=float)    # holds every cell's JKR force vector
        self.cell_nearest_gata6 = np.empty((0, 1))  # holds index of nearest gata6 high neighbor
        self.cell_nearest_nanog = np.empty((0, 1))  # holds index of nearest nanog high neighbor
        self.cell_nearest_diff = np.empty((0, 1))  # holds index of nearest differentiated neighbor
        self.cell_highest_fgf4 = np.empty((0, 3))  # holds the location of highest fgf4 point

        self.cell_array_names = ["cell_locations", "cell_radii", "cell_motion", "cell_fds", "cell_states",
                                 "cell_diff_counter", "cell_div_counter", "cell_death_counter", "cell_fds_counter",
                                 "cell_motility_force", "cell_jkr_force", "cell_nearest_gata6", "cell_nearest_nanog",
                                 "cell_nearest_diff", "cell_highest_fgf4"]

        # holds the run time for key functions to track efficiency. each step these are outputted to the CSV file.
        self.update_diffusion_time = float()
        self.check_neighbors_time = float()
        self.nearest_time = float()
        self.cell_motility_time = float()
        self.cell_update_time = float()
        self.update_queue_time = float()
        self.handle_movement_time = float()
        self.jkr_neighbors_time = float()
        self.get_forces_time = float()
        self.apply_forces_time = float()

        # neighbor graph is used to locate cells that are in close proximity, while the JKR graph holds adhesion bonds
        # between cells that are either currently overlapping or still maintain an adhesive bond
        self.neighbor_graph, self.jkr_graph = igraph.Graph(), igraph.Graph()

        # squaring the approximation of the differential
        self.dx2, self.dy2, self.dz2 = self.dx ** 2, self.dy ** 2, self.dz ** 2

        # get the time step value for diffusion updates depending on whether 2D or 3D
        if self.size[2] == 0:
            self.dt = (self.dx2 * self.dy2) / (2 * self.diffuse * (self.dx2 + self.dy2))
        else:
            self.dt = (self.dx2 * self.dy2 * self.dz2) / (2 * self.diffuse * (self.dx2 + self.dy2 + self.dz2))

        # the points at which the diffusion values are calculated
        x_steps = (int(self.size[0] / self.dx) + 1)
        y_steps = (int(self.size[1] / self.dy) + 1)
        z_steps = (int(self.size[2] / self.dz) + 1)
        self.fgf4_values = np.zeros((x_steps, y_steps, z_steps))
        self.extracellular_names = ["fgf4_values"]

        # holds all indices of cells that will divide at a current step or be removed at that step
        self.cells_to_divide, self.cells_to_remove = np.empty((0, 1), dtype=int), np.empty((0, 1), dtype=int)

        # min and max radius lengths are used to calculate linear growth of the radius over time in 2D
        self.min_radius = self.max_radius / 2 ** 0.5
        self.pluri_growth = (self.max_radius - self.min_radius) / self.pluri_div_thresh
        self.diff_growth = (self.max_radius - self.min_radius) / self.diff_div_thresh

        # the csv and video objects that will be updated each step
        self.csv_object = object()
        self.video_object = object()

        # given all of the above parameters, run the corresponding mode
        Input.setup_simulation(self)

    def add_cell(self, location, radius, motion, fds, state, diff_counter, div_counter, death_counter, fds_counter,
                 motility_force, jkr_force, nearest_gata6, nearest_nanog, nearest_diff, highest_fgf4):
        """ Adds each of the new cell's values to
            the array holders, graphs, and total
            number of cells.
        """
        # adds the cell to the arrays holding the cell values, the 2D arrays have to be handled a bit differently as
        # axis=0 has to be provided and the appended array should also be of the same shape with additional brackets
        self.cell_locations = np.append(self.cell_locations, [location], axis=0)
        self.cell_radii = np.append(self.cell_radii, radius)
        self.cell_motion = np.append(self.cell_motion, motion)
        self.cell_fds = np.append(self.cell_fds, [fds], axis=0)
        self.cell_states = np.append(self.cell_states, state)
        self.cell_diff_counter = np.append(self.cell_diff_counter, diff_counter)
        self.cell_div_counter = np.append(self.cell_div_counter, div_counter)
        self.cell_death_counter = np.append(self.cell_death_counter, death_counter)
        self.cell_fds_counter = np.append(self.cell_fds_counter, fds_counter)
        self.cell_motility_force = np.append(self.cell_motility_force, [motility_force], axis=0)
        self.cell_jkr_force = np.append(self.cell_jkr_force, [jkr_force], axis=0)
        self.cell_nearest_gata6 = np.append(self.cell_nearest_gata6, nearest_gata6)
        self.cell_nearest_nanog = np.append(self.cell_nearest_nanog, nearest_nanog)
        self.cell_nearest_diff = np.append(self.cell_nearest_diff, nearest_diff)
        self.cell_highest_fgf4 = np.append(self.cell_highest_fgf4, [highest_fgf4], axis=0)

        # add it to the following graphs, this is done implicitly by increasing the length of the vertex list by
        # one, which the indices directly correspond to the cell holder arrays
        self.neighbor_graph.add_vertex()
        self.jkr_graph.add_vertex()

        # revalue the total number of cells
        self.number_cells += 1
