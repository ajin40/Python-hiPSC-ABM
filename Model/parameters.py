import random as r
import numpy as np
import igraph


# used to hold all values necessary to the simulation based on the various modes
class Simulation:
    def __init__(self, templates_path, name, path, mode, separator):
        self.name = name    # name of the simulation, used to name files
        self.path = path    # path to the main simulation directory

        # these directories fall under the main simulation directory
        self.images_path = path + name + "_images" + separator    # path to image directory
        self.values_path = path + name + "_values" + separator    # path to step csv directory
        self.gradients_path = path + name + "_gradients" + separator    # path to gradient directory
        self.tda_path = path + name + "_tda" + separator    # path to TDA directory

        # ------------- general template file -------------------------
        # open the .txt file and get a list of the lines
        with open(templates_path + "general.txt") as general_file:
            general = general_file.readlines()

        # create instance variables based on template parameters
        self.parallel = eval(general[4][2:-3])
        self.end_step = int(general[7][2:-3])
        self.number_cells = int(general[10][2:-3])
        self.size = np.array(eval(general[13][2:-3]))

        # ------------- imaging template file -------------------------
        # open the .txt file and get a list of the lines
        with open(templates_path + "imaging.txt") as imaging_file:
            imaging = imaging_file.readlines()

        # create instance variables based on template parameters
        self.output_images = eval(imaging[4][2:-3])
        self.image_quality = int(imaging[8][2:-3])
        self.fps = float(imaging[11][2:-3])
        self.color_mode = eval(imaging[15][2:-3])
        self.output_gradient = eval(imaging[18][2:-3])

        # ------------- experimental template file -------------------------
        # open the .txt file and get a list of the lines
        with open(templates_path + "experimental.txt") as experimental_file:
            experimental = experimental_file.readlines()

        # create instance variables based on template parameters
        self.pluri_div_thresh = int(experimental[4][2:-3])
        self.diff_div_thresh = int(experimental[7][2:-3])
        self.pluri_to_diff = int(experimental[10][2:-3])
        self.death_thresh = int(experimental[13][2:-3])
        self.fds_thresh = int(experimental[16][2:-3])
        self.fgf4_thresh = int(experimental[19][2:-3])
        self.lonely_cell = int(experimental[22][2:-3])
        self.group = int(experimental[25][2:-3])
        self.guye_move = eval(experimental[28][2:-3])
        self.eunbi_move = eval(experimental[31][2:-3])
        self.max_concentration = float(experimental[34][2:-3])
        self.fgf4_move = eval(experimental[37][2:-3])
        self.output_tda = eval(experimental[40][2:-3])
        self.dox_value = float(experimental[43][2:-3])
        self.field = int(experimental[46][2:-3])

        # the following instance variables are only necessary for a new simulation. these include arrays for storing
        # the cell values, graphs for cell neighbors, and other values that aren't included in the template files
        if mode == 0:
            # start the simulation at step 1
            self.beginning_step = 1

            # the spatial resolution of the space
            self.spat_res = 0.00001  # 10 um
            self.spat_res2 = self.spat_res ** 2

            # the temporal resolution for the simulation
            self.step_dt = 1800  # dt of each simulation step (1800 sec)
            self.move_dt = 200  # dt for incremental movement (200 sec)
            self.diffuse_dt = 0.5  # dt for stable diffusion model (0.5 sec)

            # min and max radius lengths are used to calculate linear growth of the radius over time in 2D
            self.max_radius = 0.000005  # 5 um
            self.min_radius = self.max_radius / 2 ** 0.5
            self.pluri_growth = (self.max_radius - self.min_radius) / self.pluri_div_thresh
            self.diff_growth = (self.max_radius - self.min_radius) / self.diff_div_thresh

            # place the names of the cell arrays in this list. the model will use to delete and add cells
            # automatically (order doesn't matter)
            self.cell_array_names = ["cell_locations", "cell_radii", "cell_motion", "cell_fds", "cell_states",
                                     "cell_diff_counter", "cell_div_counter", "cell_death_counter", "cell_fds_counter",
                                     "cell_motility_force", "cell_jkr_force", "cell_nearest_gata6",
                                     "cell_nearest_nanog", "cell_nearest_diff", "cell_highest_fgf4",
                                     "cell_nearest_cluster", "cell_dox_value", "cell_rotation"]

            # these are the cell arrays used to hold all values of the cells. each index corresponds to a cell so
            # parallelization is made easier. this is an alternative to individual cell objects
            self.cell_locations = np.empty((self.number_cells, 3), dtype=float)
            self.cell_radii = np.empty(self.number_cells, dtype=float)
            self.cell_motion = np.empty(self.number_cells, dtype=bool)
            self.cell_fds = np.empty((self.number_cells, 4), dtype=float)
            self.cell_states = np.empty(self.number_cells, dtype='<U14')
            self.cell_diff_counter = np.empty(self.number_cells, dtype=int)
            self.cell_div_counter = np.empty(self.number_cells, dtype=int)
            self.cell_death_counter = np.empty(self.number_cells, dtype=int)
            self.cell_fds_counter = np.empty(self.number_cells, dtype=int)
            self.cell_motility_force = np.empty((self.number_cells, 3), dtype=float)
            self.cell_jkr_force = np.empty((self.number_cells, 3), dtype=float)
            self.cell_nearest_gata6 = np.empty(self.number_cells)
            self.cell_nearest_nanog = np.empty(self.number_cells)
            self.cell_nearest_diff = np.empty(self.number_cells)
            self.cell_highest_fgf4 = np.empty((self.number_cells, 3))
            self.cell_nearest_cluster = np.empty(self.number_cells)
            self.cell_dox_value = np.empty(self.number_cells, dtype=float)
            self.cell_rotation = np.empty(self.number_cells, dtype=float)

            # iterate through all cell arrays setting initial values
            for i in range(self.number_cells):
                n = self.field    # get the fds field
                div_counter = r.randrange(0, self.pluri_div_thresh)    # get division counter for division/cell size

                # apply initial value for each cell
                self.cell_locations[i] = np.random.rand(3) * self.size
                self.cell_radii[i] = self.min_radius + self.pluri_growth * div_counter
                self.cell_motion[i] = True
                self.cell_fds[i] = np.array([r.randrange(0, n), r.randrange(0, n), 0,  r.randrange(1, n)])
                self.cell_states[i] = "Pluripotent"
                self.cell_diff_counter[i] = r.randrange(0, self.pluri_to_diff)
                self.cell_div_counter[i] = div_counter
                self.cell_death_counter[i] = r.randrange(0, self.death_thresh)
                self.cell_fds_counter[i] = r.randrange(0, self.fds_thresh)
                self.cell_motility_force[i] = np.zeros(3, dtype=float)
                self.cell_jkr_force[i] = np.zeros(3, dtype=float)
                self.cell_nearest_gata6[i] = np.nan
                self.cell_nearest_nanog[i] = np.nan
                self.cell_nearest_diff[i] = np.nan
                self.cell_highest_fgf4[i] = np.array([np.nan, np.nan, np.nan])
                self.cell_nearest_cluster[i] = np.nan
                self.cell_dox_value[i] = r.random()
                self.cell_rotation[i] = r.random() * 360

            # holds all indices of cells that will divide or be removed during each step
            self.cells_to_divide = np.array([], dtype=int)
            self.cells_to_remove = np.array([], dtype=int)

            # much like the cell arrays add any graphs to this list for automatic addition/removal
            self.graph_names = ["neighbor_graph", "jkr_graph"]

            # create graphs used to all cells and their neighbors, initialize them with the number of cells
            self.neighbor_graph = igraph.Graph(self.number_cells)
            self.jkr_graph = igraph.Graph(self.number_cells)

            # the diffusion constant for the molecule gradients and the radius of search for diffusion points
            self.diffuse = 0.00000000005    # 50 um^2/s
            self.diffuse_radius = self.spat_res * 0.707106781187

            # much like the cell arrays add any gradient names to list this so that a diffusion function can
            # act on them automatically
            self.extracellular_names = ["fgf4_values", "fgf4_alt"]

            # calculate the size of the array holding the diffusion points and create gradient objects
            self.gradient_size = np.ceil(self.size / self.spat_res).astype(int) + np.ones(3, dtype=int)
            self.fgf4_values = np.zeros(self.gradient_size)
            self.fgf4_alt = np.zeros(self.gradient_size)

            # used to hold the run times of functions
            self.function_times = dict()
