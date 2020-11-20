import numpy as np
import igraph
from backend import Base


# used to hold all values necessary to the simulation based on the various modes
class Simulation(Base):
    def __init__(self, templates_path, name, path, separator, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
