import numpy as np
import igraph
import math
import input
from backend import Base


class Simulation(Base):
    """ This object holds all of the important information about the simulation as it
        runs. Variables can be specified either directly or through the template files.
    """
    def __init__(self, paths, name):
        super().__init__(paths, name)    # initialize the Base object
        """
        The following instance variables can be updated through template files located in the "templates"
        directory under the "Model" directory. The values must be specified in the .txt files as follows.

            (outputs.txt)
            1   How many frames per second of the output video that collects all step images? Ex. 6
            2   | 6 |
            3

        Note: extraneous spaces before or after the pipes will not affect the interpretation of the
        parameter. Use get_parameter(path-to-file, line number, data type) to read a specific line
        of a template file and interpret the value as the desired data type.

            self.fps = get_parameter(path, 2, float)
        """
        # ------------- general template file ------------------------------
        general_path = paths.templates + "general.txt"    # path to general.txt template file
        self.parallel = input.get_parameter(general_path, 5, bool)
        self.end_step = input.get_parameter(general_path, 8, int)
        self.num_nanog = input.get_parameter(general_path, 11, int)
        self.num_gata6 = input.get_parameter(general_path, 14, int)
        self.size = np.array(input.get_parameter(general_path, 17, tuple))
        self.order_66 = input.get_parameter(general_path, 20, str)

        # ------------- outputs template file ------------------------------
        outputs_path = paths.templates + "outputs.txt"    # path to outputs.txt template file
        self.output_values = input.get_parameter(outputs_path, 5, bool)
        self.output_tda = input.get_parameter(outputs_path, 9, bool)
        self.output_gradients = input.get_parameter(outputs_path, 12, bool)
        self.output_images = input.get_parameter(outputs_path, 15, bool)
        self.image_quality = input.get_parameter(outputs_path, 19, int)
        self.in_pixels = input.get_parameter(outputs_path, 23, bool)
        self.origin_bottom = input.get_parameter(outputs_path, 26, bool)
        self.back_color = input.get_parameter(outputs_path, 29, tuple)
        self.fps = input.get_parameter(outputs_path, 32, float)
        self.color_mode = input.get_parameter(outputs_path, 36, bool)
        self.output_fgf4_image = input.get_parameter(outputs_path, 39, bool)

        # ------------- experimental template file -------------------------
        experimental_path = paths.templates + "experimental.txt"    # path to experimental.txt template file
        self.group = input.get_parameter(experimental_path, 5, int)
        self.dox_step = input.get_parameter(experimental_path, 9, int)
        self.guye_move = input.get_parameter(experimental_path, 13, bool)
        self.lonely_thresh = input.get_parameter(experimental_path, 17, int)

        # define any other instance variables that are not part of the template files

        # the temporal resolution for the simulation
        self.step_dt = 1800  # dt of each simulation step (1800 sec)
        self.move_dt = 200  # dt for incremental movement (200 sec)
        self.diffuse_dt = 0.24  # dt for stable diffusion model (0.5 sec)
        self.move_steps = math.ceil(self.step_dt / self.move_dt)

        # the field for the finite dynamical system
        self.field = 3

        # the rates (in steps) of division, differentiation, death, and finite dynamical system updating
        self.pluri_div_thresh = 36
        self.diff_div_thresh = 72
        self.pluri_to_diff = 72
        self.death_thresh = 144
        self.fds_thresh = 1

        # min and max radius lengths are used to calculate linear growth of the radius over time
        self.max_radius = 0.000005    # 5 um
        self.min_radius = self.max_radius / 2 ** 0.5    # half the area for max radius cell in 2D
        self.pluri_growth = (self.max_radius - self.min_radius) / self.pluri_div_thresh
        self.diff_growth = (self.max_radius - self.min_radius) / self.diff_div_thresh

        # the neighbor graph holds all nearby cells within a fixed radius, and the JKR graph is used for
        # storing adhesive bonds between cells
        self.neighbor_graph = igraph.Graph()
        self.jkr_graph = igraph.Graph()

        # add the names of the graphs below for automatic cell addition and removal
        self.graph_names = ["neighbor_graph", "jkr_graph"]

        # the spatial resolution of the space, the diffusion constant for the molecule gradients, the radius of
        # search for diffusion points, and the max concentration at a diffusion point
        self.spat_res = 0.00000707106
        self.spat_res2 = self.spat_res ** 2
        self.diffuse_const = 0.00000000005    # 50 um^2/s
        self.max_concentration = 200    # very arbitrary

        # calculate the size of the array for the diffusion points and create gradient array
        self.gradient_size = np.ceil(self.size / self.spat_res).astype(int) + 1
        self.fgf4_values = np.zeros(self.gradient_size, dtype=float)
        # self.fgf4_alt = np.zeros(self.gradient_size, dtype=float)

        # add the names of the gradients below for automatic diffusion updating
        # self.gradient_names = ["fgf4_values", "fgf4_alt"]
        self.gradient_names = ["fgf4_values"]
