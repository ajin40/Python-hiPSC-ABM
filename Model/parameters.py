import numpy as np
import igraph
from backend import get_parameter


class Simulation:
    """ This object holds all of the important information about the simulation as it
        runs. Variables can be specified either directly or through the template files.
    """
    def __init__(self, paths, name, mode):
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
        self.parallel = get_parameter(general_path, 5, bool)
        self.end_step = get_parameter(general_path, 8, int)
        self.num_nanog = get_parameter(general_path, 11, int)
        self.num_gata6 = get_parameter(general_path, 14, int)
        self.size = np.array(get_parameter(general_path, 17, tuple))
        self.order_66 = get_parameter(general_path, 20, str)

        # ------------- outputs template file ------------------------------
        outputs_path = paths.templates + "outputs.txt"    # path to outputs.txt template file
        self.output_values = get_parameter(outputs_path, 5, bool)
        self.output_tda = get_parameter(outputs_path, 9, bool)
        self.output_gradients = get_parameter(outputs_path, 13, bool)
        self.output_images = get_parameter(outputs_path, 16, bool)
        self.image_quality = get_parameter(outputs_path, 20, int)
        self.fps = get_parameter(outputs_path, 23, float)
        self.color_mode = get_parameter(outputs_path, 27, bool)
        self.output_fgf4_image = get_parameter(outputs_path, 30, bool)

        # ------------- experimental template file -------------------------
        experimental_path = paths.templates + "experimental.txt"    # path to experimental.txt template file
        self.group = get_parameter(experimental_path, 5, int)
        self.dox_step = get_parameter(experimental_path, 9, int)
        self.guye_move = get_parameter(experimental_path, 13, bool)
        self.lonely_thresh = get_parameter(experimental_path, 17, int)

        # define any other instance variables that are not part of the template files

        # the temporal resolution for the simulation
        self.step_dt = 1800  # dt of each simulation step (1800 sec)
        self.move_dt = 200  # dt for incremental movement (200 sec)
        self.diffuse_dt = 0.24  # dt for stable diffusion model (0.5 sec)

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
        self.min_radius = self.max_radius / 2 ** (1/3)    # half the volume for max radius cell in 3D
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

        ########################################################################################################
        ########################################################################################################
        ########################################################################################################

        # these instance variables are rarely changed and are necessary for the model to run

        # hold the name/mode of the simulation and the Paths object
        self.name = name
        self.mode = mode
        self.paths = paths

        # hold the number of cells and the step to begin at (can be altered by various modes)
        self.number_cells = 0
        self.beginning_step = 1

        # arrays to store the cells set to divide or die
        self.cells_to_divide = np.array([], dtype=int)
        self.cells_to_remove = np.array([], dtype=int)

        # various other holders
        self.cell_array_names = list()    # stores the names of the cell arrays
        self.cell_types = dict()          # holds the names of the cell types defined in run.py
        self.method_times = dict()      # store the runtimes of the various methods as the model runs

    def cell_type(self, name, number):
        """ Creates a new cell type for setting initial parameters
            and defines a section in the cell arrays that corresponds
            to this cell type.
        """
        # determine the bounds of the section and update the number of cells
        begin = self.number_cells
        end = self.number_cells = begin + number

        # add that many cells to each of the graphs
        for graph in self.graph_names:
            self.__dict__[graph].add_vertices(number)

        # define the bounds of the section for the cell type name
        self.cell_types[name] = (begin, end)

    def cell_arrays(self, *args):
        """ Creates Simulation instance variables for each cell array
            which are used to hold cell values.
        """
        # go through all tuples passed
        for array_params in args:
            # add the array name to a list for automatic addition/removal when cells divide/die
            self.cell_array_names.append(array_params[0])

            # get the tuple length to designate if this is a 1D or 2D array
            length = len(array_params)

            # get the initial size of 1D array if the length is 2, but make a 2D array if a third index is provided
            if length == 2:
                size = self.number_cells
            elif length == 3:
                size = (self.number_cells, array_params[2])
            else:
                raise Exception("Tuples for defining cell array parameters should have length 2 or 3.")

            # if it's the python string data type, use object type instead
            if array_params[1] == str:
                array_type = object
            else:
                array_type = array_params[1]

            # create instance variable in Simulation object to represent cell array
            self.__dict__[array_params[0]] = np.empty(size, dtype=array_type)

    def initials(self, array_name, func, cell_type=None):
        """ Given a lambda function for the initial values
            of a cell array, update the arrays accordingly.
        """
        # if no cell type name provided
        if cell_type is None:
            # go through all cells and give this initial parameter
            for i in range(self.number_cells):
                self.__dict__[array_name][i] = func()

        # otherwise update the section of the array based on the cell type
        else:
            # get the beginning and end of the slice
            begin = self.cell_types[cell_type][0]
            end = self.cell_types[cell_type][1]

            # update only this slice of the cell array
            for i in range(begin, end):
                self.__dict__[array_name][i] = func()
