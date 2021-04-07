import numpy as np
import random as r
import igraph

from backend import template_param, commandline_param
from simulation import Simulation
from cellmethods import CellMethods
from celloutputs import CellOutputs


class CellSimulation(CellMethods, CellOutputs, Simulation):
    """ This class inherits a base Simulation class with additional methods from CellMethods
        and CellOutputs. More instance variables are specified below either directly or
        through the template files.
    """
    def __init__(self, paths, name):
        Simulation.__init__(self, paths, name)   # initialize the Simulation object instance variables
        """
        The following instance variables can be updated through template files located in the "templates"
        directory under the "Model" directory. The values must be specified in the .txt files as follows.

            (outputs.txt)
            1   How many frames per second of the output video that collects all step images? Ex. 6
            2   | 6 |
            3

        Note: extraneous spaces before or after the pipes will not affect the interpretation of the
        parameter. Use template_param(path-to-file, line number, data type) to read a specific line
        of a template file and interpret the value as the desired data type.

            self.fps = template_param(path, 2, float)
        """
        # ------------- general template file ------------------------------
        general_path = paths.templates + "general.yaml"    # path to general.yaml template file
        self.parallel = template_param(general_path, "parallel")
        self.end_step = template_param(general_path, "end_step")
        self.num_nanog = template_param(general_path, "num_nanog")
        self.num_gata6 = template_param(general_path, "num_gata6")
        self.size = np.array(template_param(general_path, "size"))
        self.order_66 = template_param(general_path, "order_66")
        # self.order_66 = commandline_param("-o", bool)

        # ------------- outputs template file ------------------------------
        outputs_path = paths.templates + "outputs.yaml"    # path to outputs.yaml template file
        self.output_values = template_param(outputs_path, "output_values")
        self.output_tda = template_param(outputs_path, "output_tda")
        self.output_gradients = template_param(outputs_path, "output_gradients")
        self.output_images = template_param(outputs_path, "output_images")
        self.image_quality = template_param(outputs_path, "image_quality")
        self.video_quality = template_param(outputs_path, "video_quality")
        self.fps = template_param(outputs_path, "fps")
        self.color_mode = template_param(outputs_path, "color_mode")

        # ------------- experimental template file -------------------------
        experimental_path = paths.templates + "experimental.yaml"    # path to experimental.yaml template file
        self.group = template_param(experimental_path, "group")
        self.dox_step = template_param(experimental_path, "dox_step")
        self.guye_move = template_param(experimental_path, "guye_move")
        self.lonely_thresh = template_param(experimental_path, "lonely_thresh")

        # define any other instance variables that are not part of the template files

        # the temporal resolution for the simulation
        self.step_dt = 1800  # dt of each simulation step (1800 sec)
        self.move_dt = 180  # dt for incremental movement (180 sec)
        # self.diffuse_dt = 0.24  # dt for stable diffusion model (0.24 sec)
        # self.diffuse_dt = 6.24  # dt for stable diffusion model (6 sec)

        # the field for the finite dynamical system
        self.field = 2

        # probability of randomly increasing FDS value to high
        self.GATA6_prob = 0.01
        self.NANOG_prob = 0.01

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

        # create graph with instance variable jkr_graph for holding adhesions between contacting cells
        self.jkr_graph = igraph.Graph()
        self.graph_names.append("jkr_graph")

        # the spatial resolution of the space, the diffusion constant for the molecule gradients, the radius of
        # search for diffusion points, and the max concentration at a diffusion point (not currently in use)
        # self.spat_res = 0.00000707106
        # self.diffuse_const = 0.00000000005    # 50 um^2/s
        # self.spat_res = 0.00001
        # self.spat_res2 = self.spat_res ** 2
        # self.diffuse_const = 0.000000000002  # 2 um^2/s
        # self.max_concentration = 2

        # calculate the size of the array for the diffusion points and create gradient array(s) (not currently in use)
        # self.gradient_size = np.ceil(self.size / self.spat_res).astype(int) + 1
        # self.fgf4_values = np.zeros(self.gradient_size, dtype=float)
        # self.gradient_names = ["fgf4_values"]  # add names for automatic CSV output of gradients
        # self.degradation = 0.1    # this will degrade the morphogen by this much at each step

        # self.fgf4_alt = np.zeros(self.gradient_size, dtype=float)    # for testing morphogen release methods
        # self.gradient_names = ["fgf4_values", "fgf4_alt"]    # add names for automatic CSV output of gradients

    def steps(self):
        """ Specify any Simulation instance methods called before/during/after
            the simulation, see example below.

            Example:
                self.before_steps()

                for self.current_step in range(self.beginning_step, self.end_step + 1):
                    self.during_steps()

                self.after_steps()
        """
        # Iterate over all steps specified in the Simulation object
        for self.current_step in range(self.beginning_step, self.end_step + 1):
            # Records model run time for the step and prints the current step/number of cells.
            self.info()

            # Finds the neighbors of each cell that are within a fixed radius and store this info in a graph.
            self.get_neighbors(distance=0.00001)    # double max cell radius

            # Updates cells by adjusting trackers for differentiation, division, growth, etc. based on intracellular,
            # intercellular, and extracellular conditions through a series of separate methods.
            # functions.cell_death(self)
            self.cell_diff_surround()
            self.cell_division()
            self.cell_growth()
            # self.cell_stochastic_update()
            self.cell_pathway()
            self.cell_differentiate()

            # Simulates diffusion the specified extracellular gradient via the forward time centered space method.
            # self.update_diffusion("fgf4_values")
            # self.update_diffusion("fgf4_alt")    # for testing morphogen release methods

            # Adds/removes cells to/from the simulation either all together or in desired groups of cells. If done in
            # groups, the apply_forces() function will be used to better represent asynchronous division and death.
            self.update_queue()

            # Finds the nearest NANOG high, GATA6 high, and differentiated cells within a fixed radius. This provides
            # information that can be used for approximating cell motility.
            # self.nearest(distance=0.000015)    # triple max cell radius

            # Calculates the direction/magnitude of a cell's movement depending on a variety of factors such as state
            # and presence of neighbors.
            self.cell_motility()
            # self.eunbi_motility()

            # Through the series of methods, attempt to move the cells to a state of physical equilibrium between
            # adhesive and repulsive forces acting on the cells, while applying active motility forces.
            self.apply_forces()

            # Saves multiple forms of information about the simulation at the current step, including an image of the
            # space, CSVs with values of the cells, a temporary pickle of the Simulation object, and performance stats.
            # See the outputs.txt template file for turning off certain outputs.
            self.step_image()
            self.step_values(arrays=["locations", "FGF4", "FGFR", "ERK", "GATA6", "NANOG", "states", "diff_counters",
                                     "div_counters"])
            # self.step_gradients()
            self.step_tda(in_pixels=True)
            self.temp()
            self.data()

        # Ends the simulation by creating a video from all of the step images
        self.create_video()

    def agent_initials(self):
        """ Add cells into the simulation and specify any values the cells should have.
            The cell arrays will default to float64, 1-dim arrays of zeros. Use the
            parameters to adjust the data type, 2-dim size, and initial conditions. The
            "cell_type" keyword is used to apply initial conditions to the group of cells
            marked with the same cell type in add_cells().
        """
        # Add the specified number of NANOG/GATA6 high cells and create cell type GATA6_high.
        self.add_agents(self.num_nanog)
        self.add_agents(self.num_gata6, agent_type="GATA6_high")

        # Create the following cell arrays with initial conditions.
        self.agent_array("locations", override=np.random.rand(self.number_agents, 3) * self.size)
        self.agent_array("radii")
        self.agent_array("motion", dtype=bool, func=lambda: True)
        self.agent_array("FGF4", dtype=int, func=lambda: r.randrange(0, self.field))
        self.agent_array("FGFR", dtype=int, func=lambda: r.randrange(0, self.field))
        self.agent_array("ERK", dtype=int, func=lambda: r.randrange(0, self.field))
        self.agent_array("GATA6", dtype=int)
        self.agent_array("NANOG", dtype=int, func=lambda: r.randrange(0, self.field))
        self.agent_array("states", dtype=int)
        self.agent_array("death_counters", dtype=int, func=lambda: r.randrange(0, self.death_thresh))
        self.agent_array("diff_counters", dtype=int, func=lambda: r.randrange(0, self.pluri_to_diff))
        self.agent_array("div_counters", dtype=int, func=lambda: r.randrange(0, self.pluri_div_thresh))
        self.agent_array("fds_counters", dtype=int, func=lambda: r.randrange(0, self.fds_thresh))
        self.agent_array("motility_forces", vector=3)
        self.agent_array("jkr_forces", vector=3)
        # self.agent_array("nearest_nanog", dtype=int, func=lambda: -1)
        # self.agent_array("nearest_gata6", dtype=int, func=lambda: -1)
        # self.agent_array( "nearest_diff", dtype=int, func=lambda: -1)

        # Update the number of cells marked with the "GATA6_high" cell type with alternative initial conditions.
        self.agent_array("GATA6", agent_type="GATA6_high", func=lambda: r.randrange(1, self.field))
        self.agent_array("NANOG", agent_type="GATA6_high", func=lambda: 0)
