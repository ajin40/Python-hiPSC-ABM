import numpy as np
import random as r
import igraph
import math

import backend
import functions
import output
from run import template_param


def setup_cells(simulation):
    """ Indicate the cell arrays in the simulation and any initial
        conditions of these arrays. See documentation for more info.
    """
    # Add the specified number of NANOG/GATA6 high cells and create cell type GATA6_high for setting initial parameters
    # with cell_array().
    simulation.add_cells(simulation.num_nanog)
    simulation.add_cells(simulation.num_gata6, cell_type="GATA6_high")

    # Create the following cell arrays in the Simulation object. The instance variable simulation."array-name" will
    # point to this array. The arrays default to float64, 1-dim arrays (length of # cells). Use the parameters to
    # adjust the data type, 2-dim size, and initial condition for the entire array.
    simulation.cell_array("locations", override=np.random.rand(simulation.number_cells, 3) * simulation.size)
    simulation.cell_array("radii")
    simulation.cell_array("motion", dtype=bool, func=lambda: True)
    simulation.cell_array("FGFR", dtype=int, func=lambda: r.randrange(0, simulation.field))
    simulation.cell_array("ERK", dtype=int, func=lambda: r.randrange(0, simulation.field))
    simulation.cell_array("GATA6", dtype=int)
    simulation.cell_array("NANOG", dtype=int, func=lambda: r.randrange(1, simulation.field))
    simulation.cell_array("states", dtype=str, func=lambda: "Pluripotent")
    simulation.cell_array("death_counters", dtype=int, func=lambda: r.randrange(0, simulation.death_thresh))
    simulation.cell_array("diff_counters", dtype=int, func=lambda: r.randrange(0, simulation.pluri_to_diff))
    simulation.cell_array("div_counters", dtype=int, func=lambda: r.randrange(0, simulation.pluri_div_thresh))
    simulation.cell_array("fds_counters", dtype=int, func=lambda: r.randrange(0, simulation.fds_thresh))
    simulation.cell_array("motility_forces", vector=3)
    simulation.cell_array("jkr_forces", vector=3)
    # simulation.cell_array("nearest_nanog", dtype=int, func=lambda: -1)
    # simulation.cell_array("nearest_gata6", dtype=int, func=lambda: -1)
    # simulation.cell_array("nearest_diff", dtype=int, func=lambda: -1)

    # Update the number of cells marked with the "GATA6_high" cell type with alternative initial conditions.
    simulation.cell_array("GATA6", cell_type="GATA6_high", func=lambda: r.randrange(1, simulation.field))
    simulation.cell_array("NANOG", cell_type="GATA6_high", func=lambda: 0)


def run_steps(simulation):
    """ Specify the functions/methods called before/during/after
        a simulation's steps. See documentation for more info.
    """
    # Iterate over all steps specified in the Simulation object
    for simulation.current_step in range(simulation.beginning_step, simulation.end_step + 1):
        # Records model run time for the step and prints the current step/number of cells.
        backend.info(simulation)

        # Finds the neighbors of each cell that are within a fixed radius and store this info in a graph.
        functions.get_neighbors(simulation, distance=0.00001)    # double max cell radius

        # Updates cells by adjusting trackers for differentiation, division, growth, etc. based on intracellular,
        # intercellular, and extracellular conditions through a series of separate methods.
        functions.cell_death(simulation)
        functions.cell_diff_surround(simulation)
        functions.cell_division(simulation)
        functions.cell_growth(simulation)
        functions.cell_pathway(simulation)

        # Simulates molecular diffusion the specified extracellular gradient via the forward time centered space method.
        functions.update_diffusion(simulation, "fgf4_values")
        # functions.update_diffusion(simulation, "fgf4_alt")    # for testing morphogen release methods

        # Adds/removes cells to/from the simulation either all together or in desired groups of cells. If done in
        # groups, the apply_forces() function will be used to better represent asynchronous division and death.
        functions.update_queue(simulation)

        # Finds the nearest NANOG high, GATA6 high, and differentiated cells within a fixed radius. This provides
        # information that can be used for approximating cell motility.
        # functions.nearest(simulation, distance=0.000015)    # triple max cell radius

        # Calculates the direction/magnitude of a cell's movement depending on a variety of factors such as state
        # and presence of neighbors.
        functions.cell_motility(simulation)
        # functions.eunbi_motility(simulation)

        # Through the series of methods, attempt to move the cells to a state of physical equilibrium between adhesive
        # and repulsive forces acting on the cells, while applying active motility forces.
        functions.apply_forces(simulation)

        # Saves multiple forms of information about the simulation at the current step, including an image of the
        # space, CSVs with values of the cells, a temporary pickle of the Simulation object, and performance stats.
        # See the outputs.txt template file for turning off certain outputs.
        output.step_image(simulation)
        output.step_values(simulation)
        output.step_gradients(simulation)
        output.step_tda(simulation, in_pixels=True)
        output.temporary(simulation)
        output.simulation_data(simulation)

    # Ends the simulation by creating a video from all of the step images
    output.create_video(simulation, fps=6)


class Simulation(backend.Base):
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
        self.parallel = template_param(general_path, 5, bool)
        self.end_step = template_param(general_path, 8, int)
        self.num_nanog = template_param(general_path, 11, int)
        self.num_gata6 = template_param(general_path, 14, int)
        self.size = np.array(template_param(general_path, 17, tuple))
        self.order_66 = template_param(general_path, 20, str)

        # ------------- outputs template file ------------------------------
        outputs_path = paths.templates + "outputs.txt"    # path to outputs.txt template file
        self.output_values = template_param(outputs_path, 5, bool)
        self.output_tda = template_param(outputs_path, 9, bool)
        self.output_gradients = template_param(outputs_path, 12, bool)
        self.output_images = template_param(outputs_path, 15, bool)
        self.image_quality = template_param(outputs_path, 19, int)
        self.color_mode = template_param(outputs_path, 23, bool)

        # ------------- experimental template file -------------------------
        experimental_path = paths.templates + "experimental.txt"    # path to experimental.txt template file
        self.group = template_param(experimental_path, 5, int)
        self.dox_step = template_param(experimental_path, 9, int)
        self.guye_move = template_param(experimental_path, 13, bool)
        self.lonely_thresh = template_param(experimental_path, 17, int)

        # define any other instance variables that are not part of the template files

        # the temporal resolution for the simulation
        self.step_dt = 1800  # dt of each simulation step (1800 sec)
        self.move_dt = 181  # dt for incremental movement (180 sec)
        self.diffuse_dt = 0.24  # dt for stable diffusion model (0.24 sec)

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
        self.max_concentration = 30

        # calculate the size of the array for the diffusion points and create gradient array(s)
        self.gradient_size = np.ceil(self.size / self.spat_res).astype(int) + 1
        self.fgf4_values = np.zeros(self.gradient_size, dtype=float)
        self.gradient_names = ["fgf4_values"]  # add names for automatic CSV output of gradients

        # self.fgf4_alt = np.zeros(self.gradient_size, dtype=float)    # for testing morphogen release methods
        # self.gradient_names = ["fgf4_values", "fgf4_alt"]    # add names for automatic CSV output of gradients
