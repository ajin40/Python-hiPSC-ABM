import numpy as np
import random as r
import input
import output
import backend
import functions


# setup() will direct how the model is run based on inputted parameters. If a new simulation is desired, setup()
# will return an instance of the Simulation object which holds all important information of the simulation as it runs.
simulation = input.setup()

# For continuing a previous simulation (mode 1), bypass the initialization of cell array values
if simulation.mode == 0:
    # Define the number of cells for each cell type. These names can be used to initialize the model with specific
    # numbers of cell types that may differ in initial parameters.
    simulation.cell_type("NANOG_high", simulation.num_nanog)
    simulation.cell_type("GATA6_high", simulation.num_gata6)

    # Define the cell arrays and their initial parameters with lambda functions. This will create instance variables
    # in the Simulation object with the name specified. The vector keyword can be used to make 2-dimensional arrays.
    simulation.cell_array("locations", float, lambda: np.random.rand(3) * simulation.size, vector=3)
    simulation.cell_array("radii", float, lambda: simulation.min_radius)
    simulation.cell_array("motion", bool, lambda: True)
    simulation.cell_array("FGFR", int, lambda: r.randrange(0, simulation.field))
    simulation.cell_array("ERK", int, lambda: r.randrange(0, simulation.field))
    simulation.cell_array("GATA6", int, lambda: 0)
    simulation.cell_array("NANOG", int, lambda: r.randrange(1, simulation.field))
    simulation.cell_array("states", str, lambda: "Pluripotent")
    simulation.cell_array("death_counters", int, lambda: r.randrange(0, simulation.death_thresh))
    simulation.cell_array("diff_counters", int, lambda: r.randrange(0, simulation.pluri_to_diff))
    simulation.cell_array("div_counters", int, lambda: r.randrange(0, simulation.pluri_div_thresh))
    simulation.cell_array("fds_counters", int, lambda: r.randrange(0, simulation.fds_thresh))
    simulation.cell_array("motility_forces", float, lambda: np.zeros(3), vector=3)
    simulation.cell_array("jkr_forces", float, lambda: np.zeros(3), vector=3)
    simulation.cell_array("nearest_nanog", int, lambda: -1)
    simulation.cell_array("nearest_gata6", int, lambda: -1)
    simulation.cell_array("nearest_diff", int, lambda: -1)

    # Modify the initial parameters for the number of cells specified with GATA6_high cell type
    simulation.alt_initials("GATA6", "GATA6_high", lambda: r.randrange(1, simulation.field))
    simulation.alt_initials("NANOG", "GATA6_high", lambda: 0)

# Add any functions under the loop that will be called during each step of the simulation.
for simulation.current_step in range(simulation.beginning_step, simulation.end_step + 1):
    # Records model run time for the step and prints the current step/number of cells,
    backend.info(simulation)

    # Finds the neighbors of each cell that are within a fixed radius and store this info in a graph.
    functions.get_neighbors(simulation)

    # Updates cells by adjusting trackers for differentiation, division, growth, etc. based on intracellular,
    # intercellular, and extracellular conditions through a series of separate methods.
    functions.cell_death(simulation)
    functions.cell_diff_surround(simulation)
    functions.cell_division(simulation)
    functions.cell_growth(simulation)
    functions.cell_pathway(simulation)

    # Simulates the diffusion for each of the extracellular gradients via the forward time centered space method.
    functions.update_diffusion(simulation)

    # Adds/removes cells to/from the simulation either all together or in desired groups of cells. If done in
    # groups, the handle_movement() function will be used to better represent asynchronous division and death.
    functions.update_queue(simulation)

    # Finds the nearest NANOG high, GATA6 high, and differentiated cells within a fixed radius. This provides
    # information that can be used for approximating cell motility.
    functions.nearest(simulation)

    # Calculates the direction/magnitude of a cell's movement depending on a variety of factors such as state
    # and presence of neighbors.
    functions.cell_motility(simulation)
    # functions.eunbi_motility(simulation)

    # Attempts to move the cells to a state of physical equilibrium between adhesive and repulsive forces acting on
    # the cells, while applying active motility forces from the previous cell_motility() function.
    functions.handle_movement(simulation)

    # Saves multiple forms of information about the simulation at the current step, including an image of the space,
    # CSVs with values of the cells, a temporary pickle of the Simulation object, and performance stats.
    output.step_outputs(simulation)

# Ends the simulation by creating a video from all of the step images
output.create_video(simulation)
