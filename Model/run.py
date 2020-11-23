import numpy as np
import random as r
import input
import output
import functions

# setup() will direct how the model is to be run based on the selected mode. If a new simulation is desired, setup()
# will return an instance of the Simulation which holds all important information of the simulation as it runs.
simulation = input.setup()

# Define the names of any cell types. These names will be used to begin the model with a set number of cells that
# correspond to the particular initial parameters for that cell type.
simulation.cell_type("NANOG_high", simulation.num_nanog)
simulation.cell_type("GATA6_high", simulation.num_gata6)

# Define the cell arrays used to store values of the cell. Each tuple corresponds to a cell array that will be
# generated. The first index of the tuple is the instance variable name for the Simulation class, the second being the
# data type, and the last (if present) can be used to create a 2D array
simulation.cell_arrays(("locations", float, 3), ("radii", float), ("motion", bool), ("FGFR", int), ("ERK", int),
                       ("GATA6", int), ("NANOG", int), ("states", "<U14"), ("diff_counters", int), ("div_counters", int),
                       ("death_counters", int), ("fds_counters", int), ("motility_forces", float, 3),
                       ("jkr_forces", float, 3), ("rotations", float))

# Define the initial parameters for the cells using lambda expressions. The following lines have no "cell_type"
# argument, which is used to designate that these are initial parameters for all cells; however, these can be overridden
# when defining specific cell types. This is really meant to reduce overall writing for cell types that only differ
# slightly from the base parameters.
simulation.initials("locations", lambda: np.random.rand(3) * simulation.size)
simulation.initials("radii", lambda: simulation.min_radius)
simulation.initials("motion", lambda: True)
simulation.initials("FGFR", lambda: r.randrange(0, simulation.field))
simulation.initials("ERK", lambda: r.randrange(0, simulation.field))
simulation.initials("GATA6", lambda: 0)
simulation.initials("NANOG", lambda: r.randrange(1, simulation.field))
simulation.initials("states", lambda: "Pluripotent")
simulation.initials("death_counters", lambda: r.randrange(0, simulation.death_thresh))
simulation.initials("diff_counters", lambda: r.randrange(0, simulation.pluri_to_diff))
simulation.initials("div_counters", lambda: r.randrange(0, simulation.pluri_div_thresh))
simulation.initials("fds_counters", lambda: r.randrange(0, simulation.fds_thresh))
simulation.initials("motility_forces", lambda: np.zeros(3, dtype=float))
simulation.initials("jkr_forces", lambda: np.zeros(3, dtype=float))
simulation.initials("rotations", lambda: r.random() * 360)

# These are the initial parameters for the "GATA6_high" cells, the cell_type argument is used to flag this
simulation.initials("GATA6", lambda: r.randrange(0, simulation.field), cell_type="GATA6_high")
simulation.initials("NANOG", lambda: 0, cell_type="GATA6_high")

# Places all of the diffusion points into bins so that the model can use a bin sorting method when determining
# cell motility based on diffusion point locations. (not in use)
# functions.setup_diffusion_bins(simulation)

# Add any functions under the loop that will be called during each step of the simulation.
for simulation.current_step in range(simulation.beginning_step, simulation.end_step + 1):
    # Prints the current step, number of cells, and records model run time.
    functions.info(simulation)

    # Find the neighbors of each cell that are within a fixed radius and stores this info in a graph.
    functions.check_neighbors(simulation)

    # Updates cells by adjusting trackers for differentiation and division based on intracellular, intercellular,
    # and extracellular conditions through a series of methods.
    functions.cell_death(simulation)
    functions.cell_diff_surround(simulation)
    functions.cell_growth(simulation)
    functions.cell_division(simulation)
    functions.cell_pathway(simulation)

    # Simulates the diffusion for each of the extracellular gradients via the forward time centered space method.
    functions.update_diffusion(simulation)

    # Adds/removes cells to/from the simulation either all together or in desired groups of cells. If done in
    # groups, the handle_movement() function will be used to better represent asynchronous division and death.
    functions.update_queue(simulation)

    # Find the nearest NANOG high, GATA6 high, and differentiated cells within a fixed radius. this provides
    # information that can potentially be used for approximating cell motility. (not in use)
    # functions.nearest(simulation)

    # Find the nearest pluripotent cell within a fixed radius that is not part of the same component of the underlying
    # graph of all pluripotent cells. Used to represent the movement of pluripotent clusters. (not in use)
    # functions.nearest_cluster(simulation)

    # Locate the diffusion point (within a fixed radius) that has the highest FGF4 concentration. Can be used to
    # approximate the chemotactic movement of cells. (not in use)
    # functions.highest_fgf4(simulation)

    # Calculates the direction/magnitude of the movement of the cell depending on a variety of factors such as state,
    # extracellular gradient, and presence of neighbors.
    functions.cell_motility(simulation)

    # Attempts to move the cells to a state of physical equilibrium between adhesive and repulsive forces acting on
    # the cells, while also applying active motility forces from the previous cell_motility() function.
    functions.handle_movement(simulation)

    # Saves multiple forms of information about the simulation at the current step, including an image of the space,
    # CSVs with values of the cells, a temporary pickle of the Simulation instance, and performance stats.
    output.step_outputs(simulation)

# Ends the simulation by creating a video from all of the step images
output.create_video(simulation)
