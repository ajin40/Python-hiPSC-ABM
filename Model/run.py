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
simulation.cell_type("NANOG_high", 1000)
simulation.cell_type("GATA6_high", 0)

# Define the cell arrays used to store values of the cell. Each tuple corresponds to a cell array that will be
# generated. The first index of the tuple is the instance variable name for the Simulation class, the second being the
# data type, and the last (if present) can be used to create a 2D array
simulation.cell_arrays(("locations", float, 3), ("radii", float), ("motion", bool), ("FGFR", int), ("ERK", int),
                       ("GATA6", int), ("NANOG", int), ("state", "<U14"), ("diff_counter", int), ("div_counter", int),
                       ("death_counter", int), ("fds_counter", int), ("motility_force", float, 3),
                       ("jkr_force", float, 3), ("rotation", float))

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
simulation.initials("state", lambda: "Pluripotent")
simulation.initials("death_counter", lambda: r.randrange(0, simulation.death_thresh))
simulation.initials("diff_counter", lambda: r.randrange(0, simulation.pluri_to_diff))
simulation.initials("div_counter", lambda: r.randrange(0, simulation.pluri_div_thresh))
simulation.initials("fds_counter", lambda: r.randrange(0, simulation.fds_thresh))
simulation.initials("motility_force", lambda: np.zeros(3, dtype=float))
simulation.initials("jkr_force", lambda: np.zeros(3, dtype=float))
simulation.initials("rotation", lambda: r.random() * 360)

# These are the initial parameters for the "GATA6_high" cells, the cell_type argument is used to flag this
simulation.initials("GATA6", lambda: r.randrange(0, simulation.field), cell_type="GATA6_high")
simulation.initials("NANOG", lambda: 0, cell_type="GATA6_high")

# places all of the diffusion points into bins so that the model can use a bin sorting method to when determining
# highest/lowest concentrations of the extracellular gradient(s)
# functions.setup_diffusion_bins(simulation)

# this will loop over all steps defined in the general template file in addition to updating the current step
# of the simulation. this is done to explicitly/easily show what happens at each step.
for simulation.current_step in range(simulation.beginning_step, simulation.end_step + 1):
    # prints the current step, number of cells, and records run time. used to give an idea of the simulation progress.
    functions.info(simulation)

    # updates the graph used to hold neighbors of cells within a fixed radius.
    functions.check_neighbors(simulation)

    # updates cells by adjusting trackers for differentiation and division based on intracellular, intercellular,
    # and extracellular conditions. this is done through a series of methods
    functions.cell_death(simulation)
    functions.cell_diff_surround(simulation)
    functions.cell_growth(simulation)
    functions.cell_division(simulation)
    functions.cell_pathway(simulation)

    # updates each of the extracellular gradients via the finite difference method after cells have interacted
    # with the gradient in cell_update().
    # functions.update_diffusion(simulation)

    # adds/removes cells to/from the simulation either all together or in desired groups of cells. if done in
    # groups, the handle_movement() function will be used to better represent asynchronous division and death.
    functions.update_queue(simulation)

    # find the nearest NANOG high, GATA6 high, and differentiated cell within a fixed radius. this provides information
    # that can potentially be used for movement such as moving toward the nearest NANOG high cell
    # functions.nearest(simulation)

    # find the nearest pluripotent cell within a fixed radius that is not part of the same component of the underlying
    # graph of all pluripotent cells. used to represent the movement of pluripotent clusters
    # functions.nearest_cluster(simulation)

    # locate the diffusion point (within a fixed radius) that has the highest FGF4 concentration. can be used to
    # represent chemotactic movement of cells
    # functions.highest_fgf4(simulation)

    # calculates the direction/magnitude of the movement of the cell depending on a variety of factors such as state,
    # extracellular gradient, and presence of neighbors
    functions.cell_motility(simulation)

    # moves the cells to a state of physical equilibrium between adhesive and repulsive forces acting on the cells,
    # while also applying active movement forces from the previous cell_motility() function.
    functions.handle_movement(simulation)

    # saves multiple forms of information about the simulation at the current step, including an image of the space,
    # csvs with values of the cells, a temporary pickle of the Simulation instance, and performance stats.
    output.step_outputs(simulation)

# ends the simulation by creating a video from all of the images created by the simulation
output.create_video(simulation)
