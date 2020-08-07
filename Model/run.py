import input
import output
import functions

# setup() will create an instance of the Simulation class that holds all relevant information of the model.
# this is done by reading a template .txt file that contains all initial parameters of the model.
simulation = input.setup()

# Will locate the diffusion points for the extracellular gradient to bins, used for chemotactic movement
# called once as the locations don't change. not currently in use.
functions.setup_diffusion_bins(simulation)

# This will loop over all steps defined in the template file in addition to updating the current step
# of the simulation.
for simulation.current_step in range(simulation.beginning_step, simulation.end_step + 1):
    # Prints the current step, number of cells, and records run time. Used to give an idea of the simulation progress.
    functions.info(simulation)

    # Refreshes the graph used to represent cells as nodes and nearby neighbors as edges.
    functions.check_neighbors(simulation)

    # Updates cells by adjusting trackers for differentiation and division based on intracellular, intercellular,
    # and extracellular conditions.
    functions.cell_update(simulation)

    # Updates each of the extracellular gradients via the finite difference method.
    functions.update_diffusion(simulation)

    # Adds/removes cells to/from the simulation either all together or in desired groups of cells. If done in
    # groups, the handle_movement() function will be used to better represent asynchronous division and death.
    functions.update_queue(simulation)

    # Find the nearest NANOG high, GATA6 high, and differentiated cell within a fixed radius, used for movement
    functions.nearest(simulation)

    # Locate the diffusion point (within a fixed radius) that has the highest FGF4 concentration.
    # not currently in use.
    functions.highest_fgf4(simulation)

    # Gets motility forces depending on a variety of factors involving state and presence of neighbors
    functions.cell_motility(simulation)

    # Moves the cells to a state of physical equilibrium so that there is minimal overlap of cells, while also
    # applying forces from the previous motility_cells() function.
    functions.handle_movement(simulation)

    # The first function will save a 2D image of the space, the second will create a csv with each row corresponding to
    # an individual cell, and the last will save performance statistics to a running csv.
    output.step_image(simulation)
    output.step_csv(simulation)
    output.temporary(simulation)
    output.simulation_data(simulation)

# Ends the simulation by closing any necessary files.
output.finish_files(simulation)
