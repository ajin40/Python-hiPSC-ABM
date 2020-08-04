"""

This is the Python file that you run to begin the simulation. Before you begin, make sure you have
 updated "Simulation.Simulation(...)" such that it represents the string that is the path pointing to the
 .txt file a.k.a. the template file used for simulation information. All lines indicated with "(base)" are
 necessary to for the model to work. Removing such lines may cause the model to either run incorrectly
 or not at all. Functions underneath the for loop are run each step, while functions before or after
 will be run before or after all steps are executed.

"""
import Output
import Simulation
import Functions

# setup() will create an instance of the Simulation class that holds all relevant information of the model.
# this is done by reading a template .txt file that contains all initial parameters of the model.
simulation = Simulation.Simulation("C:\\Python37\\Seed Project\\Model\\template.txt")

# Will locate the diffusion points for the extracellular gradient to bins, used for chemotactic movement
# called once as the locations don't change. not currently in use.
simulation.setup_diffusion_bins()

# This will loop over all steps defined in the template file in addition to updating the current step
# of the simulation.
for simulation.current_step in range(simulation.beginning_step, simulation.end_step + 1):
    # Prints the current step, number of cells, and records run time. Used to give an idea of the simulation progress.
    Functions.info(simulation)

    # Refreshes the graph used to represent cells as nodes and nearby neighbors as edges.
    Functions.check_neighbors(simulation)

    # Updates cells by adjusting trackers for differentiation and division based on intracellular, intercellular,
    # and extracellular conditions.
    simulation.cell_update()

    # Updates each of the extracellular gradients via the finite difference method.
    Functions.update_diffusion(simulation)

    # Adds/removes cells to/from the simulation either all together or in desired groups of cells. If done in
    # groups, the handle_movement() function will be used to better represent asynchronous division and death.
    Functions.update_queue(simulation)

    # Find the nearest NANOG high, GATA6 high, and differentiated cell within a fixed radius, used for movement
    simulation.nearest()

    # Locate the diffusion point (within a fixed radius) that has the highest FGF4 concentration.
    # not currently in use.
    simulation.highest_fgf4()

    # Gets motility forces depending on a variety of factors involving state and presence of neighbors
    simulation.cell_motility()

    # Moves the cells to a state of physical equilibrium so that there is minimal overlap of cells, while also
    # applying forces from the previous motility_cells() function.
    Functions.handle_movement(simulation)

    # The first function will save a 2D image of the space, the second will create a csv with each row corresponding to
    # an individual cell, and the last will save performance statistics to a running csv.
    Output.step_image(simulation)
    Output.step_csv(simulation)
    Output.simulation_data(simulation)

# Ends the simulation by closing any necessary files.
Output.finish_files(simulation)
