import input
import output
import functions

# setup() will create an instance of the Simulation class that holds all relevant information of the model.
# this is done by reading some template files that contains initial parameters of the model.
simulation = input.setup()

# places all of the diffusion points into bins so that the model can use a bin sorting method to when
# determining highest/lowest concentrations of the extracellular gradient(s).
functions.setup_diffusion_bins(simulation)

# make directories for the outputs and create a header for the model efficiency csv
output.initialize_outputs(simulation)

# this will loop over all steps defined in the general template file in addition to updating the current step
# of the simulation. this is done to explicitly/easily show what happens at each step
for simulation.current_step in range(simulation.beginning_step, simulation.end_step + 1):
    # prints the current step, number of cells, and records run time. Used to give an idea of the simulation progress.
    functions.info(simulation)

    # updates the graph used to hold neighbors of cells within a fixed radius
    functions.check_neighbors(simulation)

    # updates cells by adjusting trackers for differentiation and division based on intracellular, intercellular,
    # and extracellular conditions.
    functions.cell_update(simulation)

    # updates each of the extracellular gradients via the finite difference method after cells have interacted
    # with the gradient in cell_update.
    functions.update_diffusion(simulation)

    # adds/removes cells to/from the simulation either all together or in desired groups of cells. if done in
    # groups, the handle_movement() function will be used to better represent asynchronous division and death.
    functions.update_queue(simulation)

    # find the nearest NANOG high, GATA6 high, and differentiated cell within a fixed radius, used for movement.
    functions.nearest(simulation)

    # find the nearest pluripotent cell within a fixed radius that is not part of the same component
    functions.nearest_cluster(simulation)

    # locate the diffusion point (within a fixed radius) that has the highest FGF4 concentration.
    # functions.highest_fgf4(simulation)

    # gets motility forces depending on a variety of factors involving state, gradient, and presence of neighbors
    functions.cell_motility(simulation)

    # moves the cells to a state of physical equilibrium so that there is minimal overlap of cells, while also
    # applying forces from the previous cell_motility() function.
    functions.handle_movement(simulation)

    # saves multiple forms of information about the simulation at the current step
    output.step_outputs(simulation)

# ends the simulation by closing any necessary files.
output.create_video(simulation)
