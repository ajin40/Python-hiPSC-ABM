import input
import output
import functions

# setup() will create an instance of the Simulation class that holds all relevant information of the model.
# this is done by reading the template files that contain initial parameters of the model.
simulation = input.setup()

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
    functions.update_diffusion(simulation)

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
    functions.highest_fgf4(simulation)
    # functions.alt_highest_fgf4(simulation)
    # functions.chemotactic(simulation)

    # calculates the direction/magnitude of the movement of the cell depending on a variety of factors such as state,
    # extracellular gradient, and presence of neighbors
    functions.cell_motility(simulation)
    # functions.alt_cell_motility(simulation)

    # moves the cells to a state of physical equilibrium between adhesive and repulsive forces acting on the cells,
    # while also applying active movement forces from the previous cell_motility() function.
    functions.handle_movement(simulation)

    # saves multiple forms of information about the simulation at the current step, including an image of the space,
    # csvs with values of the cells, a temporary pickle of the Simulation instance, and performance stats.
    output.step_outputs(simulation)

# ends the simulation by creating a video from all of the images created by the simulation
output.create_video(simulation)
