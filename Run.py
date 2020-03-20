#########################################################
# Name:    Run                                          #
# Author:  Jack Toppen                                  #
# Date:    3/17/20                                      #
#########################################################
import Input
import Output



"""
Run this file in order to the simulation as a whole.


If you run into issues, please read the following quote.

"That's just how the peaches roll."
                     - Garret Fritz
"""


# This calls the Simulation class allowing something to hold all important parameters to the model.
Simulations = Input.Setup()

# loops over all simulations and runs them one at a time
for Simulation in Simulations:

    # Sets up all of the defined gradients with initial concentrations and parameters
    Simulation.initialize_gradients()

    # Check for neighbors surrounding a cell and adds them to the graph
    Simulation.check_neighbors()

    # Move the cells to a state a equilibrium so that there is minimal overlap
    Simulation.handle_collisions()

    # Save the first image and stats of the simulation. This is before any updates
    Output.save_file(Simulation)

    # run simulation until end time
    while Simulation.time_counter <= Simulation.end_time:

        # prints the time step and increases the time counter
        Simulation.info()

        # updates the grid by degrading the amount of FGF4
        Simulation.update_gradients()

        # create/break connections between cells depending on distance apart
        Simulation.check_neighbors()

        # if cells are without a neighbor for too long it will die
        # Simulation.kill_cells()

        # updates all of the objects (motion, state, booleans)
        Simulation.update_cells()

        # moves cells in "motion" in a random fashion
        # Simulation.random_movement()

        # if enough differentiated cells surround a cell then it will increase the differentiation
        Simulation.diff_surround_cells()

        # change the sizes and shapes of the cells
        Simulation.change_size_cells()

        # adds/removes all objects from the simulation
        Simulation.update_cell_queue()

        # re-checks for neighbors before handling collisions
        Simulation.check_neighbors()

        # move the cells to a state a equilibrium so that there is minimal overlap
        Simulation.handle_collisions()

        # saves the image file and txt file with all important information
        Output.save_file(Simulation)

        # increase time closer to threshold
        Simulation.time_counter += Simulation.time_step

    # turns all images into a video at the end
    Output.image_to_video(Simulation)