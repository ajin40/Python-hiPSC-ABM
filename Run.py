#########################################################
# Name:    Run                                          #
# Author:  Jack Toppen                                  #
# Date:    3/4/20                                       #
#########################################################
import Input
import Output
import Functions


"""
Run this file in order to the simulation as a whole
Input.Setup() takes Initials.txt and creates an instance of the Simulation class
This instance hold all import important parameters and holds the StemCell objects too
"""


# This calls the Simulation class allowing something to hold all important parameters to the model.
Simulation = Input.Setup()



# Checks the name and save a copy of Initials.txt
Functions.check_name(Simulation)

# Set the grid up with initial concentrations of FGF4
Functions.initialize_grid(Simulation)

# Check for neighbors surrounding a cell
Functions.check_edges(Simulation)

# Move the cells to a state a equilibrium so that there is minimal overlap
Functions.handle_collisions(Simulation)

# Save the first image and stats of the simulation. This is before any updates
Output.save_file(Simulation)

# run simulation until end time
while Simulation.time_counter <= Simulation.end_time:

    # prints the time step and increases the time counter
    Functions.info(Simulation)

    # kills cells that are without neighbors for too long
    Functions.kill_cells(Simulation)

    # updates the grid by degrading the amount of FGF4
    Functions.update_grid(Simulation)

    # updates all of the objects (motion, state, booleans)
    Functions.update(Simulation)

    # sees if cells can differentiate based on pluripotent cells surrounding by differentiated cells
    Functions.diff_surround(Simulation)

    # adds/removes all objects from the simulation
    Functions.update_object_queue(Simulation)

    # create/break connections between cells depending on distance apart
    Functions.check_edges(Simulation)

    # moves cells in "motion" in a random fashion
    # Functions.random_movement(Simulation)

    # Move the cells to a state a equilibrium so that there is minimal overlap
    Functions.handle_collisions(Simulation)

    # saves the image file and txt file with all important information
    Output.save_file(Simulation)

# turns all images into a video at the end
Output.image_to_video(Simulation)