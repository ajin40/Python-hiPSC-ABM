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
Input.Setup() takes Setup_files and creates an instance of the Simulation class
This instance hold all import important parameters and holds the StemCell objects too

Stuff I need to do:
Variable grid size
Renaming everything
Documentation
Color schemes
Cuda for collisions
Collisions with boundaries
Random movement
Resize boundaries so that cells are visible
Variable size and mass
Variable Boolean function length
Error handling
Ease of use
Determine libraries
Delete unnecessary crap
Cuda Boolean update
Variable Boolean updates
Asynchronous updates?
Cell death
Imposing membranes
Saving variability
Numpy array sizes
Cuda block/thread optimization
3D
Boolean values setup
Easter eggs

"""



# This calls the Simulation class allowing something to hold all important parameters to the model.
Simulations = Input.Setup()


for Simulation in Simulations:

    # Sets up all of the defined gradients with initial concentrations and parameters
    Simulation.initialize_gradients()

    # Check for neighbors surrounding a cell
    Functions.check_edges(Simulation)

    # Move the cells to a state a equilibrium so that there is minimal overlap
    Functions.handle_collisions(Simulation)

    # Save the first image and stats of the simulation. This is before any updates
    Output.save_file(Simulation)

    # run simulation until end time
    while Simulation.time_counter <= Simulation.end_time:

        # prints the time step and increases the time counter
        Simulation.info()

        # kills cells that are without neighbors for too long
        Functions.kill_cells(Simulation)

        # updates the grid by degrading the amount of FGF4
        Simulation.update_gradients()

        # updates all of the objects (motion, state, booleans)
        Simulation.update_cells()

        # sees if cells can differentiate based on pluripotent cells surrounding by differentiated cells
        Functions.diff_surround(Simulation)

        # adds/removes all objects from the simulation
        Simulation.update_object_queue()

        # create/break connections between cells depending on distance apart
        Functions.check_edges(Simulation)

        # moves cells in "motion" in a random fashion
        Functions.random_movement(Simulation)

        # move the cells to a state a equilibrium so that there is minimal overlap
        Functions.handle_collisions(Simulation)

        # saves the image file and txt file with all important information
        Output.save_file(Simulation)

        # increase time closer to threshold
        Simulation.time_counter += Simulation.time_step

    # turns all images into a video at the end
    Output.image_to_video(Simulation)