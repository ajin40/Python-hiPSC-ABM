#########################################################
# Name:    Run                                          #
# Author:  Jack Toppen                                  #
# Date:    3/17/20                                      #
#########################################################
import Input
import Output

"""
Run this file in order to run the program as a whole. Everything is pretty malleable 
but there are a few lines of code that should remain to keep the model running;
however, this isn't set in stone. Important lines are marked with "(base)". I encourage
the addition, deletion, and editing of the function other than "(base)". I have merely
provided an example of functions I put together for my purposes.

Here's what each of the files do:
    Input: This will open up the Setup_files directory and parse through each of the
    .txt files. These files are written in a certain way so the they can be interpreted
    easily by the model. Though, if you need to change them, directions will be in 
    Input. The files contain important parameters on how the model will be set up for
    each simulation run.
    
    Simulation:
    
    Gradient:
    
    Cell: The Cell class is housed here. Each cell in the simulation is representative
    of single cell in an experiment. All of the cell objects are held by an array as
    an instance in the Simulation class. You'll see that the cells will have instance 
    variables that correspond to values such as radius, mass, state, and many others.
    The class also holds methods for updating the Cell instance variables. You may ask
    why some methods for the Cell class weren't integrated into the Simulation class;
    however, my answer for that is simplicity and interpretation.
    
    Parallel:
    
    Output: All outgoing data from the model will be processed in this file. For each
    time step, an image and a .csv file will be produced. The image provides a visual
    representation of the cells in the simulation. The .csv is a way of transporting
    data from the model to other forms of statistical analysis. The .csv will contain
    information for each cell such as location and whether it is differentiated
    or pluripotent.
    

If you run into issues, please read the following quote.

"That's just how the peaches roll."
                     - Garret Fritz
"""


# Creates a list of simulation instances each corresponding to the setup file    (base)
Simulations = Input.Setup()

# Runs the simulations in succession    (base)
for Simulation in Simulations:

    # Adds the initial concentration amounts to the grid for each gradient/molecule    (base)
    Simulation.initialize_gradients()

    # This will run the simulation until the end time is reached    (base)
    while Simulation.time_counter <= Simulation.end_time:

        # Prints important information corresponding to the simulation as it runs
        Simulation.info()

        # Updates each of the gradients/molecules by adjusting concentrations
        Simulation.update_gradients()

        # Determines if two cells are close enough together to designate a neighbor
        Simulation.check_neighbors()

        # If cells are by themselves for too long, they will be removed from the simulation
        Simulation.kill_cells()

        # Updates cells by adjusting values differentiation and division
        Simulation.update_cells()

        # If enough neighbor differentiated cells surround a pluripotent cell, it may cause differentiation
        Simulation.diff_surround_cells()

        # Adjusts the mass and radius of the cell
        Simulation.change_size_cells()

        # Adds/removes objects at once to/from the simulation
        Simulation.update_cell_queue()

        # Allows the cells in motion to move in a random fashion
        Simulation.random_movement()

        # Moves the cells to a state of equilibrium so that there is minimal overlap
        Simulation.handle_collisions()

        # Saves a 2D image and a .csv file containing key information from each cell    (base)
        Output.save_file(Simulation)

        # Increases the time counter for the while loop    (base)
        Simulation.time_counter += Simulation.time_step

    # Turns all of the images into a video    (base)
    Output.image_to_video(Simulation)