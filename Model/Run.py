from Model import Input
from Model import Output



# Creates a list of simulation instances each corresponding to the setup file    (base)
Simulations = Input.Setup()

# Runs the simulations in succession    (base)
for Simulation in Simulations:

    # Adds the initial concentration amounts to the grid for each diffusing extracellular molecule    (base)
    Simulation.initialize_gradients()

    # This will run the simulation until the end time is reached    (base)
    while Simulation.time_counter <= Simulation.end_time:

        # Prints number of cells, timestep, amount of cells being removed and added
        Simulation.info()

        # Updates each of the gradients/molecules by adjusting concentrations
        Simulation.update_gradients()

        # If cells are by themselves for too long, they will be removed from the simulation
        Simulation.kill_cells()

        # If enough neighbor differentiated cells surround a pluripotent cell, it may cause differentiation
        Simulation.diff_surround_cells()

        # Updates cells by adjusting trackers for differentiation and division
        Simulation.update_cells()

        # Represents growth as mass increase and recalculates radii based on mass increases
        Simulation.change_size_cells()

        # Adds/removes objects at once to/from the simulation includes handling collisions when cells are added
        Simulation.update_cell_queue()

        # Allows the cells in motion to move in a random fashion
        Simulation.randomly_move_cells()

        # Moves the cells to a state of physical equilibrium so that there is minimal overlap between cells
        Simulation.handle_collisions()

        # Checks for neighboring cells
        Simulation.check_neighbors()

        # Saves a 2D image and a .csv file containing key simulation information for each cell     (base)
        Output.save_file(Simulation)

        # Increases the time counter for the while loop    (base)
        Simulation.time_counter += Simulation.time_step

    # Turns all of the images into a video    (base)
    Output.image_to_video(Simulation)