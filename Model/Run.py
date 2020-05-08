"""

This is the Python file that you run to begin the simulation. Before you begin, make sure you have
updated Locations.txt such that it points to the template files and where you would like the files
to be outputted. All lines indicated with "(base)" are necessary to for the model to function.
Removing such lines will cause the model to either run incorrectly or not at all. Feel free to modify
or delete any functions that are not labeled in this fashion. Add any necessary functions here to
incorporate them into the model.

Input.py and Output.py are standalone files that are not part of any class. Input reads the
template files and creates an instance of the Simulation class, which holds all created cell
and extracellular objects. Output will take the Simulation and create images, CSVs, and a
video based on a collection of the images.

"""
import Input
import Output

# Creates a list of Simulation instances each corresponding to a template .txt file in Setup_files. Then
# runs each simulation in succession, allowing for multiple simulations to be run one after another if you
# were to use a high computing cluster/node.   (base)
list_of_simulations = Input.setup()
for simulation in list_of_simulations:

    # Adds the initial concentration amounts to the space for each instance of the extracellular class    (base)
    simulation.initialize_diffusion()

    # This will loop over all steps. The first image and CSV produced are based on starting conditions. The
    # following outputs are representative of one step run by the model.    (base)
    for step in range(simulation.beginning_step, simulation.end_step + 1):

        # Updates the simulation instance variable to the current step number.    (base)
        simulation.current_step = step

        # Prints the current step number and the count of cells. Gives an idea of how the model is running.
        simulation.info()

        # Updates each of the extracellular gradients by adjusting concentrations.
        simulation.update_diffusion()

        # Updates the graph containing all edges that represent a cell's neighbors. Each time the cells move this must
        # be run to maintain accuracy of which cells surround a given cell.
        simulation.check_neighbors()

        # If cells are by themselves for too long, they will be added to the queue that removes the cells.
        simulation.kill_cells()

        # If enough neighbor differentiated cells surround a pluripotent cell, it may induce differentiation
        simulation.diff_surround_cells()

        # Applies forces to each cell based on random or Guye movement
        simulation.motility_cells()

        # Updates cells by adjusting trackers for differentiation and division based on intracellular, intercellular,
        # and extracellular conditions.
        simulation.update_cells()

        # Adds/removes cells to/from the simulation either all together or in desired amounts of cell. If done in
        # groups, the handle_movement() function will be called to better represent asynchronous division and death
        simulation.update_cell_queue()

        # Moves the cells to a state of physical equilibrium so that there is minimal overlap between cells
        simulation.handle_movement()

        # Saves a snapshot of the simulation at the given step. This includes images and a CSV file.     (base)
        Output.save_file(simulation)

    # Looks at all images produced by the simulation and turns them into a video in order.
    Output.image_to_video(simulation)