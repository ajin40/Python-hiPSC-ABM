import numpy as np
import random as r

import input
import output
import functions
import backend


# start the model only if this file is being run directly
if __name__ == "__main__":
    input.start()


def setup_cells(simulation):
    """ Specify how many cells a simulation will begin with and any cell types
        which are used to create initial parameters for those cells.

        Example:
            Add cells into the simulation with the add_cells() method. The first argument is the number of
            cells, and the optional keyword argument is used to designate an addition of cells with a specific
            cell type which can be referenced by the cell_array() method for assigning initial conditions.

                simulation.add_cells(1000)
                simulation.add_cells(500, cell_type="GATA6_high")

            The cell_array() will generate NumPy arrays used to hold all values of the cells. The first argument
            denotes the name of the instance variable generated for that array in the Simulation object. The
            following parameters can be used to set initial values, to specify data types, and to create 2D arrays.

                simulation.cell_array("FGFR", lambda: r.randrange(0, simulation.field), dtype=int)
                simulation.cell_array("locations", override=np.random.rand(simulation.number_cells) * simulation.size)
                simulation.cell_array("motility_forces", dtype=float, vector=3)
                simulation.cell_array("colors", lambda: "green", dtype=str)
                simulation.cell_array("colors", lambda: "red", cell_type="GATA6_high")
    """
    # add the specified number of NANOG/GATA6 high cells and create cell type GATA6_high for initial parameters
    simulation.add_cells(simulation.num_nanog)
    simulation.add_cells(simulation.num_gata6, cell_type="GATA6_high")

    # create the following cell arrays with initial conditions, arrays will default to zero
    simulation.cell_array("locations", override=np.random.rand(simulation.number_cells) * simulation.size)
    simulation.cell_array("radii", lambda: simulation.min_radius, dtype=float)
    simulation.cell_array("motion", lambda: True, dtype=bool)
    simulation.cell_array("FGFR", lambda: r.randrange(0, simulation.field), dtype=int)
    simulation.cell_array("ERK", lambda: r.randrange(0, simulation.field), dtype=int)
    simulation.cell_array("GATA6", dtype=int)
    simulation.cell_array("NANOG", lambda: r.randrange(1, simulation.field), dtype=int)
    simulation.cell_array("states", lambda: "Pluripotent", dtype=str)
    simulation.cell_array("death_counters", lambda: r.randrange(0, simulation.death_thresh), dtype=int)
    simulation.cell_array("diff_counters", lambda: r.randrange(0, simulation.pluri_to_diff), dtype=int)
    simulation.cell_array("div_counters", lambda: r.randrange(0, simulation.pluri_div_thresh), dtype=int)
    simulation.cell_array("fds_counters", lambda: r.randrange(0, simulation.fds_thresh), dtype=int)
    simulation.cell_array("motility_forces", dtype=float, vector=3)
    simulation.cell_array("jkr_forces", dtype=float, vector=3)
    simulation.cell_array("nearest_nanog", lambda: -1, dtype=int)
    simulation.cell_array("nearest_gata6", lambda: -1, dtype=int)
    simulation.cell_array("nearest_diff", lambda: -1, dtype=int)

    # update the "GATA6_high" cells with alternative initial conditions
    simulation.cell_array("GATA6", lambda: r.randrange(1, simulation.field), cell_type="GATA6_high")
    simulation.cell_array("NANOG", lambda: 0, cell_type="GATA6_high")


def steps(simulation):
    """ Specify the order of the methods for each step and include
        any methods that are called before or after all steps.

        Example:
            functions.before_steps(simulation)

            for simulation.current_step in range(simulation.beginning_step, simulation.end_step + 1):
                functions.during_steps(simulation)

            functions.after_steps(simulation)
    """
    for simulation.current_step in range(simulation.beginning_step, simulation.end_step + 1):
        # Records model run time for the step and prints the current step/number of cells,
        backend.info(simulation)

        # Finds the neighbors of each cell that are within a fixed radius and store this info in a graph.
        functions.get_neighbors(simulation)

        # Updates cells by adjusting trackers for differentiation, division, growth, etc. based on intracellular,
        # intercellular, and extracellular conditions through a series of separate methods.
        functions.cell_death(simulation)
        functions.cell_diff_surround(simulation)
        functions.cell_division(simulation)
        functions.cell_growth(simulation)
        functions.cell_pathway(simulation)

        # Simulates the diffusion for each of the extracellular gradients via the forward time centered space method.
        functions.update_diffusion(simulation)

        # Adds/removes cells to/from the simulation either all together or in desired groups of cells. If done in
        # groups, the handle_movement() function will be used to better represent asynchronous division and death.
        functions.update_queue(simulation)

        # Finds the nearest NANOG high, GATA6 high, and differentiated cells within a fixed radius. This provides
        # information that can be used for approximating cell motility.
        functions.nearest(simulation)

        # Calculates the direction/magnitude of a cell's movement depending on a variety of factors such as state
        # and presence of neighbors.
        functions.cell_motility(simulation)
        # functions.eunbi_motility(simulation)

        # Attempts to move the cells to a state of physical equilibrium between adhesive and repulsive forces acting on
        # the cells, while applying active motility forces from the previous cell_motility() function.
        functions.handle_movement(simulation)

        # Saves multiple forms of information about the simulation at the current step, including an image of the space,
        # CSVs with values of the cells, a temporary pickle of the Simulation object, and performance stats.
        output.step_outputs(simulation)

    # Ends the simulation by creating a video from all of the step images
    output.create_video(simulation)
