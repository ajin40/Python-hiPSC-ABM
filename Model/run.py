import numpy as np
import random as r

import setup
import output
import functions
import backend


# Only start the model if this file is being run directly.
if __name__ == "__main__":
    setup.start()


def setup_cells(simulation):
    """ Here you can specify how many cells a simulation will begin with
        and define any cell arrays with initial conditions.

        How-to:
            The lines below first add 1000 general cells into the simulation and then add 500 cells marked with the
            "GATA6_high" parameter. This allows for specifying initial conditions to just the 500 cells.

            simulation.add_cells(1000)
            simulation.add_cells(500, cell_type="GATA6_high")

            The cell_array() method will generate a NumPy array used to hold all values of the cells. The first
            argument is required to name the array as an instance variable in the Simulation object. Other optional
            parameters can be used to customize the array. Note: the array will default to a 1-dimension array of
            zeros represented as floats. See examples below.

            simulation.cell_array("colors", lambda: "green", dtype=str)
            simulation.cell_array("colors", lambda: "red", cell_type="GATA6_high")
            simulation.cell_array("locations", override=some_array)
            simulation.cell_array("motility_forces", vector=3)

    """
    # Add the specified number of NANOG/GATA6 high cells and create cell type GATA6_high for initial parameters.
    simulation.add_cells(simulation.num_nanog)
    simulation.add_cells(simulation.num_gata6, cell_type="GATA6_high")

    # Create the following cell arrays with initial conditions.
    simulation.cell_array("locations", override=np.random.rand(simulation.number_cells, 3) * simulation.size)
    simulation.cell_array("radii", func=lambda: simulation.min_radius)
    simulation.cell_array("motion", dtype=bool, func=lambda: True)
    simulation.cell_array("FGFR", dtype=int, func=lambda: r.randrange(0, simulation.field))
    simulation.cell_array("ERK", dtype=int, func=lambda: r.randrange(0, simulation.field))
    simulation.cell_array("GATA6", dtype=int)
    simulation.cell_array("NANOG", dtype=int, func=lambda: r.randrange(1, simulation.field))
    simulation.cell_array("states", dtype=str, func=lambda: "Pluripotent")
    simulation.cell_array("death_counters", dtype=int, func=lambda: r.randrange(0, simulation.death_thresh))
    simulation.cell_array("diff_counters", dtype=int, func=lambda: r.randrange(0, simulation.pluri_to_diff))
    simulation.cell_array("div_counters", dtype=int, func=lambda: r.randrange(0, simulation.pluri_div_thresh))
    simulation.cell_array("fds_counters", dtype=int, func=lambda: r.randrange(0, simulation.fds_thresh))
    simulation.cell_array("motility_forces", vector=3)
    simulation.cell_array("jkr_forces", vector=3)
    simulation.cell_array("nearest_nanog", dtype=int, func=lambda: -1)
    simulation.cell_array("nearest_gata6", dtype=int, func=lambda: -1)
    simulation.cell_array("nearest_diff", dtype=int, func=lambda: -1)

    # Update the "GATA6_high" cells with alternative initial conditions.
    simulation.cell_array("GATA6", cell_type="GATA6_high", func=lambda: r.randrange(1, simulation.field))
    simulation.cell_array("NANOG", cell_type="GATA6_high", func=lambda: 0)


def steps(simulation):
    """ This method is used to specify the order of the methods that
        happen before, during, and after the simulation steps.

        How-to:
            before_steps(simulation)
            for simulation.current_step in range(simulation.beginning_step, simulation.end_step + 1):
                during_steps(simulation)
                some_method(simulation)
                other_method(simulation)
            after_steps(simulation)

    """
    for simulation.current_step in range(simulation.beginning_step, simulation.end_step + 1):
        # Records model run time for the step and prints the current step/number of cells.
        backend.info(simulation)

        # Finds the neighbors of each cell that are within a fixed radius and store this info in a graph.
        functions.get_neighbors(simulation, distance=0.000015)

        # Updates cells by adjusting trackers for differentiation, division, growth, etc. based on intracellular,
        # intercellular, and extracellular conditions through a series of separate methods.
        functions.cell_death(simulation)
        functions.cell_diff_surround(simulation)
        functions.cell_division(simulation)
        functions.cell_growth(simulation)
        functions.cell_pathway(simulation)

        # Simulates molecular diffusion the specified extracellular gradient via the forward time centered space method.
        functions.update_diffusion(simulation, "fgf4_values")
        # functions.update_diffusion(simulation, "fgf4_alt")

        # Adds/removes cells to/from the simulation either all together or in desired groups of cells. If done in
        # groups, the handle_movement() function will be used to better represent asynchronous division and death.
        functions.update_queue(simulation)

        # Finds the nearest NANOG high, GATA6 high, and differentiated cells within a fixed radius. This provides
        # information that can be used for approximating cell motility.
        functions.nearest(simulation, distance=0.000015)

        # Calculates the direction/magnitude of a cell's movement depending on a variety of factors such as state
        # and presence of neighbors.
        functions.cell_motility(simulation)
        # functions.eunbi_motility(simulation)

        # Through the series of methods, attempt to move the cells to a state of physical equilibrium between adhesive
        # and repulsive forces acting on the cells, while applying active motility forces.
        for _ in range(simulation.move_steps):
            functions.jkr_neighbors(simulation)
            functions.get_forces(simulation)
            functions.apply_forces(simulation)

        # Saves multiple forms of information about the simulation at the current step, including an image of the
        # space, CSVs with values of the cells, a temporary pickle of the Simulation object, and performance stats.
        # See the outputs.txt template file for turning off certain outputs.
        output.step_image(simulation)
        output.step_values(simulation)
        output.step_gradients(simulation)
        output.step_tda(simulation, in_pixels=True)
        output.temporary(simulation)
        output.simulation_data(simulation)

    # Ends the simulation by creating a video from all of the step images
    output.create_video(simulation, fps=6)
