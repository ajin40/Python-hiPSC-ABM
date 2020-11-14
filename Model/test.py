import test_backend
import random as r
import numpy as np

# create a Simulation instance used to store information of the simulation as it runs
simulation = test_backend.Simulation()

# define the cell arrays used to store values of the cell. each tuple corresponds to a cell array with the first index
# being the reference name, the second being the data type, and the last can be providing for a 2D array
simulation.cell_arrays(("locations", float, 3), ("radii", float), ("motion", bool), ("FGFR", int), ("ERK", int),
                       ("GATA6", int), ("NANOG", int), ("state", "<U14"), ("diff_counter", int), ("div_counter", int),
                       ("death_counter", int), ("fds_counter", int), ("motility_force", float, 3),
                       ("jkr_force", float, 3), ("rotation", float))

# define the initial parameters for all cells. these can be overridden when defining specific cell types though this
# is meant to reduce writing for cell types that only differ slightly from the base parameters.
simulation.initials("locations", lambda: np.random.rand(3) * simulation.size)
simulation.initials("motion", lambda: True)
simulation.initials("FGFR", lambda: r.randrange(0, simulation.field))
simulation.initials("ERK", lambda: r.randrange(0, simulation.field))
simulation.initials("GATA6", lambda: r.randrange(0, simulation.field))
simulation.initials("NANOG", lambda: r.randrange(0, simulation.field))
simulation.initials("state", lambda: "Pluripotent")
simulation.initials("death_counter", lambda: r.randrange(0, simulation.death_thresh))
simulation.initials("diff_counter", lambda: r.randrange(0, simulation.pluri_to_diff))
simulation.initials("div_counter", lambda: r.randrange(0, simulation.pluri_div_thresh))
simulation.initials("fds_counter", lambda: r.randrange(0, simulation.fds_thresh))
simulation.initials("motility_force", lambda: np.zeros(3, dtype=float))
simulation.initials("jkr_force", lambda: np.zeros(3, dtype=float))
simulation.initials("rotation", lambda: r.random() * 360)


