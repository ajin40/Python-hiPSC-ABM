import Input
import Output
import Functions
import os
import shutil
import numpy as np
import Parallel



Simulation = Input.Setup()

try:
    os.mkdir(Simulation.path + Simulation._sep + Simulation.name)
except OSError:
    # directory already exists overwrite it
    print("Directory already exists... overwriting directory")

# setup grid and patches
Functions.initialize_grid(Simulation)

# run check_edge() to create connections between cells
if Simulation.parallel:
    Parallel.check_edge_gpu(Simulation)
else:
    Functions.check_edges(Simulation)

Functions.handle_collisions(Simulation)

# save the first image and data of simulation
Output.save_file(Simulation)

shutil.copy("Initials.txt", Simulation.path + Simulation._sep + Simulation.name + Simulation._sep)

# run simulation until end time
while Simulation.time_counter <= Simulation.end_time:

    np.random.shuffle(Simulation.objects)

    print("Time: " + str(Simulation.time_counter))
    print("Number of objects: " + str(len(Simulation.objects)))

    Functions.kill_cells(Simulation)

    # updates all of the objects (motion, state, booleans)

    if Simulation.parallel:
        Parallel.update_grid_gpu(Simulation)
    else:
        Functions.update_grid(Simulation)

    Functions.update(Simulation)

    # sees if cells can differentiate based on pluripotent cells surrounding by differentiated cells
    Functions.diff_surround(Simulation)

    # adds/removes all objects from the simulation
    Functions.update_object_queue(Simulation)

    # create/break connections between cells depending on distance apart
    if Simulation.parallel:
        Parallel.check_edge_gpu(Simulation)
    else:
        Functions.check_edges(Simulation)

    # moves cells in "motion" in a random fashion
    # self.random_movement()
    Functions.handle_collisions(Simulation)

    # increments the time by time step
    Simulation.time_counter += Simulation.time_step

    # saves the image file and txt file with all important information
    Output.save_file(Simulation)

# turns all images into a video at the end
Output.image_to_video(Simulation)