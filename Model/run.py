import os
import pickle
import shutil
import psutil

from backend import commandline_param, Paths, output_dir, start_params
from cellsimulation import CellSimulation


def start():
    """ Based on the desired mode, this will setup and run
        a simulation.
    """
    # get the path to the directory where simulations are outputted and the name/mode/final_step for the simulation
    output_path = output_dir()
    name, mode, final_step = start_params(output_path, possible_modes=[0, 1, 2, 3])

    # create Paths object for storing important paths and name of simulation
    paths = Paths(name, output_path)

    # -------------------------- new simulation ---------------------------
    if mode == 0:
        # copy model files to simulation output, ignore pycache files
        copy_path = paths.main_path + name + "_copy"
        shutil.copytree(os.getcwd(), copy_path, ignore=shutil.ignore_patterns("__pycache__"))

        # create Simulation object
        sim = CellSimulation(paths)

        # add cell arrays to Simulation object and run the model
        sim.agent_initials()
        sim.steps()

    # ---------------- continuation of previous simulation ----------------
    elif mode == 1:
        # load previous CellSimulation object from pickled file
        file_name = paths.main_path + name + "_temp.pkl"
        with open(file_name, "rb") as file:
            sim = pickle.load(file)

        # update the following
        sim.paths = paths  # change paths object for cross platform compatibility
        sim.beginning_step = sim.current_step + 1    # update starting step
        sim.end_step = final_step    # update final step

        # run the model
        sim.steps()

    # ------------------------- images to video ---------------------------
    elif mode == 2:
        # create CellSimulation object used to get imaging and path information
        sim = CellSimulation(paths)

        # make the video
        sim.create_video()

    # --------------------- zip a simulation directory --------------------
    elif mode == 3:
        # print statement and remove the separator of the path to the simulation directory
        print('Compressing "' + name + '" simulation...')
        simulation_dir = paths.main_path[:-1]

        # zip a copy of the directory and save it to the output directory
        shutil.make_archive(simulation_dir, 'zip', root_dir=output_path, base_dir=name)
        print("Done!")


# Only start the model if this file is being run directly.
if __name__ == "__main__":
    # get process (run.py) and set priority to high
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)

    # start the model
    start()
