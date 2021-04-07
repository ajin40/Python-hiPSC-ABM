import os
import pickle
import shutil
import psutil

from backend import output_dir, start_params, Paths
from cellsimulation import CellSimulation


def start():
    """ Configures/runs the model based on the specified
        simulation mode.
    """
    # read paths.yaml to get the output directory where simulation folders are outputted
    output_path = output_dir()

    # get the name/mode of the simulation and make sure there is a output directory named after this simulation
    name, mode, final_step = start_params(output_path, possible_modes=[0, 1, 2, 3])

    # create Paths object for storing important output paths
    paths = Paths(name, output_path)

    # -------------------------- new simulation ---------------------------
    if mode == 0:
        # copy model files to simulation directory, ignoring __pycache__ files
        copy_path = paths.main_path + name + "_copy"
        shutil.copytree(os.getcwd(), copy_path, ignore=shutil.ignore_patterns("__pycache__"))

        # create CellSimulation object
        sim = CellSimulation(paths, name)

        # add agent arrays to CellSimulation object and run the simulation steps
        sim.agent_initials()
        sim.steps()

    # ---------------- continuation of previous simulation ----------------
    elif mode == 1:
        # load previous CellSimulation object from pickled file
        file_name = paths.main_path + name + "_temp.pkl"
        with open(file_name, "rb") as file:
            sim = pickle.load(file)

        # update the following
        sim.paths = paths    # change paths object in case of system changing
        sim.beginning_step = sim.current_step + 1    # update starting step
        sim.end_step = final_step    # update new final step from start_params()

        # run the simulation steps
        sim.steps()

    # ------------------------- images to video ---------------------------
    elif mode == 2:
        # create CellSimulation object for video and path information
        sim = CellSimulation(paths, name)

        # make the video with images from past simulation
        sim.create_video()

    # --------------------- zip a simulation folder -----------------------
    elif mode == 3:
        # remove the separator of the path to the simulation directory
        print("Compressing \"" + name + "\" simulation...")
        simulation_dir = paths.main_path[:-1]

        # zip a copy of the folder and save it to the output directory
        shutil.make_archive(simulation_dir, "zip", root_dir=output_path, base_dir=name)
        print("Done!")


# only call start() if this file is being run directly
if __name__ == "__main__":
    # get process for run.py and set priority to high
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)

    # start the model
    start()
