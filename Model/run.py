import psutil
import os

from cellsimulation import CellSimulation
# from simulation import Simulation

# only call start() if this file is being run directly
if __name__ == "__main__":
    # get process for run.py and set priority to high
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)

    # start the model by calling the class method of the Simulation (or child of Simulation) class
    CellSimulation.start()
