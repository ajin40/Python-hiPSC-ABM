import random as r
from Model_Simulation import *
from Model_SimulationObject import *
import numpy as np
import os

def main():

    # where the model will output images and cell locations ex. ("C:\\Python27\\Model")
    path = "C:\\Python27\\MEGA-ARRAY"

    # total time step counter limit to less than 30
    Run_Time = 10.0
    
    # when does the model begin (usually 1)
    Start_Time = 1
    
    # change in time between time steps
    Time_Step = 1

    # size of grid can be 3D with extra layers (layers, rows, columns)
    size = (1, 1000, 1000)

    # define boolean functions here ex. ("(x3+1) * (x4+1)")
    funct_1 = "x5"
    funct_2 = "x1 * x4"
    funct_3 = "x2"
    funct_4 = "x5 + 1"
    funct_5 = "(x3+1) * (x4+1)"

    # add functions to the list
    functions = [funct_1, funct_2, funct_3, funct_4, funct_5]

    # radius of each cell depending on state ex. ([state_1, state_2])
    radius = np.asarray([8.0])

    # length of time steps required for a pluripotent cell to divide
    pluri_div_thresh = 18.0

    # length of time steps required for a differentiated cell to divide
    diff_div_thresh = 18.0

    # length of time steps required for a pluripotent cell to differentiate
    pluri_to_diff = 4.0

    # max interaction length between two cells (larger length...longer run time)
    interaction_max = 12

    # amount of differentiated cells needed to surround a pluripotent cell and differentiate it
    diff_surround = 10




#######################################################################################################################

    # names the file
    Model_ID = newDirect(path)

    # initializes simulation class which holds all information about the simulation
    sim = Simulation(Model_ID, path, Start_Time, Run_Time, Time_Step, pluri_div_thresh, diff_div_thresh, pluri_to_diff, size, interaction_max,diff_surround, functions)

    # counts the number of cells in "cell_coords"
    f = open(os.getcwd() + "/cell_coords.txt")
    cells_txt = f.read()
    cells = cells_txt.split('\n')
    count_cells = len(cells) - 1

    # loops over all cells and creates a stem cell object for each one
    for i in range(0, count_cells):
        ID = i
        line = cells[i].split(',')
        point = [float(line[1]), float(line[2])]
        state = str(line[3])
        x1 = float(line[4])
        x2 = float(line[5])
        x3 = float(line[6])
        x4 = float(line[7])
        x5 = float(line[8])
        booleans = [x1, x2, x3, x4, x5]


        diff_timer = pluri_to_diff * r.random()

        division_timer = (pluri_div_thresh + diff_div_thresh) * 0.5 * r.random()

        sim_obj = StemCell(point,radius,ID,booleans,state,diff_timer,division_timer)


        # add object to simulation
        sim.add_object(sim_obj)

        # IDs the cell
        sim.inc_current_ID()

    try:
        sim.collide()
    except:
        sim.collide_lowDens()



    # run the simulation
    sim.run()
    
def newDirect(path):
    """
    This function opens the specified save path and finds the highest folder number.
    It then returns the next highest number as a name for the currently running simulation.
    """

    files = os.listdir(path)
    n = len(files)
    number_files = []
    if n > 0:
        for i in range(n):
            try:
                number_files.append(float(files[i]))
            except ValueError:
                pass
        if len(number_files) > 0:
            k = max(number_files)
        else:
            k = 0
    else:
        k = 0

    return k + 1.0

