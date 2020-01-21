import random as r
from Model_Simulation import *
from Model_SimulationObject import *
import numpy as np
import os

def main():

    # path
    path = "C:\\Python27\\MEGA-ARRAY"

    # run time
    Run_Time = 5.0
    
    # when does the model begin (usually 1)
    Start_Time = 1
    
    # change in time for steps
    Time_Step = 1

    # size of array (dimensions, rows, columns)
    size = (2, 1000, 1000)

    # functions
    funct_1 = "x5"
    funct_2 = "x1 * x4"
    funct_3 = "x2"
    funct_4 = "x5 + 1"
    funct_5 = "(x3+1) * (x4+1)"

    # radius of each cell state
    radius = np.asarray([6.0])

    # mitosis threshold for pluripotent cells
    pluri_div_thresh = 3.0

    # mitosis threshold for differentiated cells
    diff_div_thresh = 3.0

    # differentiating threshold
    pluri_to_diff = 3.0

#######################################################################################################################
    functions = [funct_1, funct_2, funct_3, funct_4, funct_5]




    Model_ID = newDirect(path)



    # initializes simulation class
    sim = Simulation(Model_ID, path, Start_Time, Run_Time, Time_Step, pluri_div_thresh, diff_div_thresh, pluri_to_diff, size, functions)


    # counts the number of cells
    f = open(os.getcwd() + "/cell_coords.txt")
    cells_txt = f.read()
    cells = cells_txt.split('\n')
    count_cells = len(cells) - 1


    # loops over all cells
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

        division_timer = 0

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

