import random as r
from Model_Simulation import *
from Model_SimulationObjects import *
import numpy as np


def main(Model_ID, Run_Time, path):
    
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

    # production/degradation constants for each extracellular molecule
    source_sink_params = np.asarray([[[5e-14, 4e-14], [5e-14, 4e-14]], ])

    # mitosis threshold for pluripotent cells
    pluri_threshold = 1.0

    # mitosis threshold for differentiated cells
    diff_threshold = 1.0












#######################################################################################################################
    functions = [funct_1, funct_2, funct_3, funct_4, funct_5]


    # initializes simulation class
    sim = Simulation(Model_ID, path, Start_Time, Run_Time, Time_Step, pluri_threshold, diff_threshold, source_sink_params, size, functions)


    # counts the number of cells
    f = open(os.getcwd() + "/cell_coords.txt")
    cells_txt = f.read()
    cells = cells_txt.split('\n')
    count_cells = len(cells) - 1


    # loops over all cells
    for i in range(0, count_cells):
        src_snk = source_sink_params
        ID = i
        line = cells[i].split(',')
        point = [float(line[1]), float(line[2])]
        state = str(line[3])
        x1 = float(line[4])
        x2 = float(line[5])
        x3 = float(line[6])
        x4 = float(line[7])
        x5 = float(line[8])

        #
        # if state == "Pluripotent":
        #     division_time = pluri_threshold
        #
        # else:
        #     division_time = diff_threshold

        division_time = 3.0

        diff_timer = 0
        division_timer = 0
        if division_time == 0:
            div_set = 0
            sim_obj = StemCell(point,radius,ID,src_snk,x1,x2,x3,x4,x5,state,diff_timer,division_timer,Run_Time+1)
        else:
            div_set = r.random() * division_time
            sim_obj = StemCell(point,radius,ID,src_snk,x1,x2,x3,x4,x5,state,diff_timer,division_timer,division_time)


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
    
