
import os
import Classes
import numpy as np
import random as r
import Functions


def Setup():
    setup_file = open(os.getcwd() + "/Initials.txt")
    setup_list = setup_file.readlines()
    parameters = []
    for i in range(len(setup_list)):
        if i % 3 == 1:
            parameters.append(setup_list[i][2:-3])

    _name = str(parameters[0])
    _path = str(parameters[1])
    _parallel = bool(parameters[2])
    _end_time = float(parameters[3])
    _time_step = float(parameters[4])
    _num_GATA6 = int(parameters[5])
    _num_NANOG = int(parameters[6])
    _stochastic = bool(parameters[7])
    _size = eval(parameters[8])
    _functions = eval(parameters[9])
    _pluri_div_thresh = float(parameters[10])
    _diff_div_thresh = float(parameters[11])
    _pluri_to_diff = float(parameters[12])
    _diff_surround_value = int(parameters[13])
    _bounds = eval(parameters[14])
    _max_fgf4 = int(parameters[15])
    _death_threshold = int(parameters[16])
    _move_time_step = float(parameters[17])
    _move_max_time = float(parameters[18])
    _spring_constant = float(parameters[19])
    _friction = float(parameters[20])
    _energy_kept = float(parameters[21])
    _neighbor_distance = float(parameters[22])
    _mass = float(parameters[23])
    _nuclear_radius = float(parameters[24])
    _cytoplasm_radius = float(parameters[25])

    # initializes simulation class which holds all information about the simulation
    simulation = Classes.Simulation(_name, _path, _end_time, _time_step, _pluri_div_thresh, _diff_div_thresh, _pluri_to_diff,
                            _size, _diff_surround_value, _functions, _parallel, _max_fgf4, _bounds, _death_threshold,
                            _move_time_step, _move_max_time, _spring_constant, _friction, _energy_kept,
                            _neighbor_distance)

    # loops over all NANOG_high cells and creates a stem cell object for each one with given parameters
    for i in range(_num_NANOG):
        ID = i
        location = np.array([r.random() * _size[1], r.random() * _size[2]])
        state = "Pluripotent"
        motion = True
        mass = _mass
        if _stochastic:
            booleans = np.array([r.randint(0, 1), r.randint(0, 1), 0, 1])
        else:
            booleans = np.array([0, 0, 0, 1])

        nuclear_radius = _nuclear_radius
        cytoplasm_radius = _cytoplasm_radius

        diff_timer = _pluri_to_diff * r.random() * 0.5
        division_timer = _pluri_div_thresh * r.random()
        death_timer = _death_threshold * r.random()

        sim_obj = Classes.StemCell(ID, location, motion, mass, nuclear_radius, cytoplasm_radius, booleans, state,
                                   diff_timer, division_timer, death_timer)

        Functions.add_object(simulation, sim_obj)
        Functions.inc_current_ID(simulation)

    # loops over all GATA6_high cells and creates a stem cell object for each one with given parameters
    for i in range(_num_GATA6):
        ID = i + _num_NANOG
        location = np.array([r.random() * _size[1], r.random() * _size[2]])
        state = "Pluripotent"
        motion = True
        mass = _mass
        if _stochastic:
            booleans = np.array([r.randint(0, 1), r.randint(0, 1), 1, 0])
        else:
            booleans = np.array([0, 0, 1, 0])

        nuclear_radius = _nuclear_radius
        cytoplasm_radius = _cytoplasm_radius

        diff_timer = _pluri_to_diff * r.random() * 0.5
        division_timer = _pluri_div_thresh * r.random()
        death_timer = _death_threshold * r.random()

        sim_obj = Classes.StemCell(ID, location, motion, mass, nuclear_radius, cytoplasm_radius, booleans, state,
                                   diff_timer, division_timer, death_timer)

        Functions.add_object(simulation, sim_obj)
        Functions.inc_current_ID(simulation)

    return simulation



