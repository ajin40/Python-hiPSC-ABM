import numpy as np


class Simulation:
    def __init__(self):
        self.field = 3
        self.size = np.array([1000, 1000, 0])
        self.cell_array_names = []
        self.death_thresh = 4
        self.pluri_to_diff = 4
        self.pluri_div_thresh = 4
        self.fds_thresh = 4

    def cell_types(self, *args):
        """ go through the cell types adding them to the
            simulation
        """
        self.holder = dict()
        self.number_cells = 0
        for cell_type in args:
            begin = self.number_cells
            self.number_cells += cell_type[1]
            end = self.number_cells
            self.holder[cell_type[0]] = (begin, end)

    def cell_arrays(self, *args):
        """ creates the Simulation instance arrays that
            correspond to particular cell values
        """
        # go through all arguments passed
        for array_params in args:
            self.cell_array_names.append(array_params[0])

            # get the length of the tuple
            length = len(array_params)

            # if the tuple passed is of length two, make a 1-dimensional array
            if length == 2:
                size = 0

            # if the tuple
            elif length == 3:
                size = (0, array_params[2])

            # raise error if otherwise
            else:
                raise Exception("tuples should have length 2 or 3")

            # create an instance variable for the cell array with the specified size and type
            self.__dict__[array_params[0]] = np.empty(size, dtype=array_params[1])

    def initials(self, cell_type, array_name, func):
        """ given a lambda function for the initial values
            of a cell array this updates that accordingly
        """
        if cell_type == "all":
            # get the cell array
            cell_array = self.__dict__[array_name]

            shape = list(cell_array.shape)
            shape[0] = self.number_cells
            array_type = cell_array.dtype
            empty_array = np.empty(shape, dtype=array_type)
            self.__dict__[array_name] = np.concatenate((cell_array, empty_array))

            for i in range(self.number_cells):
                self.__dict__[array_name][i] = func()

        else:
            begin = self.holder[cell_type][0]
            end = self.holder[cell_type][1]

            for i in range(begin, end):
                self.__dict__[array_name][i] = func()
