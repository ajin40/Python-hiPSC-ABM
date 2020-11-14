import numpy as np


class Simulation:
    def __init__(self):
        self.number_cells = 4
        self.field = 3

    def cell_arrays(self, *args):
        """ creates the Simulation instance arrays that
            correspond to particular cell values
        """
        # go through all arguments passed
        for array_params in args:
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

    def initials(self, array_name, func):
        new_array = np.empty_like(self.__dict__[array_name], shape=(self.number_cells, 3))
        self.__dict__[array_name] = np.concatenate((self.__dict__[array_name], new_array))

        for i in range(self.number_cells):
            self.__dict__[array_name][i] = func()
