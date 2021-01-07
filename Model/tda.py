import numpy as np
import ripser
import os


def calculate_persistence(paths):
    """ Get the persistent homology values and save these values
        to a CSV file.
    """
    # list all of the files in the directory
    print(*os.listdir(paths.tda), sep='\t')

    # get the name of the TDA file of interest
    file_name = input("What is the name of the CSV with TDA outputs? ")

    # make sure ends with ".csv"
    if not file_name.endswith(".csv"):
        file_name += ".csv"

    # get the full path and read the data into a 2D array
    file_path = paths.tda + file_name
    data = np.genfromtxt(file_path, delimiter=",")

    # calculate the persistent homology values
    diagrams = ripser.ripser(data, maxdim=1)['dgms']

    # make directory for persistent homology outputs
    dir_path = paths.tda + "Persistence_values" + paths.separator
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    # save the outputs for 0-dimensional analysis
    output_path = dir_path + "0-dim_" + file_name
    np.savetxt(output_path, diagrams[0], delimiter=",")

    # save the outputs for 1-dimensional analysis
    output_path = dir_path + "1-dim_" + file_name
    np.savetxt(output_path, diagrams[1], delimiter=",")
