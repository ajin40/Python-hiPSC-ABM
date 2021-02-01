from tkinter.filedialog import askopenfilename
import numpy as np
import persim
from matplotlib import pyplot as plt
import ripser
import os
import tkinter

import input


# only run TDA pipeline if being run directly
if __name__ == "__main__":
    # get the separator and the output directory
    separator = os.path.sep
    output_dir = input.output_dir(separator)

    # suppress tkinter GUI
    root = tkinter.Tk()
    root.withdraw()
    root.lift()

    # open mini file explorer to get the
    file_path = askopenfilename()

    # make sure ends with ".csv"
    if not file_path.endswith(".csv"):
        raise Exception("TDA input file should be a CSV")

    # get the name of the file and make a directory with the name of the file
    file_name = os.path.basename(file_path)
    name_no_ext = os.path.splitext(file_name)[0]
    output_path = output_dir + "TDA_results_" + name_no_ext + separator
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # get CSV data as an array
    data = np.genfromtxt(file_path, delimiter=",")

    # calculate the persistent homology values
    diagrams = ripser.ripser(data, maxdim=1)['dgms']
    persim.plot_diagrams(diagrams, xy_range=[0, 300, 0, 300])
    plt.savefig(output_path + "figure.png")

    # save the outputs for 0-dimensional analysis
    file_path = output_path + "0-dim_" + file_name
    np.savetxt(file_path, diagrams[0], delimiter=",")

    # save the outputs for 1-dimensional analysis
    file_path = output_path + "1-dim_" + file_name
    np.savetxt(file_path, diagrams[1], delimiter=",")
