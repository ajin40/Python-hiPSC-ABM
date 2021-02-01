from tkinter.filedialog import askopenfilename
import numpy as np
from matplotlib import pyplot as plt
import ripser
import os
import tkinter

import input


# only run TDA pipeline if being run directly
if __name__ == "__main__":
    # -------------------- options ---------------------
    xy_max = 200    # max size of the axes
    H0_color = "cornflowerblue"    # color of 0-dim points from matplotlib colors
    H1_color = "forestgreen"    # color of 1-dim points from matplotlib colors
    image_name = "figure.png"    # name of the persistence diagram
    dpi = 400    # resolution of persistence diagram
    # --------------------------------------------------

    # get the separator and the output directory
    separator = os.path.sep
    output_dir = input.output_dir(separator)

    # suppress tkinter GUI, put file explorer on top
    root = tkinter.Tk()
    root.attributes('-topmost', True)
    root.withdraw()

    # open mini file explorer to get the
    file_path = askopenfilename(filetypes=[("TDA files", "*.csv")])

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

    # save the outputs for 0-dimensional analysis
    file_path = output_path + "0dim_" + file_name
    np.savetxt(file_path, diagrams[0], delimiter=",")

    # save the outputs for 1-dimensional analysis
    file_path = output_path + "1dim_" + file_name
    np.savetxt(file_path, diagrams[1], delimiter=",")

    # make persistence diagram figure (5 inches x 4.75 inches) and add one Axes object
    fig = plt.figure(figsize=(5, 4.75))   # make new figure
    ax = plt.axes()  # add new axes

    # add the following labels and legend
    ax.set_title("Persistence Diagram")
    ax.set_xlabel("birth")
    ax.set_ylabel("death")

    # sizing of the plot
    xy_max = 200
    ax.set_aspect(1)    # set aspect ratio to 1:1
    ax.set_xlim(-5, xy_max)    # set x limits (-5 to show 0-dim points)
    ax.set_ylim(0, xy_max)    # set y limits

    # draw diagonal line
    ax.plot([0, xy_max], [0, xy_max], color="k", linestyle="--")

    # add the zero dimensional and one dimensional data as points
    ax.scatter(diagrams[0][:, 0], diagrams[0][:, 1], c=H0_color, marker=".", label="$H_0$")
    ax.scatter(diagrams[1][:, 0], diagrams[1][:, 1], c=H1_color, marker=".", label="$H_1$")

    # save the figure as a png
    ax.legend(loc='lower right')  # legend lower right
    file_path = output_path + image_name
    fig.savefig(file_path, dpi=dpi)
