import os

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

