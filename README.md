# Stem Cell Patterning Python
#### Agent-based model of pluripotent colonies patterning through differentiation.

Developed from a seed project with the Southeast Center for Mathematics and Biology at Georgia Tech.

Project Description: [https://scmb.gatech.edu](https://scmb.gatech.edu/elena-dimitrova-clemson-melissa-kemp-gt-modeling-emergent-patterning-within-pluripotent-colonies)

Lab website: [https://kemp.gatech.edu](https://kemp.gatech.edu)

##
![Picture2](https://user-images.githubusercontent.com/57497258/80270182-8f35f980-867b-11ea-80c4-b954540a8fcd.jpg)
Experimental immunofluorescence results from Eunbi Park

### Download and Libraries:
Use "Clone or download" to either open the model in GitHub Desktop or download as a ZIP.

This command will achieve the above too.
```
$ git clone https://github.com/JackToppen/stem-cell-patterning_Python.git
```
The model requires Python 3.6 or later. Install any necessary modules. 
```
$ pip install -r requirements.txt
```
##

### Quick Setup Guide

- Place the "Setup_files" directory wherever you so choose. This contains all of the simulation 
templates txt files. Each template corresponds to a separate simulation run by the model. If you put
multiple files in the directory, the model will run them successively.

- Indicate the path of the "Setup_files" directory in "Locations.txt". "Setup_files" can be renamed and relocated as 
long as this is updated in "Locations.txt". Additionally, update the path to which the model will output data as it runs

- Edit the template files as you wish. Recommended values will be indicated with "Ex."

- Run the model. See Documentation.docx for further explanations of classes, functions, and parameters.
```
$ python Run.py
```

##

### Parallel GPU Processing
Currently the model only supports NVIDIA CUDA. Though in the future, AMD ROCm tools will be implemented.

- Download from NVIDIA directly.
CUDA Toolkit: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

If you are using Anaconda, you can simply use the following.
```
$ conda install cudatoolkit
```
See [http://numba.pydata.org](http://numba.pydata.org/) for additional information about parallel processing 
and solving any issues.


##

### Issues, Problems, or Questions

Contact Jack Toppen (jtoppen3 at gatech.edu)... or not that's ok too.

##



