# Python-hiPSC-CBM
#### Description
This center-based model aims to understand the emergent patterning of human induced pluripotent
 stem cells (hiPSCs) throughout differentiation. Efficient algorithms combined with Graphical
 Processing Unit (GPU) parallelization, allow the model to simulate upwards of 200,000+ cells,
 while remaining in development friendly Python.

Developed as an offshoot of a Southeast Center for Mathematics and Biology (SCMB) seed project
 housed at Georgia Tech. [https://scmb.gatech.edu](https://scmb.gatech.edu/elena-dimitrova-clemson-melissa-kemp-gt-modeling-emergent-patterning-within-pluripotent-colonies)

##

![image](images/front_page.png)

##

### Setup Guide
Download the model either through GitHub or with the following command.
```
$ git clone https://github.com/JackToppen/Python-hiPSC-CBM.git
```
Requires Python 3.6 or later. All necessary modules are Python packages, so use pip to download them.
```
$ pip install -r requirements.txt
```

Under the ***Model*** directory, update ***paths.txt*** such the model knows where to find the template
 files and where to output the directory corresponding to each simulation.

You can specify certain parameters in the ***templates*** directory that tailor each of the simulations. 

The following command will start a text-based GUI to get the name of the simulation and the mode.
```
$ python run.py
```
Different simulation modes:
- 0 -> New simulation
- 1 -> Continue a past simulation
- 2 -> Turn past simulation images to video
- 3 -> CSVs to images/video

The name and mode can be passed at the command line...without the parentheses.
```
$ python run.py (name) (mode)
```

##

### NVIDIA CUDA support
Currently the model only supports NVIDIA CUDA. Though in the future, AMD ROCm tools will be implemented.

- Download from NVIDIA directly.
CUDA Toolkit: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
- If you do not have Microsoft Visual Studio, download that prior to the toolkit.

If you are using Anaconda, you can simply use the following.
```
$ conda install cudatoolkit
```
See [http://numba.pydata.org](http://numba.pydata.org/) for additional information about parallel processing 
and solving any issues.


##

### Issues, Problems, or Questions

Contact Jack Toppen (jtoppen3 at gatech.edu)... or not that's ok too.
Lab website: [https://kemp.gatech.edu](https://kemp.gatech.edu)

##



