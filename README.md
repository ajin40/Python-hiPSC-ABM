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

### Setup guide
Download the model either through GitHub or with the following command.
```
$ git clone https://github.com/JackToppen/Python-hiPSC-CBM.git
```
Requires Python 3.6-3.8. All necessary modules are Python packages, so use pip to download them.
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
- 4 -> Zip a past simulation


The name and mode can be passed at the command line by using options...without the parentheses.
```
$ python run.py -n (name) -m (mode)
```

##

### NVIDIA CUDA support
The model has optional GPU parallelization for some elements of the code. Currently its only
available for NVIDIA CUDA though AMD ROCm support will come in the future. Download NVIDIA's CUDA 
toolkit so that Numba library can create CUDA kernels.

If you do not have Microsoft Visual Studio, download that prior to the toolkit. 

- Download from NVIDIA directly.
CUDA Toolkit: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

If you are using Anaconda, simply use conda.
```
$ conda install cudatoolkit
```

##

### Issues, problems, or questions

Contact Jack Toppen (jtoppen3 at gatech.edu)...or not that's ok too.

##
