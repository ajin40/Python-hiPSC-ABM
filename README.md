# Python-hiPSC-ABM
#### Description
This agent-based model (ABM) aims to understand the emergent patterning of human induced pluripotent
 stem cells (hiPSCs) as they differentiate. Multiple modeling schemes such as morphogen diffusion
 and collision-handling are employed for biological accuracy. The ABM uses CPU parallelization
 alongside NVIDIA's CUDA platform for GPUs, allowing for simulations with 500,000+ cells.
 
This work is part of a [Southeast Center for Mathematics and Biology](https://scmb.gatech.edu/elena-dimitrova-clemson-melissa-kemp-gt-modeling-emergent-patterning-within-pluripotent-colonies)
 (SCMB) seed project housed at Georgia Tech. 

##

![front_page](https://user-images.githubusercontent.com/57497258/119276216-a5e4b200-bbe7-11eb-9468-1e67effbb12d.png)

##

### Installation
This ABM ***requires*** Python 3.6-3.8 for full functionality. A CUDA compatible
GPU is necessary for enabling the optional parallelization of various simulation methods. More
information on this can be found at the bottom.

Download the model directly from GitHub or with the Git command shown below.
```
git clone https://github.com/JackToppen/Python-hiPSC-ABM.git
```

The downloaded folder (Python-hiPSC-ABM) will contain an ***ABM*** folder (where
all the model code is) and some additional files including a ***requirements.txt***
file. With the following command, this file can be used to install
all required dependencies.

```
pip install -r requirements.txt
```

##
### Running a simulation
Under the ***ABM*** directory, update ***paths.yaml*** such the model knows where to put
outputs from each simulation. Further, you can specify simulation parameters using the 
YAML files in the ***templates*** directory.

The Simulation class, found in simulation.py, will run as a standalone ABM, but it's 
meant to be inherited to provide any child classes with necessary/helpful methods
for running a simulation. In particular, Simulation's start() class-method will launch
the ABM, in addition to any subclass of Simulation. You can configure run.py to call the
particular start() method and run it as follows.
```
python run.py
```
The text-based UI will prompt for the ***name*** identifier for the simulation and
corresponding ***mode*** as described below.
- 0: New simulation
- 1: Continue a previous simulation
- 2: Turn a previous simulation's images to a video
- 3: Archive (.zip) a previous simulation's outputs

To avoid the text-based UI, the name and mode can be passed at the command line by using flags
 (without the parentheses).
```
python run.py -n (name) -m (mode)
```

##

### NVIDIA CUDA support
In order to use the code associated with CUDA GPU parallelization, you'll need a CUDA
compatible GPU and NVIDIA's CUDA toolkit. If you don't have the toolkit installed, make
sure you have Microsoft Visual Studio prior to installation.

Download the toolkit directly from NVIDIA [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
or with the conda command show below.
```
conda install cudatoolkit
```


##