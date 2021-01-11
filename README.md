# Python-hiPSC-CBM
#### Description
This center-based model aims to understand the emergent patterning of human induced pluripotent
 stem cells (hiPSCs) as they differentiate. Multiple modeling schemes such as morphogen diffusion
 and collision-handling are employed for biological accuracy. Graphical Processing Unit (GPU)
 parallelization through the CUDA platform allows for the model to simulate 500,000+ cells.
 
Developed as part of a Southeast Center for Mathematics and Biology (SCMB) seed project
 located at Georgia Tech. [https://scmb.gatech.edu](https://scmb.gatech.edu/elena-dimitrova-clemson-melissa-kemp-gt-modeling-emergent-patterning-within-pluripotent-colonies)

##

![image](images/front_page.png)

##

### Setup guide
The model ***requires*** Python 3.6-3.8. The latest Anaconda distribution should already
include most of the necessary dependencies.

Download the model either through GitHub (.zip) or with the Git command below.
```
$ git clone https://github.com/JackToppen/Python-hiPSC-CBM.git
```

The downloaded folder (Python-hiPSC-CBM) will contain the ***Model*** directory (where
all the code is) and some additional files including documentation and a requirements.txt
file. 

You can move this downloaded folder to wherever you like and even rename it. The following
command can be used to automatically install any required dependencies, but just be sure
to change the path to this downloaded folder that contains the requirements.txt file.

```
$ pip install -r requirements.txt
```

Under the ***Model*** directory, update ***paths.txt*** such the model knows where to output 
the folder corresponding to each simulation. This directory should exist prior to running a
simulation.

You can specify certain parameters using the .txt files in the ***templates*** directory. Additional
parameters may be added in ***parameters.py***, which are held by the Simulation object.

The following command will start a text-based GUI to start a simulation by choosing a
name (whatever you want) and the mode (described below).
```
$ python run.py
```
Different simulation modes:
- 0: New simulation
- 1: Continue a past simulation
- 2: Turn previous simulation images to a video
- 3: Zip a previous simulation
- 4: Extract previous simulation zip in output directory
- 5: Calculate persistent homology values


Additionally, the name and mode can be passed at the command line by using options
 (without the parentheses). This avoids the text-based GUI altogether.
```
$ python run.py -n (name) -m (mode)
```

##

### NVIDIA CUDA support
The model has optional GPU parallelization for some elements of the code. Currently its only
available for NVIDIA CUDA though AMD ROCm support may come in the future. Download NVIDIA's
CUDA toolkit to support this feature.

If you do not have Microsoft Visual Studio, download that prior to the toolkit. 

- Download the toolkit from NVIDIA directly: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

If you are using Anaconda, simply use conda.
```
$ conda install cudatoolkit
```

##

### Issues, problems, or questions

Contact Jack Toppen (jtoppen3 at gatech.edu)...or not that's ok too.

##