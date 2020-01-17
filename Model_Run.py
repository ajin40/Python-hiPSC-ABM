import Model_Setup
from Count_Directory import newDirect
import os


# absolutized version of parent dictionary path
# path = os.path.abspath(os.pardir)
path = "C:\\Python27\\MEGA-ARRAY"
# length of model runtime
Run_Time = 15.0
# returns the number name of current model
Model_ID = newDirect(path)
# runs Model_Setup
Model_Setup.main(Model_ID, Run_Time, path)
