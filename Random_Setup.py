
import random as r


def Random_Bool(cells):
    new_file = open("C:\\Users\\JaTop\\stem-cell-patterning_Python\\cell_coords.txt", "w")
    for i in range(cells):
        new_file.write(str(i) + "," + str(r.random() * 1000) + "," + str(r.random() * 1000)+ "," + "Pluripotent" + "," + str(r.randint(0,1)) + "," + str(r.randint(0,1)) + "," + str(r.randint(0,1)) + "," + str(r.randint(0,1)) + "," + str(r.randint(0,1))+"\n")



Random_Bool(2000)