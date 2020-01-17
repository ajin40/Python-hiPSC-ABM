
import random as r


def Random_Bool(cells):
    new_file = open("C:\\Python27\\MEGA-ARRAY\\cell_coords.txt", "w")
    for i in range(cells + 1):
        new_file.write(str(i) + "," + str(r.random() * 1000) + "," + str(r.random() * 1000)+ "," + "Pluripotent" + "," + str(r.randint(0,1)) + "," + str(r.randint(0,1)) + "," + str(r.randint(0,1)) + "," + str(r.randint(0,1)) + "," + str(r.randint(0,1))+"\n")



Random_Bool(1000)




