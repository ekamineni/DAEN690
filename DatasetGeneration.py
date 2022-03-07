import numpy as np
import pandas as pd
import random
from random import randint

# import sys
# print(sys.getrecursionlimit())
# sys.setrecursionlimit(10000)
# print(sys.getrecursionlimit())

class Particle:
    #MAX_POSITION = 9
    #MIN_POSITION = 0

    def __init__(self,x,y,dir):
        self.x = x
        self.y =y
        self.dir = dir

    def move(self):
        #### in the Matrix Scenarios
        if self.x > 0 and self.x < 9 and self.y > 0 and self.y < 9:
        # if self.x > Particle.MIN_POSITION and self.x < Particle.MAX_POSITION and \
        #         self.y > Particle.MIN_POSITION and self.y < Particle.MAX_POSITION:
            if self.dir==0:
               self.y -= 1

            elif self.dir==1:
                self.x = self.x + 1
                self.y = self.y - 1

            elif self.dir==2:
                self.x = self.x + 1

            elif self.dir==3:
                self.x = self.x + 1
                self.y = self.y + 1

            elif self.dir==4:
                self.y = self.y + 1

            elif self.dir==5:
                self.x = self.x - 1
                self.y = self.y + 1

            elif self.dir==6:
                self.x = self.x - 1

            elif self.dir==7:

                self.x = self.x - 1
                self.y = self.y - 1
        ####covering all the corners

        elif (self.x == 0 and self.y == 0):
            self.x += 1
            self.y += 1
            self.dir = 3

        elif (self.x == 9 and self.y == 0):
            self.x = self.x - 1
            self.y = self.y + 1
            self.dir = 5

        elif (self.x == 0 and self.y == 9):
            self.x = self.x + 1
            self.y = self.y - 1
            self.dir = 1

        elif (self.x == 9 and self.y == 9):
            self.x = self.x - 1
            self.y = self.y - 1
            self.dir = 7

        elif (self.x == 0):
            if (self.dir == 0):
                self.y = self.y - 1

            elif (self.dir == 1):
                self.x = self.x + 1
                self.y = self.y - 1

            elif (self.dir == 2):
                self.x = self.x + 1

            elif (self.dir == 3):
                self.x = self.x + 1
                self.y = self.y + 1

            elif (self.dir == 4):
                self.y = self.y + 1

            elif (self.dir == 5):
                self.x = self.x + 1
                self.y = self.y + 1
                self.dir = 3

            elif (self.dir == 6):
                self.x = self.x + 1
                self.dir = 2

            elif (self.dir == 7):
                self.x = self.x + 1
                self.y = self.y - 1
                self.dir = 1

            """ End of inside if loop"""

        elif (self.x == 9):
            if (self.dir == 0):
                self.y = self.y - 1

            elif (self.dir == 1):
                self.x = self.x - 1
                self.y = self.y - 1
                self.dir = 7

            elif (self.dir == 2):
                self.x = self.x - 1
                self.dir = 6

            elif (self.dir == 3):
                self.x = self.x - 1
                self.y = self.y + 1
                self.dir = 5

            elif (self.dir == 4):
                self.y = self.y + 1

            elif (self.dir == 5):
                self.x = self.x - 1
                self.y = self.y + 1

            elif (self.dir == 6):
                self.x = self.x - 1

            elif (self.dir == 7):
                self.x = self.x - 1
                self.y = self.y - 1

            """ End of inside if loop"""

        elif (self.y == 0):
            if (self.dir == 0):
                self.y = self.y + 1
                self.dir = 4

            elif (self.dir == 1):
                self.x = self.x + 1
                self.y = self.y + 1
                self.dir = 3

            elif (self.dir == 2):
                self.x = self.x + 1

            elif (self.dir == 3):
                self.x = self.x + 1
                self.y = self.y + 1

            elif (self.dir == 4):
                self.y = self.y + 1

            elif (self.dir == 5):
                self.x = self.x + 1
                self.y = self.y + 1

            elif (self.dir == 6):
                self.x = self.x - 1

            elif (self.dir == 7):
                self.x = self.x - 1
                self.y = self.y + 1
                self.dir = 5

            """ End of inside if loop"""

        elif (self.y == 9):
            if (self.dir == 0):
                self.y = self.y - 1

            elif (self.dir == 1):
                self.x = self.x + 1
                self.y = self.y - 1

            elif (self.dir == 2):
                self.x = self.x + 1

            elif (self.dir == 3):
                self.x = self.x + 1
                self.y = self.y - 1
                self.dir = 1

            elif (self.dir == 4):
                self.y = self.y - 1
                self.dir = 0

            elif (self.dir == 5):
                self.x = self.x - 1
                self.y = self.y - 1
                self.dir = 7

            elif (self.dir == 6):
                self.x = self.x - 1

            elif (self.dir == 7):
                self.x = self.x - 1
                self.y = self.y - 1

def simulation(obj1, obj2,counter=0,a=True):
    while a==True:
        if counter == 20 or (obj1.x == obj2.x and obj1.y == obj2.y):
            a=False
            return (obj1,counter)
        counter += 1
        obj1.move()
        obj2.move()

       # simulation(obj1,obj2,counter,a)

def generateInitialposition():
    obj1x = randint(0, 9)
    obj1y = randint(0, 9)
    obj1dir = randint(0, 7)  # '0-N, 1-NE, 2-E, 3-SE, 4-S, 6 - W, 7 - NW

    obj2x = randint(0, 9)
    obj2y = randint(0, 9)
    obj2dir = randint(0, 7)

    obj1 = Particle(obj1x,obj1y,obj1dir)
    obj2 = Particle(obj2x,obj2y,obj2dir)
    return obj1, obj2

def main():
    obj1,obj2 = generateInitialposition()
    #print(obj1.x, obj1.y,obj1.dir,obj1.x,obj2.y,obj2.dir)
    start_pos = [obj1.x, obj1.y,obj1.dir,obj1.x,obj2.y,obj2.dir]
    #counter = 0
    objc, counter = simulation(obj1, obj2)
    #print(objc,counter,'inside main')
    end_pos = [objc.x, objc.y, counter]

    return start_pos + end_pos

if __name__ == "__main__":
    n = 100000 #number of datapoints to be simulated
    dataset_list = []
    for i in range(n):
        dataset_points = main()
        dataset_list.append(dataset_points)

    dataset = pd.DataFrame(dataset_list,
        columns=["obj1xPos", "obj1yPos", "obj1Dir", "obj2xPos", "obj2yPos", "obj2Dir", "objCxPos", "objCyPos", "count"])

    ##select only collision points
    collision_points = dataset[dataset['count'] != 20]
    print(len(collision_points))
    dataset = dataset.append([collision_points] * 7, ignore_index=True)
    print(len(dataset))
    dataset.to_csv('./output/simulated_dataset.csv', encoding='utf-8')
