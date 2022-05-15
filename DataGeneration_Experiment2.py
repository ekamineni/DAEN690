# -*- coding: utf-8 -*-
"""
Created on Tue May 10 22:58:09 2022

@author: sushanth
"""

#Importing requried libraries
import pandas as pd
import time as t
from random import randint
from sklearn.model_selection import  train_test_split

#Function def to generate initial positions of obj 1 x, y position using random number generator
def generateInitialCoordinate():
    obj1xPos = randint(0,9)
    obj1yPos = randint(0,9)
    # 0 to 7 in directions represent: 0-N, 1-NE, 2-E, 3-SE, 4-S, 6 - W, 7 - NW
    obj1Dir = randint(0,7)
    # Returning the x,y,direction of object 1
    return [obj1xPos,obj1yPos,obj1Dir]

#Function def to find the final position of object 1 x,y after 20 loop iterations
def objectNoEqual(obj1xPos,obj1yPos,obj1Dir):
    # count value is used to keep track of "how many times the while loop has run"
    count = 0
    # Max value of count is set to 20
    while (count<20):
        count = count +1
        # Checking for obj1 direction
        if (obj1xPos > 0 and obj1xPos < 9 and obj1yPos > 0 and obj1yPos < 9):
            if (obj1Dir == 0):
                obj1yPos = obj1yPos - 1
            elif (obj1Dir == 1):
                obj1xPos = obj1xPos + 1
                obj1yPos = obj1yPos - 1
            elif (obj1Dir == 2):
                obj1xPos = obj1xPos + 1
            elif (obj1Dir == 3):
                obj1xPos = obj1xPos + 1
                obj1yPos = obj1yPos + 1
            elif (obj1Dir == 4):
                obj1yPos = obj1yPos + 1
            elif (obj1Dir == 5):
                obj1xPos = obj1xPos - 1
                obj1yPos = obj1yPos + 1
            elif (obj1Dir == 6):
                obj1xPos = obj1xPos - 1
            elif (obj1Dir == 7):
                obj1xPos = obj1xPos - 1
                obj1yPos = obj1yPos - 1
            #End if of inside loop
        elif (obj1xPos == 0 and obj1yPos == 0):
            obj1xPos == obj1xPos + 1
            obj1yPos == obj1yPos + 1
            obj1Dir = 3
        elif (obj1xPos == 9 and obj1yPos == 0):
            obj1xPos = obj1xPos - 1
            obj1yPos = obj1yPos + 1
            obj1Dir = 5
        elif (obj1xPos == 0 and obj1yPos == 9):
            obj1xPos = obj1xPos + 1
            obj1yPos = obj1yPos - 1
            obj1Dir = 1
        elif (obj1xPos == 9 and obj1yPos == 9):
            obj1xPos = obj1xPos - 1
            obj1yPos = obj1yPos - 1
            obj1Dir = 7        
        elif (obj1xPos == 0):
            if (obj1Dir == 0):
                obj1yPos = obj1yPos - 1
            elif (obj1Dir == 1):
                obj1xPos = obj1xPos + 1
                obj1yPos = obj1yPos - 1
            elif (obj1Dir == 2):
                obj1xPos = obj1xPos + 1
            elif (obj1Dir == 3):
                obj1xPos = obj1xPos + 1
                obj1yPos = obj1yPos + 1
            elif (obj1Dir == 4):
                obj1yPos = obj1yPos + 1
            elif (obj1Dir == 5):
                obj1xPos = obj1xPos + 1
                obj1yPos = obj1yPos + 1
                obj1Dir = 3
            elif (obj1Dir == 6):
                obj1xPos = obj1xPos + 1
                obj1Dir = 2
            elif (obj1Dir == 7):
                obj1xPos = obj1xPos + 1
                obj1yPos = obj1yPos - 1
                obj1Dir = 1
            #End of inside if loop
        elif (obj1xPos == 9): 
            if (obj1Dir == 0): 
                obj1yPos = obj1yPos - 1
            elif (obj1Dir == 1):
                obj1xPos = obj1xPos - 1
                obj1yPos = obj1yPos - 1
                obj1Dir = 7
            elif (obj1Dir == 2):
                obj1xPos = obj1xPos - 1
                obj1Dir = 6
            elif (obj1Dir == 3):
                obj1xPos = obj1xPos - 1
                obj1yPos = obj1yPos + 1
                obj1Dir = 5
            elif (obj1Dir == 4):
                obj1yPos = obj1yPos + 1
            elif (obj1Dir == 5):
                obj1xPos = obj1xPos - 1
                obj1yPos = obj1yPos + 1
            elif (obj1Dir == 6):
                obj1xPos = obj1xPos - 1
            elif (obj1Dir == 7):
                obj1xPos = obj1xPos - 1
                obj1yPos = obj1yPos - 1
            # End of inside if loop
        elif (obj1yPos == 0): 
            if (obj1Dir == 0):
                obj1yPos = obj1yPos + 1
                obj1Dir = 4
            elif (obj1Dir == 1):
                obj1xPos = obj1xPos + 1
                obj1yPos = obj1yPos + 1
                obj1Dir = 3
            elif (obj1Dir == 2):
                obj1xPos = obj1xPos + 1
            elif (obj1Dir == 3):
                obj1xPos = obj1xPos + 1
                obj1yPos = obj1yPos + 1
            elif (obj1Dir == 4):
                obj1yPos = obj1yPos + 1
            elif (obj1Dir == 5):
                obj1xPos = obj1xPos + 1
                obj1yPos = obj1yPos + 1
            elif (obj1Dir == 6):  
                obj1xPos = obj1xPos - 1
            elif (obj1Dir == 7):
                obj1xPos = obj1xPos - 1
                obj1yPos = obj1yPos + 1
                obj1Dir = 5
            # End of inside if loop

        elif (obj1yPos == 9): 
            if (obj1Dir == 0):
                obj1yPos = obj1yPos - 1
            elif (obj1Dir == 1):
                obj1xPos = obj1xPos + 1
                obj1yPos = obj1yPos - 1
            elif (obj1Dir == 2):
                obj1xPos = obj1xPos + 1
            elif (obj1Dir == 3):
                obj1xPos = obj1xPos + 1
                obj1yPos = obj1yPos - 1
                obj1Dir = 1
            elif (obj1Dir == 4):
                obj1yPos = obj1yPos - 1
                obj1Dir = 0
            elif (obj1Dir == 5):
                obj1xPos = obj1xPos - 1
                obj1yPos = obj1yPos - 1
                obj1Dir = 7
            elif (obj1Dir == 6):
                obj1xPos = obj1xPos - 1
            elif (obj1Dir == 7):
                obj1xPos = obj1xPos - 1
                obj1yPos = obj1yPos - 1
            # End of inside if loop
        # End of obj 1 if loop

    return[obj1xPos,obj1yPos,count]

# Main function  is used to call generateInitialCoordinate and objectNoEqual functions.
def main():
    # "NumberOfRecordsReq" object specify the number of records generated using this code.
    NumberOfRecordsReq = 100000
    start_time= t.time()
    i =0
    # Creating empty dataframe to store the initial coordinates of object 1
    initinalCoordinate_Df = pd.DataFrame(columns=["obj1xPos","obj1yPos","obj1Dir","objCxPos","objCyPos","loop"])
    #While loop to generate the above mentioned number of records
    while i < NumberOfRecordsReq:
        # Funtion call to generate the initial coordinates
        obj1xPos,obj1yPos,obj1Dir = generateInitialCoordinate()
        #Function call to find the position of obj 1 after 20 loops
        objCxPos,objCyPos,count = objectNoEqual(obj1xPos,obj1yPos,obj1Dir)    
        initinalCoordinate_Df = initinalCoordinate_Df.append({"obj1xPos":obj1xPos,"obj1yPos":obj1yPos,"obj1Dir":obj1Dir,"objCxPos":objCxPos,"objCyPos":objCyPos,"loop": count}, ignore_index=True)
        i = i+1
    # Printing the time taken to generate the records
    print("Start Time",start_time)
    print("End time", t.time()- start_time)
    # Returning the final dataframe containing the obj 1 initial and final coordinates
    return initinalCoordinate_Df

# Calling main function to generate the requried number of data records
initinalCoordinate_Df = main()
initinalCoordinate_Df.min()

# ------------ Data for Experiment 2.1 -------------

df = initinalCoordinate_Df.drop_duplicates()
df = df.apply(pd.to_numeric)
# Various type of records present in the dataframe
print(df.describe())
print("Number of total records: ", len(df))
print("Number of duplicate records present: ",df.duplicated().sum())

train,test = train_test_split(df,test_size=0.30,random_state=0)

#saving the train and test dataframe as CSV to local computer.
df.to_csv("/Users/sushanth/Desktop/DAEN 690/Mom NewData/Exp21_2/Exp21_Fulldata.csv", index = False)
test.to_csv("/Users/sushanth/Desktop/DAEN 690/Mom NewData/Exp21_2/Exp21_test.csv", index=False)


# ------------ Data for Experiment 2.2 -------------

df = initinalCoordinate_Df.drop_duplicates()
df = df.apply(pd.to_numeric)
# Various type of records present in the dataframe
print(df.describe())
print("Number of total records: ", len(df))
print("Number of duplicate records present: ",df.duplicated().sum())

train,test = train_test_split(df,test_size=0.10,random_state=0)

#saving the train and test dataframe as CSV to local computer.
train.to_csv("/Users/sushanth/Desktop/DAEN 690/Mom NewData/Exp22/Exp22_train.csv", index = False)
test.to_csv("/Users/sushanth/Desktop/DAEN 690/Mom NewData/Exp22/Exp22_test.csv", index=False)

