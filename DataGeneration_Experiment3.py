# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 03:00:42 2022

@author: shanthan
"""

# Importing the requried libraries
import pandas as pd
import time as t
from random import randint
from random import choice
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

#Function def to generate random initial positions of  obj 1 and 2 using choice library
def generateInitialCoordinate():
    #nums represent the values from which a initial condition can be choosen from
    nums = [2,4,6,8]
    obj1xPos = choice(nums)
    obj1yPos = 0
    # 0 to 7 in directions represent: 0-N, 1-NE, 2-E, 3-SE, 4-S, 6 - W, 7 - NW
    obj1Dir = randint(0,7)
    obj2xPos = choice(nums)
    obj2yPos = 9
    obj2Dir = randint(0,7)
    # Returning the initial conditions of object 1 and 2
    return [obj1xPos,obj1yPos,obj1Dir,obj2xPos,obj2yPos,obj2Dir]


#Function def to find the collision x, y and loop value using the 6 initial conditions
def objectNoEqual(obj1xPos,obj1yPos,obj1Dir,obj2xPos,obj2yPos,obj2Dir):
    # count value is used to keep track of "how many times the while loop has run"
    count = 0
    # while loop to check if initial and final values are equal or not.
    # Max value of count is set to 20
    while ((obj1xPos != obj2xPos or obj1yPos != obj2yPos) and (count<20)):
        #Incrementing the count value everytime loop is executed
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

        elif (obj1yPos == 9): # Issue 4:
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
        
        if (obj2xPos > 0 and obj2xPos < 9 and obj2yPos > 0 and obj2yPos < 9):
            if (obj2Dir == 0):
                obj2yPos = obj2yPos - 1
            elif (obj2Dir == 1):
                obj2xPos = obj2xPos + 1
                obj2yPos = obj2yPos - 1
            elif (obj2Dir == 2):
                obj2xPos = obj2xPos + 1
            elif (obj2Dir == 3):
                obj2xPos = obj2xPos + 1
                obj2yPos = obj2yPos + 1
            elif (obj2Dir == 4):
                obj2yPos = obj2yPos + 1
            elif (obj2Dir == 5):
                obj2xPos = obj2xPos - 1
                obj2yPos = obj2yPos + 1
            elif (obj2Dir == 6):
                obj2xPos = obj2xPos - 1
            elif (obj2Dir == 7):
                obj2xPos = obj2xPos - 1
                obj2yPos = obj2yPos - 1
            # End of inside if loop 
        elif (obj2xPos == 0 and obj2yPos == 0):
            obj2xPos = obj2xPos + 1
            obj2yPos = obj2yPos + 1
            obj2Dir = 3

        elif (obj2xPos == 9 and obj2yPos == 0):
            obj2xPos = obj2xPos - 1
            obj2yPos = obj2yPos + 1
            obj2Dir = 5

        elif (obj2xPos == 0 and obj2yPos == 9):
            obj2xPos = obj2xPos + 1
            obj2yPos = obj2yPos - 1
            obj2Dir = 1

        elif (obj2xPos == 0 and obj2yPos == 9):
            obj2xPos = obj2xPos - 1
            obj2yPos = obj2yPos - 1
            obj2Dir = 7

        elif (obj2xPos == 0):   #Issue 5
            if (obj2Dir == 0):
                obj2yPos = obj2yPos - 1
            elif (obj2Dir == 1):
                obj2xPos = obj2xPos + 1
                obj2yPos = obj2yPos - 1
            elif (obj2Dir == 2):
                obj2xPos = obj2xPos + 1
            elif (obj2Dir == 3):
                obj2xPos = obj2xPos + 1
                obj2yPos = obj2yPos + 1
            elif (obj2Dir == 4):
                obj2yPos = obj2yPos + 1
            elif (obj2Dir == 5):
                obj2xPos = obj2xPos + 1
                obj2yPos = obj2yPos + 1
                obj2Dir = 3
            elif (obj2Dir == 6):
                obj2xPos = obj2xPos + 1
                obj2Dir = 2
            elif (obj2Dir == 7):
                obj2xPos = obj2xPos + 1
                obj2yPos = obj2yPos - 1
                obj2Dir = 1
            #End of inside if loop #
        elif (obj2xPos == 9):
            if (obj2Dir == 0):
                obj2yPos = obj2yPos - 1
            elif (obj2Dir == 1):
                obj2xPos = obj2xPos - 1
                obj2yPos = obj2yPos - 1
                obj2Dir = 7
            elif (obj2Dir == 2):
                obj2xPos = obj2xPos - 1
                obj2Dir = 6
            elif (obj2Dir == 3):
                obj2xPos = obj2xPos - 1
                obj2yPos = obj2yPos + 1
                obj2Dir = 5
            elif (obj2Dir == 4):
                obj2yPos = obj2yPos + 1
            elif (obj2Dir == 5):
                obj2xPos = obj2xPos - 1
                obj2yPos = obj2yPos + 1
            elif (obj2Dir == 6):
                obj2xPos = obj2xPos - 1
            elif (obj2Dir == 7):
                obj2xPos = obj2xPos - 1
                obj2yPos = obj2yPos - 1
            #End of inside if loop #
        elif (obj2yPos == 0):
            if (obj2Dir == 0):
                obj2yPos = obj2yPos + 1
                obj2Dir = 4
            elif (obj2Dir == 1):
                obj2xPos = obj2xPos + 1
                obj2yPos = obj2yPos + 1
                obj2Dir = 3
            elif (obj2Dir == 2):
                obj2xPos = obj2xPos + 1
            elif (obj2Dir == 3):
                obj2xPos = obj2xPos + 1
                obj2yPos = obj2yPos + 1
            elif (obj2Dir == 4):
                obj2yPos = obj2yPos + 1
            elif (obj2Dir == 5):
                obj2xPos = obj2xPos + 1
                obj2yPos = obj2yPos + 1
            elif (obj2Dir == 6):
                obj2xPos = obj2xPos - 1
            elif (obj2Dir == 7):
                obj2xPos = obj2xPos - 1
                obj2yPos = obj2yPos + 1
                obj2Dir = 5
            #End of inside if loop #
        elif (obj2yPos == 9):
            if (obj2Dir == 0):
                obj2yPos = obj2yPos - 1
            elif (obj2Dir == 1):
                obj2xPos = obj2xPos + 1
                obj2yPos = obj2yPos - 1
            elif (obj2Dir == 2):
                obj2xPos = obj2xPos + 1
            elif (obj2Dir == 3):
                obj2xPos = obj2xPos + 1
                obj2yPos = obj2yPos - 1
                obj2Dir = 1
            elif (obj2Dir == 4):
                obj2yPos = obj2yPos - 1
                obj2Dir = 0
            elif (obj2Dir == 5):
                obj2xPos = obj2xPos - 1
                obj2yPos = obj2yPos - 1
                obj2Dir = 7
            elif (obj2Dir == 6):
                obj2xPos = obj2xPos - 1
            elif (obj2Dir == 7):
                obj2xPos = obj2xPos - 1
                obj2yPos = obj2yPos - 1
    return[obj1xPos,obj1yPos,count]

# Main function  is used to call generateInitialCoordinate and objectNoEqual functions.
def main():
    # "NumberOfRecordsReq" object specify the number of records generated using this code.
    NumberOfRecordsReq = 100000
    #Start time to check for time taken to generate the records.
    start_time= t.time()
    i =0
    # Creating empty dataframe to store the initial coordinates of object 1 and 2
    initinalCoordinate_Df = pd.DataFrame(columns=["obj1xPos","obj1yPos","obj1Dir","obj2xPos","obj2yPos","obj2Dir","objCxPos","objCyPos","loop"])
    #While loop to generate the above mentioned number of records
    while i < NumberOfRecordsReq:
        # Funtion call to generate the initial coordinates
        obj1xPos,obj1yPos,obj1Dir,obj2xPos,obj2yPos,obj2Dir = generateInitialCoordinate()
        # Function call to see if initial and final coordinates are same or not
        objCxPos,objCyPos,count = objectNoEqual(obj1xPos,obj1yPos,obj1Dir,obj2xPos,obj2yPos,obj2Dir)
        if(count==20):
            loop = 0
        else:
            loop = 1
        initinalCoordinate_Df = initinalCoordinate_Df.append({"obj1xPos":obj1xPos,"obj1yPos":obj1yPos,"obj1Dir":obj1Dir,"obj2xPos":obj2xPos,
                                                              "obj2yPos":obj2yPos,"obj2Dir":obj2Dir,"objCxPos":objCxPos,"objCyPos":objCyPos,"loop": loop}, ignore_index=True)
        i = i+1
    # Printing the time taken to generate the records
    print("Start Time",start_time)
    print("End time", t.time()- start_time)
    # Returning the final list of records which contain initial, collision coordinates and loop
    return initinalCoordinate_Df

# Calling main function to generate the requried number of data records
initinalCoordinate_Df = main()
initinalCoordinate_Df.min()

# ------------ Data for Experiment 3.1 -------------

df = initinalCoordinate_Df.drop_duplicates()
df = df.apply(pd.to_numeric)

X = df[["obj1xPos","obj1yPos","obj1Dir","obj2xPos","obj2yPos","obj2Dir"]]
y = df["loop"]

over = RandomOverSampler()
under = RandomUnderSampler()

# first performing oversampling to minority class
X_over, y_over = over.fit_resample(X, y)
print(f"Oversampled: {Counter(y_over)}")

# now to comine under sampling 
X_combined_sampling, y_combined_sampling = under.fit_resample(X_over, y_over)
print(f"Combined Random Sampling: {Counter(y_combined_sampling)}")

df =pd.concat([X_combined_sampling,y_combined_sampling],axis = 1,ignore_index=True)
df.columns = ["obj1xPos","obj1yPos","obj1Dir","obj2xPos","obj2yPos","obj2Dir","loop"]
df = df.append([df] * 5, ignore_index=True)

# Various type of records present in the dataframe
print(df.describe())
print("Number of total records: ", len(df))
print("Number of collision cases: ",len(df[df['loop'] == 1]))
print("Number of non collision cases: ",len(df[df['loop'] != 1]))
print("Number of dup in collision cases: ",df[df['loop'] == 1].duplicated().sum())
print("Number of dup in non collision cases: ",df[df['loop'] != 1].duplicated().sum())
print("Number of duplicate records present: ",df.duplicated().sum())

train,test = train_test_split(df,test_size=0.25,random_state=0)
print("Number of total records: ", len(train))
print("Number of collision cases: ",len(train[train['loop'] == 1]))
print("Number of non collision cases: ",len(train[train['loop'] != 1]))

print("Number of total records: ", len(test))
print("Number of collision cases: ",len(test[test['loop'] == 1]))
print("Number of non collision cases: ",len(test[test['loop'] != 1]))

#saving the train and test dataframe as CSV to local computer.
df.to_csv("C:/Users/shant/Documents/GMU/FInal Sem/Final Project/Final Version Files/Exp31_data.csv", index = False)
train.to_csv("C:/Users/shant/Documents/GMU/FInal Sem/Final Project/Final Version Files/Exp31_train.csv", index=False)
test.to_csv("C:/Users/shant/Documents/GMU/FInal Sem/Final Project/Final Version Files/Exp31_test.csv", index=False)


# ------------ Data for experiment 3.2 -------------
#Removing the duplicates from initial dataframe
df = initinalCoordinate_Df.drop_duplicates()
df = df.apply(pd.to_numeric)

# Various type of records present in the dataframe
print(df.describe())
print("Number of total records: ", len(df))
print("Number of collision cases: ",len(df[df['loop'] == 1]))
print("Number of non collision cases: ",len(df[df['loop'] != 1]))
print("Number of dup in collision cases: ",df[df['loop'] == 1].duplicated().sum())
print("Number of dup in non collision cases: ",df[df['loop'] != 1].duplicated().sum())
print("Number of duplicate records present: ",df.duplicated().sum())

# Saving the dataframe as a CSV file
df.to_csv("C:/Users/shant/Documents/GMU/FInal Sem/Final Project/Final Version Files/Exp32_Fulldata.csv", index=False)

#Spliting the dataset into train and test with 75:25% ratio
train,test = train_test_split(df,test_size=0.25,random_state=0)
print("Number of total records: ", len(train))
print("Number of collision cases: ",len(train[train['loop'] == 1]))
print("Number of non collision cases: ",len(train[train['loop'] != 1]))

print("Number of total records: ", len(test))
print("Number of collision cases: ",len(test[test['loop'] == 1]))
print("Number of non collision cases: ",len(test[test['loop'] != 1]))

#saving the train and test dataframe as CSV to local computer.
train.to_csv("C:/Users/shant/Documents/GMU/FInal Sem/Final Project/Final Version Files/Exp32_train.csv", index=False)
test.to_csv("C:/Users/shant/Documents/GMU/FInal Sem/Final Project/Final Version Files/Exp32_test.csv", index=False)
