# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 03:42:14 2022

@author: umaen
"""

import numpy as np
import pandas as pd
import time as t
from random import randint
import tensorflow as tf

def generateInitialCoordinate():

    obj1xPos = randint(0,9)
    obj1yPos = randint(0,9)
    obj1Dir = randint(0,7)  # '0-N, 1-NE, 2-E, 3-SE, 4-S, 6 - W, 7 - NW

    obj2xPos = randint(0,9)
    obj2yPos = randint(0,9)
    obj2Dir = randint(0,7)
    
    return [obj1xPos,obj1yPos,obj1Dir,obj2xPos,obj2yPos,obj2Dir]

def objectNoEqual(obj1xPos,obj1yPos,obj1Dir,obj2xPos,obj2yPos,obj2Dir):
    count =0
    while ((obj1xPos != obj2xPos or obj1yPos != obj2yPos) and (count<20)):
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
            """End if of inside loop"""
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
        elif (obj1xPos == 0):      ## Issue 1 if y ==0
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
            """ End of inside if loop"""
        elif (obj1xPos == 9):  ## Issue 2 : if initial y =0 
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
            """ End of inside if loop"""
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
            elif (obj1Dir == 6):  # Issue 3: x=0
                obj1xPos = obj1xPos - 1
            elif (obj1Dir == 7):
                obj1xPos = obj1xPos - 1
                obj1yPos = obj1yPos + 1
                obj1Dir = 5
            """ End of inside if loop"""

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
            """ End of inside if loop"""
        """ End of obj 1 if loop"""
        
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
            """End of inside if loop """
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
            """End of inside if loop """
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
            """End of inside if loop """
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
            """End of inside if loop """
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

def main():
    start_time= t.time()
    i =0
    initinalCoordinate_Df = pd.DataFrame(columns=["obj1xPos","obj1yPos","obj1Dir","obj2xPos","obj2yPos","obj2Dir","objCxPos","objCyPos","outcome"])
    while i < 10000:
        obj1xPos,obj1yPos,obj1Dir,obj2xPos,obj2yPos,obj2Dir = generateInitialCoordinate()
        objCxPos,objCyPos,count = objectNoEqual(obj1xPos,obj1yPos,obj1Dir,obj2xPos,obj2yPos,obj2Dir)
        
       # if ((initinalCoordinate_Df[(initinalCoordinate_Df['obj1xPos']== obj1xPos) & (initinalCoordinate_Df['obj1yPos']== obj1yPos) & 
       #                            (initinalCoordinate_Df['obj1Dir']==obj1Dir) 
       #                             &(initinalCoordinate_Df['obj2xPos']==obj2xPos) & (initinalCoordinate_Df['obj2yPos']==obj2yPos) & 
       #                            (initinalCoordinate_Df['obj2Dir']==obj2Dir) 
       #                             & (abs(initinalCoordinate_Df['objCxPos'])==abs(objCxPos)) & 
       #                            (abs(initinalCoordinate_Df['objCyPos'])==abs(objCyPos)) & (initinalCoordinate_Df['count']==count)]).count().sum()>1):
       #     continue;
       # else:
        if (count == 20):
            outcome = 0
        else:
            outcome = 1
        initinalCoordinate_Df = initinalCoordinate_Df.append({"obj1xPos":obj1xPos,"obj1yPos":obj1yPos,"obj1Dir":obj1Dir,"obj2xPos":obj2xPos,
                                                              "obj2yPos":obj2yPos,"obj2Dir":obj2Dir,"objCxPos":objCxPos,"objCyPos":objCyPos,"outcome": outcome}, ignore_index=True)
        i = i+1
    print("Start Time",start_time)
    print("End time", t.time()- start_time)
    
    return initinalCoordinate_Df

# Time algorithm took

initinalCoordinate_Df = main() #10000 points

initinalCoordinate_Df.min()

df = initinalCoordinate_Df

df = df.apply(pd.to_numeric)

print(df.describe())
print("Number of duplicate records present: ",df.duplicated().sum())

#"Converting data frame to CSV"
#df_1 = df.to_csv('Final_data.csv')


X = df.iloc[:,:6].values
Y = df.iloc[:,-1].values

#Splitting dataset into training and testing dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential()

#Single layer ANN
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))

ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
ann.fit(X_train,Y_train,batch_size=32,epochs = 100)
test_loss, test_accuracy = ann.evaluate(X_test, Y_test)

from ann_visualizer.visualize import ann_viz;

ann_viz(ann)
