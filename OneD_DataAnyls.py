#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 02:05:17 2022

@author: sushanth
"""


import pandas as pd
import time as t
from random import randint
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

def generateInitialCoordinate():

    obj1xPos = randint(0,9)
    #obj1yPos = randint(0,9)
    obj1Dir = randint(0,1)  # '0-L, 1-R

    #obj2xPos = randint(0,9)
    #obj2yPos = randint(0,9)
    #obj2Dir = randint(0,7)
    
    return [obj1xPos,obj1Dir]

def objectNoEqual(obj1xPos,obj1Dir):
    count =1
    bounce = 0
    while (count<=6):
        count = count +1
        bounce
        if (obj1xPos > 0 and obj1xPos < 9):
            if (obj1Dir == 0):
                obj1xPos = obj1xPos - 1
            elif (obj1Dir == 1):
                obj1xPos = obj1xPos + 1
        elif (obj1xPos == 0):
            if (obj1Dir == 0):
                bounce = 1
                obj1Dir = 1
                obj1xPos = obj1xPos + 1
            elif (obj1Dir == 1):
                obj1xPos = obj1xPos + 1
        elif (obj1xPos == 9):
            if (obj1Dir == 0): 
                obj1xPos = obj1xPos - 1
            elif (obj1Dir == 1):
                obj1Dir = 0
                bounce = 1
                obj1xPos = obj1xPos - 1

    return[obj1xPos,bounce]

def main():
    start_time= t.time()
    i =0
    initinalCoordinate_Df = pd.DataFrame(columns=["obj1xPos","obj1Dir","objCxPos","Bounce"])
    while i < 10000:
        obj1xPos,obj1Dir = generateInitialCoordinate()
        objCxPos,bounce = objectNoEqual(obj1xPos,obj1Dir)
        initinalCoordinate_Df = initinalCoordinate_Df.append({"obj1xPos":obj1xPos,"obj1Dir":obj1Dir,"objCxPos":objCxPos,"Bounce": bounce}, ignore_index=True)
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
print("Number of total records: ", len(df))
#print("Number of collision cases: ",len(df[df['count1'] != 20]))
#print("Number of non collision cases: ",len(df[df['count1'] == 20]))
#print("Number of dup in collision cases: ",df[df['count1'] != 20].duplicated().sum())
#print("Number of dup in non collision cases: ",df[df['count1'] == 20].duplicated().sum())
print("Number of duplicate records present: ",df.duplicated().sum())

#"Converting data frame to CSV"
#df_1 = df.to_csv('Final_data.csv')

X = df.iloc[:,:2].values
Y = df.iloc[:,-2].values



#Y = Y.astype('object')

#Splitting dataset into training and testing dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0,stratify=Y)
X_test_dup = X_test
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential()

#Single layer ANN
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))

ann.add(tf.keras.layers.Dense(units=1,activation="sigmodi"))
ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
ann.fit(X_train,Y_train,batch_size=32,epochs = 20)
test_loss, test_accuracy = ann.evaluate(X_test, Y_test)

Y_pred = ann.predict(X_test)
Y_pred

Y_pred = np.round(Y_pred,0)
Y_pred

cm = confusion_matrix(Y_test, Y_pred)
cm
 
TruePos0 = cm[0,0]
TrueNeg1 = cm[0,1]
FalsePos0 = cm[1,0]
FalseNeg1 = cm[1,1]

Accuracy = (TruePos0+TrueNeg1)/(TruePos0+TrueNeg1+FalsePos0+FalseNeg1)
print('Accuracy: %f',Accuracy)
Recall = TruePos0/(FalseNeg1+TruePos0)
print('Recall: %f',Recall)
Precision = TruePos0/(FalsePos0+TruePos0)
print('Precision: %f',Precision)
