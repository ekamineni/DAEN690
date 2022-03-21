#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 02:07:46 2022

@author: sushanth
"""


import numpy as np
import pandas as pd
import time as t
from random import randint
from random import choice
import tensorflow as tf
from sklearn.metrics import confusion_matrix


def generateInitialCoordinate():

    obj1xPos = randint(0,9)
    obj1yPos = randint(0,9)
    obj1Dir = randint(0,7)  # '0-N, 1-NE, 2-E, 3-SE, 4-S, 6 - W, 7 - NW
   
    return [obj1xPos,obj1yPos,obj1Dir]

def objectNoEqual(obj1xPos,obj1yPos,obj1Dir):
    count = 0
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
            #End of inside if loop
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
            elif (obj1Dir == 6):  # Issue 3: x=0
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

    return[obj1xPos,obj1yPos,count]

def main():
    start_time= t.time()
    i =0
    initinalCoordinate_Df = pd.DataFrame(columns=["obj1xPos","obj1yPos","obj1Dir","objCxPos","objCyPos","count"])
    while i < 100000:
        obj1xPos,obj1yPos,obj1Dir = generateInitialCoordinate()
        objCxPos,objCyPos,count = objectNoEqual(obj1xPos,obj1yPos,obj1Dir)    
        if (count == 20):
            outcome = 0
        else:
            outcome = 1
        initinalCoordinate_Df = initinalCoordinate_Df.append({"obj1xPos":obj1xPos,"obj1yPos":obj1yPos,"obj1Dir":obj1Dir,"objCxPos":objCxPos,"objCyPos":objCyPos,"count": outcome}, ignore_index=True)
        i = i+1
    print("Start Time",start_time)
    print("End time", t.time()- start_time)
    
    return initinalCoordinate_Df

# Time algorithm took

initinalCoordinate_Df = main() #10000 points
initinalCoordinate_Df.to_csv('F_data_new.csv',index=None)




#initinalCoordinate_Df = pd.read_csv(r'C:\Users\shant\Downloads\Concat_dataset.csv')

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

X = df.iloc[:,:3].values
Y = df.iloc[:,4:5].values


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

ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
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


"""

print(tf.__version__)

import numpy as np
df_new =np.concatenate((X_test_dup,Y_test,Y_pred),axis=1)

df_1 = pd.DataFrame(X_test_dup)

df_2 = pd.DataFrame(Y_test)

df_3 = pd.DataFrame(Y_pred)

frames = [df_1, df_2, df_3]
result = pd.concat(frames)

result.to_csv('F_data_new2.csv',index=None)

"""