# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 12:03:30 2022

@author: Shanthan
"""

#Importing the requried libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# ----------------------Exp 3.1 ----------------------

#Reading test and train data
train = pd.read_csv("https://raw.githubusercontent.com/ekamineni/DAEN690/main/Datarepo/Exp31_train.csv")
test = pd.read_csv("https://raw.githubusercontent.com/ekamineni/DAEN690/main/Datarepo/Exp31_test.csv")

#Copying only obj 1 -> x, y, direction and obj 2-> x, y, direction fields into X_train and X_test
#Copying only collision x, y and loop values into Y_train and Y_test
X_train = train.iloc[:,:6].values
Y_train = train.iloc[:,-1].values
X_test = test.iloc[:,:6].values
Y_test = test.iloc[:,-1].values

#Scaling the x values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Creating a model named as ann
ann = tf.keras.models.Sequential()

#Five layer ANN with activation as relu and number of nodes as 64
ann.add(tf.keras.layers.Dense(units=64,activation="relu"))
ann.add(tf.keras.layers.Dense(units=64,activation="relu"))
ann.add(tf.keras.layers.Dense(units=64,activation="relu"))
ann.add(tf.keras.layers.Dense(units=64,activation="relu"))
ann.add(tf.keras.layers.Dense(units=64,activation="relu"))

#Single output layer with activation as sigmoid
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

#Training the model with epochs as 100
ann.fit(X_train,Y_train,batch_size=32,epochs = 100)

#Evaluating the accuracy of test data
test_loss, test_accuracy = ann.evaluate(X_test, Y_test)

Y_pred = ann.predict(X_test)
Y_pred 

Y_pred = np.round(Y_pred,0)
Y_pred
pred_test_df = pd.DataFrame(Y_pred, columns=['Loop'])
pred_test_df

cm = confusion_matrix(Y_test, Y_pred)
cm

TruePos0 = cm[0,0]
TrueNeg1 = cm[1,1]
FalsePos0 = cm[1,0]
FalseNeg1 = cm[0,1]

#Results of the model
Accuracy = (TruePos0+TrueNeg1)/(TruePos0+TrueNeg1+FalsePos0+FalseNeg1)
print('Accuracy: %f',Accuracy)
Recall = TruePos0/(FalseNeg1+TruePos0)
print('Recall: %f',Recall)
Precision = TruePos0/(FalsePos0+TruePos0)
print('Precision: %f',Precision)

# ---------------------- Exp 3.1 full training ----------------------------

Exp31full = pd.read_csv("https://raw.githubusercontent.com/ekamineni/DAEN690/main/Datarepo/Exp31_Fulldata.csv")


#Copying only obj 1 -x, y, direction and obj 2-x, y, direction fields into X_train
#Copying only collision x, y and loop values into Y_train
X_train = Exp31full.iloc[:,:6].values
Y_train = Exp31full.iloc[:,-1].values

#Scaling the x values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

#Creating a model named as ann
ann = tf.keras.models.Sequential()

#Five layer ANN with activation as relu and number of nodes as 64
ann.add(tf.keras.layers.Dense(units=64,activation="relu"))
ann.add(tf.keras.layers.Dense(units=64,activation="relu"))
ann.add(tf.keras.layers.Dense(units=64,activation="relu"))
ann.add(tf.keras.layers.Dense(units=64,activation="relu"))
ann.add(tf.keras.layers.Dense(units=64,activation="relu"))

#Single output layer with activation as sigmoid
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))

ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

#Training the model with epochs as 100
ann.fit(X_train,Y_train,batch_size=32,epochs = 100)

#Evaluating the accuracy of test data
train_loss, train_accuracy = ann.evaluate(X_train, Y_train)

Y_pred_train = ann.predict(X_train)
Y_pred_train

Y_pred_train = np.round(Y_pred_train,0)
Y_pred_train

#Confusion matrix
cm = confusion_matrix(Y_train, Y_pred_train)
cm

TruePos0 = cm[0,0]
TrueNeg1 = cm[1,1]
FalsePos0 = cm[1,0]
FalseNeg1 = cm[0,1]

#Results of the model
Accuracy = (TruePos0+TrueNeg1)/(TruePos0+TrueNeg1+FalsePos0+FalseNeg1)
print('Accuracy: %f',Accuracy)
Recall = TruePos0/(FalseNeg1+TruePos0)
print('Recall: %f',Recall)
Precision = TruePos0/(FalsePos0+TruePos0)
print('Precision: %f',Precision)


# ------------------------ Exp 3.2 ----------------------
#Reading test and train data
train = pd.read_csv("https://raw.githubusercontent.com/ekamineni/DAEN690/main/Datarepo/Exp32_train.csv")
test = pd.read_csv("https://raw.githubusercontent.com/ekamineni/DAEN690/main/Datarepo/Exp32_test.csv")

#Copying only obj 1 -x, y, direction and obj 2-x, y, direction fields into X_train and X_test
#Copying only collision x, y and loop values into Y_train and Y_test
X_train = train.iloc[:,:6].values
Y_train = train.iloc[:,-1].values
X_test = test.iloc[:,:6].values
Y_test = test.iloc[:,-1].values

#Scaling the x values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Creating a model named as ann
ann = tf.keras.models.Sequential()

#Five layer ANN with activation as relu and number of nodes as 64
ann.add(tf.keras.layers.Dense(units=64,activation="relu"))
ann.add(tf.keras.layers.Dense(units=64,activation="relu"))
ann.add(tf.keras.layers.Dense(units=64,activation="relu"))
ann.add(tf.keras.layers.Dense(units=64,activation="relu"))
ann.add(tf.keras.layers.Dense(units=64,activation="relu"))

#Single output layer with activation as sigmoid
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))

ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

#Training the model with epochs as 100
ann.fit(X_train,Y_train,batch_size=32,epochs = 100)

#Evaluating the accuracy of test data
test_loss, test_accuracy = ann.evaluate(X_test, Y_test)

Y_pred = ann.predict(X_test)
Y_pred 

Y_pred = np.round(Y_pred,0)
Y_pred

#Confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
cm

TruePos0 = cm[0,0]
TrueNeg1 = cm[1,1]
FalsePos0 = cm[1,0]
FalseNeg1 = cm[0,1]

#Results of the model
Accuracy = (TruePos0+TrueNeg1)/(TruePos0+TrueNeg1+FalsePos0+FalseNeg1)
print('Accuracy: %f',Accuracy)
Recall = TruePos0/(FalseNeg1+TruePos0)
print('Recall: %f',Recall)
Precision = TruePos0/(FalsePos0+TruePos0)
print('Precision: %f',Precision)


# ---------------------- Exp 3.2 full training ----------------------------

Exp32full = pd.read_csv("https://raw.githubusercontent.com/ekamineni/DAEN690/main/Datarepo/Exp32_Fulldata.csv")


#Copying only obj 1 -x, y, direction and obj 2-x, y, direction fields into X_train
#Copying only collision x, y and loop values into Y_train
X_train = Exp32full.iloc[:,:6].values
Y_train = Exp32full.iloc[:,-1].values

#Scaling the x values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

#Creating a model named as ann
ann = tf.keras.models.Sequential()

#Five layer ANN with activation as relu and number of nodes as 64
ann.add(tf.keras.layers.Dense(units=64,activation="relu"))
ann.add(tf.keras.layers.Dense(units=64,activation="relu"))
ann.add(tf.keras.layers.Dense(units=64,activation="relu"))
ann.add(tf.keras.layers.Dense(units=64,activation="relu"))
ann.add(tf.keras.layers.Dense(units=64,activation="relu"))

#Single output layer with activation as sigmoid
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))

ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

#Training the model with epochs as 100
ann.fit(X_train,Y_train,batch_size=32,epochs = 100)

#Evaluating the accuracy of test data
train_loss, train_accuracy = ann.evaluate(X_train, Y_train)

Y_pred_train = ann.predict(X_train)
Y_pred_train

Y_pred_train = np.round(Y_pred_train,0)
Y_pred_train

#Confusion matrix
cm = confusion_matrix(Y_train, Y_pred_train)
cm

TruePos0 = cm[0,0]
TrueNeg1 = cm[1,1]
FalsePos0 = cm[1,0]
FalseNeg1 = cm[0,1]

#Results of the model
Accuracy = (TruePos0+TrueNeg1)/(TruePos0+TrueNeg1+FalsePos0+FalseNeg1)
print('Accuracy: %f',Accuracy)
Recall = TruePos0/(FalseNeg1+TruePos0)
print('Recall: %f',Recall)
Precision = TruePos0/(FalsePos0+TruePos0)
print('Precision: %f',Precision)
