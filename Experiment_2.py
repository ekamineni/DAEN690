#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 02:07:46 2022

@author: sushanth
"""

#Importing requried libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
from keras.models import Sequential


# -------------------------- Exp 2.1 -------------------------

#Reading test and train data
train = pd.read_csv("C:/Users/shant/Documents/GMU/FInal Sem/Final Project/Final Version Files/Exp21dataFull.csv")
test = pd.read_csv("C:/Users/shant/Documents/GMU/FInal Sem/Final Project/Final Version Files/Exp21test.csv")

#Copying only obj 1 -> x, y into X_train and X_test
#Copying only final obj 1 x, y values into Y_train and Y_test
X_train = train.iloc[:,:3].values
Y_train = train.iloc[:,3:5].values
X_test = test.iloc[:,:3].values
Y_test = test.iloc[:,3:5].values

#Scaling the x values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Function definition to create model with 64 nodes, kernel initializer as he_uniform and activation as relu
#input_number specify the number of inputs variables provided to model i.e 3 (obj 1 x,y,dir)
#output_number specify the number of output variables provided to model i.e 2 (final x,y)
def get_model(input_number, output_number):
    model = Sequential()
    model.add(Dense(64, input_dim=input_number, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(64, input_dim=input_number, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(64, input_dim=input_number, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(64, input_dim=input_number, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(64, input_dim=input_number, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(output_number))
    model.compile(loss='mae', optimizer='adam')
    return model

# Creating empty list
result = list()
#Calling the function to create a model with 3 inputs and 2 outputs
model = get_model(3,2)
#training the model
model.fit(X_train,Y_train, epochs=100, batch_size = 32)

#Model MAE
mae= model.evaluate(X_test,Y_test, verbose=0)
mae

# Finding Accuracy and confusion matrix for testing data
Y_pred = model.predict(X_test)
Y_pred 

Y_pred = np.round(Y_pred,0)
Y_pred

# Converting into dataframe
actual_test_df = pd.DataFrame(Y_test, columns=['x','y'])
actual_test_df

pred_test_df = pd.DataFrame(Y_pred, columns=['px','py'])
pred_test_df

actual_test_df.reset_index(inplace=True)
actual_test_df.drop('index',axis=1,inplace=True)
actual_test_df
pred_test_df.reset_index(inplace=True)
pred_test_df.drop('index',axis=1,inplace=True)
pred_test_df

leng = len(actual_test_df)

# Logic for creating the confusion matrix
count_crt = 0
count_fal = 0

for i in range(leng):   
    if((actual_test_df.iloc[i][0] == pred_test_df.iloc[i][0]) & (actual_test_df.iloc[i][1] == pred_test_df.iloc[i][1])):
        count_crt = count_crt+1
    else:
        count_fal = count_fal+1
        
print(count_fal)
print(count_crt)

# R square
from sklearn.metrics import r2_score
r2_score(actual_test_df,pred_test_df, multioutput='variance_weighted')



# ------------------------------- Exp 2.2 ------------------------

#Reading test and train data
train = pd.read_csv("C:/Users/shant/Documents/GMU/FInal Sem/Final Project/Final Version Files/Exp22train.csv")
test = pd.read_csv("C:/Users/shant/Documents/GMU/FInal Sem/Final Project/Final Version Files/Exp22test.csv")

#Copying only obj 1 -> x, y into X_train and X_test
#Copying only final obj 1 x, y values into Y_train and Y_test
X_train = train.iloc[:,:3].values
Y_train = train.iloc[:,3:5].values
X_test = test.iloc[:,:3].values
Y_test = test.iloc[:,3:5].values

#Scaling the x values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Function definition to create model with 64 nodes, kernel initializer as he_uniform and activation as relu
#input_number specify the number of inputs variables provided to model i.e 3 (obj 1 x,y,dir)
#output_number specify the number of output variables provided to model i.e 2 (final x,y)
def get_model(input_number, output_number):
    model = Sequential()
    model.add(Dense(64, input_dim=input_number, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(64, input_dim=input_number, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(64, input_dim=input_number, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(64, input_dim=input_number, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(64, input_dim=input_number, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(output_number))
    model.compile(loss='mae', optimizer='adam')
    return model

# Fitting the model
result = list()
model = get_model(3,2)

model.fit(X_train,Y_train, epochs=100, batch_size = 32)

#Model MAE
mae= model.evaluate(X_test,Y_test, verbose=0)
mae

#Finding the accuracy and confusion matrix of testing data
Y_pred = model.predict(X_test)
Y_pred 

Y_pred = np.round(Y_pred,0)
Y_pred

# Converting into dataframe
actual_test_df = pd.DataFrame(Y_test, columns=['x','y'])
actual_test_df

pred_test_df = pd.DataFrame(Y_pred, columns=['px','py'])
pred_test_df

actual_test_df.reset_index(inplace=True)
actual_test_df.drop('index',axis=1,inplace=True)
actual_test_df
pred_test_df.reset_index(inplace=True)
pred_test_df.drop('index',axis=1,inplace=True)
pred_test_df

leng = len(actual_test_df)

# Logic for creating the confusion matrix
count_crt = 0
count_fal = 0

for i in range(leng):   
    if((actual_test_df.iloc[i][0] == pred_test_df.iloc[i][0]) & (actual_test_df.iloc[i][1] == pred_test_df.iloc[i][1])):
        count_crt = count_crt+1
    else:
        count_fal = count_fal+1
        
print(count_fal)
print(count_crt)

# R square
from sklearn.metrics import r2_score
r2_score(actual_test_df,pred_test_df, multioutput='variance_weighted')

# ---------------------- Full dataset training -----------------------

#Reading test and train data
train = pd.read_csv("C:/Users/shant/Documents/GMU/FInal Sem/Final Project/Final Version Files/Exp21dataFull.csv")

#Copying only obj 1 -> x, y into X_train
#Copying only final obj 1 x, y values into Y_train
X_train = train.iloc[:,:3].values
Y_train = train.iloc[:,3:5].values

#Scaling the x values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

#Function definition to create model with 64 nodes, kernel initializer as he_uniform and activation as relu
#input_number specify the number of inputs variables provided to model i.e 3 (obj 1 x,y,dir)
#output_number specify the number of output variables provided to model i.e 2 (final x,y)
def get_model(input_number, output_number):
    model = Sequential()
    model.add(Dense(64, input_dim=input_number, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(64, input_dim=input_number, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(64, input_dim=input_number, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(64, input_dim=input_number, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(64, input_dim=input_number, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(output_number))
    model.compile(loss='mae', optimizer='adam')
    return model

# Fitting the model
result = list()
model = get_model(3,2)

model.fit(X_train,Y_train, epochs=100, batch_size = 32)

#Model MAE
mae= model.evaluate(X_train,Y_train, verbose=0)
mae

#Finding the accuracy and confusion matrix of testing data
Y_pred_train = model.predict(X_train)
Y_pred_train

Y_pred_train = np.round(Y_pred_train,0)
Y_pred_train

# Converting into dataframe
actual_train_df = pd.DataFrame(Y_train, columns=['x','y'])
actual_train_df

pred_train_df = pd.DataFrame(Y_pred_train, columns=['px','py'])
pred_train_df

actual_train_df.reset_index(inplace=True)
actual_train_df.drop('index',axis=1,inplace=True)
actual_train_df
pred_train_df.reset_index(inplace=True)
pred_train_df.drop('index',axis=1,inplace=True)
pred_train_df

leng = len(actual_train_df)

# Logic for creating the confusion matrix
count_crt = 0
count_fal = 0

for i in range(leng):   
    if((actual_train_df.iloc[i][0] == pred_train_df.iloc[i][0]) & (actual_train_df.iloc[i][1] == pred_train_df.iloc[i][1])):
        count_crt = count_crt+1
    else:
        count_fal = count_fal+1
        
print(count_fal)
print(count_crt)

# R square
from sklearn.metrics import r2_score
r2_score(actual_train_df,pred_train_df, multioutput='variance_weighted')

#