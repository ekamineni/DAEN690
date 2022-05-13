#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing library
import pandas as pd
import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split

import numpy as np
#Ref: https://machinelearningmastery.com/deep-learning-models-for-multi-output-regression/


# In[2]:


# Importing Deep learning lib
from keras.layers import Dense
from keras.models import Sequential


# # Exp 4.1

# In[5]:


#Reading the data
data41_train = pd.read_csv('/Users/supriyasardana/Library/Mobile Documents/com~apple~CloudDocs/GeorgeMason/DAEN 690/Dataset/Exp41train_NoDup.csv')
data41_test = pd.read_csv('/Users/supriyasardana/Library/Mobile Documents/com~apple~CloudDocs/GeorgeMason/DAEN 690/Dataset/Exp41test_NoDup.csv')


#Checking for shape
data41_train.shape, data41_test.shape


# In[22]:


data41_train.head()


# In[4]:


#Spliting the data in train and test (70-30)

X_train_41 = data41_train.iloc[:,:6]
Y_train_41 = data41_train.iloc[:,-3:]

X_test_41 = data41_test.iloc[:,:6]
Y_test_41 = data41_test.iloc[:,-3:]


# In[16]:


# Checking size of collision data in training dataset
print(data41_train[data41_train['loop'] != 20].shape)

# Checking the size of Colision points in testing dataset
print(data41_test[data41_test['loop'] != 20].shape)


# In[6]:


#define the function to build the model

def get_model(input_number, output_number):
    model = Sequential()
    model.add(Dense(64, input_dim=input_number, kernel_initializer='he_uniform', activation='relu')) # 64 is the node count and used 5 hidden layers
    model.add(Dense(64, input_dim=input_number, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(64, input_dim=input_number, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(64, input_dim=input_number, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(64, input_dim=input_number, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(output_number))
    model.compile(loss='mae', optimizer='adam')
    return model


# In[7]:


# Model fitting
model = get_model(6,3) #There are 6 input variables and 3 output varibales
model.fit(X_train_41,Y_train_41, verbose=0, epochs=100)

# Checking model MAE
mae= model.evaluate(X_test_41,Y_test_41, verbose=0)
mae


# In[8]:


#Predicting Values and rounding to 2 decimal place for training dataset

Y_pred_41_train = model.predict(X_train_41)
Y_pred_41_train = np.round(abs(Y_pred_41_train))

# Converting the test and predicted values into dataframe
actual_test_df_41_train = pd.DataFrame(Y_train_41)
actual_test_df_41_train.reset_index(inplace=True)
actual_test_df_41_train.drop('index',axis=1,inplace=True)

pred_test_df_41_train = pd.DataFrame(Y_pred_41_train, columns=['X_Collision','Y_Collision','Loop'])
pred_test_df_41_train.reset_index(inplace=True)
pred_test_df_41_train.drop('index',axis=1,inplace=True)


# In[9]:


# Logic for creating the confusion matrix for training data

count_true_c_41_tr = 0       # variable will store count for collision cases that are predicted correctly
count_true_nc_41_tr = 0       #variable will store count for non collision cases that are predicted correctly
count_false_41_tr = 0        # variable will store count for cases that are predicted incorrect

for i in range(actual_test_df_41_train.shape[0]):
    
    if((actual_test_df_41_train.iloc[i][0] == pred_test_df_41_train.iloc[i][0]) & (actual_test_df_41_train.iloc[i][1] == pred_test_df_41_train.iloc[i][1]) & (actual_test_df_41_train.iloc[i][2] == pred_test_df_41_train.iloc[i][2]) & (actual_test_df_41_train.iloc[i][2] == 20)):
        count_true_nc_41_tr += 1
    elif((actual_test_df_41_train.iloc[i][0] == pred_test_df_41_train.iloc[i][0]) & (actual_test_df_41_train.iloc[i][1] == pred_test_df_41_train.iloc[i][1]) & (actual_test_df_41_train.iloc[i][2] == pred_test_df_41_train.iloc[i][2]) & (actual_test_df_41_train.iloc[i][2] != 20)):
        count_true_c_41_tr += 1
    else:
        count_false_41_tr += 1
    

print(count_false_41_tr, count_true_c_41_tr, count_true_nc_41_tr) 


# In[10]:


#Predicting Values and rounding to 2 decimal place for testing dataset

Y_pred_41 = model.predict(X_test_41)
Y_pred_41 = np.round(abs(Y_pred_41))

# Converting the test and predicted values into dataframe
actual_test_df_41 = pd.DataFrame(Y_test_41)
actual_test_df_41.reset_index(inplace=True)
actual_test_df_41.drop('index',axis=1,inplace=True)

pred_test_df_41 = pd.DataFrame(Y_pred_41, columns=['X_Collision','Y_Collision','Loop'])
pred_test_df_41.reset_index(inplace=True)
pred_test_df_41.drop('index',axis=1,inplace=True)


# In[11]:


# Logic for creating the confusion matrix for testing data

count_true_c_41 = 0       # variable will store count for collision cases that are predicted correctly
count_true_nc_41 = 0       #variable will store count for non collision cases that are predicted correctly
count_false_41 = 0        # variable will store count for cases that are predicted incorrect

for i in range(actual_test_df_41.shape[0]):
    
    if((actual_test_df_41.iloc[i][0] == pred_test_df_41.iloc[i][0]) & (actual_test_df_41.iloc[i][1] == pred_test_df_41.iloc[i][1]) & (actual_test_df_41.iloc[i][2] == pred_test_df_41.iloc[i][2]) & (actual_test_df_41.iloc[i][2] == 20)):
        count_true_nc_41 += 1
    elif((actual_test_df_41.iloc[i][0] == pred_test_df_41.iloc[i][0]) & (actual_test_df_41.iloc[i][1] == pred_test_df_41.iloc[i][1]) & (actual_test_df_41.iloc[i][2] == pred_test_df_41.iloc[i][2]) & (actual_test_df_41.iloc[i][2] != 20)):
        count_true_c_41 += 1
    else:
        count_false_41 += 1
    

print(count_false_41, count_true_c_41, count_true_nc_41)   
    


# In[12]:


# Training model with complete dataset

data_41 = pd.read_csv('/Users/supriyasardana/Library/Mobile Documents/com~apple~CloudDocs/GeorgeMason/DAEN 690/Dataset/Exp41-Complete.csv')

x_data_41_com = data_41.iloc[:,1:7]
y_data_41_com = data_41.iloc[:,-3:]


# Model fitting for complete dataset
model_41_c = get_model(6,3) #There are 6 input variables and 3 output varibales
model_41_c.fit(x_data_41_com,y_data_41_com, verbose=0, epochs=100)


# In[14]:


# Predicting values for complete dataset

Y_pred_41_com = model_41_c.predict(x_data_41_com)
Y_pred_41_com = np.round(abs(Y_pred_41_com))

# Converting the test and predicted values into dataframe
actual_test_df_41_com = pd.DataFrame(y_data_41_com)
actual_test_df_41_com.reset_index(inplace=True)
actual_test_df_41_com.drop('index',axis=1,inplace=True)

pred_test_df_41_com = pd.DataFrame(Y_pred_41_com, columns=['X_Collision','Y_Collision','Loop'])
pred_test_df_41_com.reset_index(inplace=True)
pred_test_df_41_com.drop('index',axis=1,inplace=True)


# In[15]:


# Logic for creating the confusion matrix for training data

count_true_c_41_com = 0       # variable will store count for collision cases that are predicted correctly
count_true_nc_41_com = 0       #variable will store count for non collision cases that are predicted correctly
count_false_41_com = 0        # variable will store count for cases that are predicted incorrect

for i in range(actual_test_df_41_com.shape[0]):
    
    if((actual_test_df_41_com.iloc[i][0] == pred_test_df_41_com.iloc[i][0]) & (actual_test_df_41_com.iloc[i][1] == pred_test_df_41_com.iloc[i][1]) & (actual_test_df_41_com.iloc[i][2] == pred_test_df_41_com.iloc[i][2]) & (actual_test_df_41_com.iloc[i][2] == 20)):
        count_true_nc_41_com += 1
    elif((actual_test_df_41_com.iloc[i][0] == pred_test_df_41_com.iloc[i][0]) & (actual_test_df_41_com.iloc[i][1] == pred_test_df_41_com.iloc[i][1]) & (actual_test_df_41_com.iloc[i][2] == pred_test_df_41_com.iloc[i][2]) & (actual_test_df_41_com.iloc[i][2] != 20)):
        count_true_c_41_com += 1
    else:
        count_false_41_com += 1
    

print(count_false_41_com, count_true_c_41_com, count_true_nc_41_com) 


# # Exp 4.2

# In[8]:


# Reading the data
data_42 = pd.read_csv('/Users/supriyasardana/Library/Mobile Documents/com~apple~CloudDocs/GeorgeMason/DAEN 690/Dataset/Exp_42.csv')

#Checking for shape
data_42.shape


# In[18]:


#Spliting the data in train and test (70-30)

x_data_42 = data_42.iloc[:,:6]
y_data_42 = data_42.iloc[:,-3:]

X_train_42, X_test_42, Y_train_42, Y_test_42 = train_test_split(x_data_42,y_data_42, test_size=0.3, random_state=12)


# In[19]:


# Size of the train and test shape
X_train_42.shape,X_test_42.shape


# In[20]:


# Checking size of collision data in complete dataset
print(data_42[data_42['count'] != 20].shape)

# Checking the size of Colision points in testing dataset
print(Y_test_42[Y_test_42['count'] != 20].shape)


# In[21]:


# Model fitting
model = get_model(6,3) #There are 6 input variables and 3 output varibales
model.fit(X_train_42,Y_train_42, verbose=0, epochs=20)

# Checking model MAE
mae= model.evaluate(X_test_42,Y_test_42, verbose=0)
mae


# In[ ]:





# In[38]:


# Predicting values for training dataset

Y_pred_42_tr = model.predict(X_train_42)
Y_pred_42_tr = np.round(abs(Y_pred_42_tr))

# Converting the test and predicted values into dataframe
actual_test_df_42_tr = pd.DataFrame(Y_train_42)
actual_test_df_42_tr.reset_index(inplace=True)
actual_test_df_42_tr.drop('index',axis=1,inplace=True)

pred_test_df_42_tr = pd.DataFrame(Y_pred_42_tr, columns=['X_Collision','Y_Collision','Loop'])
pred_test_df_42_tr.reset_index(inplace=True)
pred_test_df_42_tr.drop('index',axis=1,inplace=True)


# In[40]:


# Logic for creating the confusion matrix for training data

count_true_c_42_tr = 0       # variable will store count for collision cases that are predicted correctly
count_true_nc_42_tr = 0       #variable will store count for non collision cases that are predicted correctly
count_false_42_tr = 0        # variable will store count for cases that are predicted incorrect

for i in range(actual_test_df_42_tr.shape[0]):
    
    if((actual_test_df_42_tr.iloc[i][0] == pred_test_df_42_tr.iloc[i][0]) & (actual_test_df_42_tr.iloc[i][1] == pred_test_df_42_tr.iloc[i][1]) & (actual_test_df_42_tr.iloc[i][2] == pred_test_df_42_tr.iloc[i][2]) & (actual_test_df_42_tr.iloc[i][2] == 20)):
        count_true_nc_42_tr += 1
    elif((actual_test_df_42_tr.iloc[i][0] == pred_test_df_42_tr.iloc[i][0]) & (actual_test_df_42_tr.iloc[i][1] == pred_test_df_42_tr.iloc[i][1]) & (actual_test_df_42_tr.iloc[i][2] == pred_test_df_42_tr.iloc[i][2]) & (actual_test_df_42_tr.iloc[i][2] != 20)):
        count_true_c_42_tr += 1
    else:
        count_false_42_tr += 1
        
print(count_false_42_tr, count_true_c_42_tr, count_true_nc_42_tr) 


# In[43]:


#Predicting Values and rounding to 2 decimal place

Y_pred_42 = model.predict(X_test_42)
Y_pred_42 = np.round(abs(Y_pred_42))

# Converting the test and predicted values into dataframe
actual_test_df_42 = pd.DataFrame(Y_test_42)
actual_test_df_42.reset_index(inplace=True)
actual_test_df_42.drop('index',axis=1,inplace=True)

pred_test_df_42 = pd.DataFrame(Y_pred_42, columns=['X_Collision','Y_Collision','Loop'])
pred_test_df_42.reset_index(inplace=True)
pred_test_df_42.drop('index',axis=1,inplace=True)


# In[21]:


actual_test_df_42['count'].isin(pred_test_df_42['Loop']).value_counts()


# In[22]:


pred_test_df_42['Loop'].isin(actual_test_df_42['count']).value_counts()


# In[37]:


# Checking the size of Colision points in testing dataset
print(Y_test_42[Y_test_42['count'] != 20].shape)


# In[46]:


# Logic for creating the confusion matrix

count_true_c_42 = 0       # variable will store count for collision cases that are predicted correctly
count_true_nc_42 = 0       #variable will store count for non collision cases that are predicted correctly
count_false_42 = 0        # variable will store count for cases that are predicted incorrect

for i in range(actual_test_df_42.shape[0]):
    
    if((actual_test_df_42.iloc[i][0] == pred_test_df_42.iloc[i][0]) & (actual_test_df_42.iloc[i][1] == pred_test_df_42.iloc[i][1]) & (actual_test_df_42.iloc[i][2] == pred_test_df_42.iloc[i][2]) & (actual_test_df_42.iloc[i][2] == 20)):
        count_true_nc_42 += 1
    elif((actual_test_df_42.iloc[i][0] == pred_test_df_42.iloc[i][0]) & (actual_test_df_42.iloc[i][1] == pred_test_df_42.iloc[i][1]) & (actual_test_df_42.iloc[i][2] == pred_test_df_42.iloc[i][2]) & (actual_test_df_42.iloc[i][2] != 20)):
        count_true_c_42 += 1
    else:
        count_false_42 += 1
    

print(count_false_42, count_true_c_42, count_true_nc_42) 
    


# In[47]:


# Checking R square value

from sklearn.metrics import r2_score
r2_score(actual_test_df_42,pred_test_df_42, multioutput='variance_weighted')


# In[48]:


# Training model with complete dataset


x_data_42_com = data_42.iloc[:,:6]
y_data_42_com = data_42.iloc[:,-3:]


# Model fitting for complete dataset
model = get_model(6,3) #There are 6 input variables and 3 output varibales
model.fit(x_data_42_com,y_data_42_com, verbose=0, epochs=20)


# In[49]:


# Predicting values for complete dataset

Y_pred_42_com = model.predict(x_data_42_com)
Y_pred_42_com = np.round(abs(Y_pred_42_com))

# Converting the test and predicted values into dataframe
actual_test_df_42_com = pd.DataFrame(y_data_42_com)
actual_test_df_42_com.reset_index(inplace=True)
actual_test_df_42_com.drop('index',axis=1,inplace=True)

pred_test_df_42_com = pd.DataFrame(Y_pred_42_com, columns=['X_Collision','Y_Collision','Loop'])
pred_test_df_42_com.reset_index(inplace=True)
pred_test_df_42_com.drop('index',axis=1,inplace=True)


# In[50]:


# Logic for creating the confusion matrix

count_true_c_42_com = 0       # variable will store count for collision cases that are predicted correctly
count_true_nc_42_com = 0       #variable will store count for non collision cases that are predicted correctly
count_false_42_com = 0        # variable will store count for cases that are predicted incorrect

for i in range(actual_test_df_42_com.shape[0]):
    
    if((actual_test_df_42_com.iloc[i][0] == pred_test_df_42_com.iloc[i][0]) & (actual_test_df_42_com.iloc[i][1] == pred_test_df_42_com.iloc[i][1]) & (actual_test_df_42_com.iloc[i][2] == pred_test_df_42_com.iloc[i][2]) & (actual_test_df_42_com.iloc[i][2] == 20)):
        count_true_nc_42_com += 1
    elif((actual_test_df_42_com.iloc[i][0] == pred_test_df_42_com.iloc[i][0]) & (actual_test_df_42_com.iloc[i][1] == pred_test_df_42_com.iloc[i][1]) & (actual_test_df_42_com.iloc[i][2] == pred_test_df_42_com.iloc[i][2]) & (actual_test_df_42_com.iloc[i][2] != 20)):
        count_true_c_42_com += 1
    else:
        count_false_42_com += 1
    

print(count_false_42_com, count_true_c_42_com, count_true_nc_42_com) 
    


# In[ ]:




