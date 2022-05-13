#!/usr/bin/env python
# coding: utf-8

# In[14]:


#imported the libraries

import numpy as np
import pandas as pd
import time as t
from random import randint


# In[2]:


# Function to generate initial coordinates

def generateInitialCoordinate():

    obj1xPos = randint(0,9)
    obj1yPos = randint(0,9)
    obj1Dir = randint(0,7)  # '0-N, 1-NE, 2-E, 3-SE, 4-S, 6 - W, 7 - NW

    obj2xPos = randint(0,9)
    obj2yPos = randint(0,9)
    obj2Dir = randint(0,7)
    
    return [obj1xPos,obj1yPos,obj1Dir,obj2xPos,obj2yPos,obj2Dir]


# In[3]:


# Function to generate the coordinates

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


# In[15]:


# Main function

def main():
    start_time= t.time()
    i =0
    initinalCoordinate_Df = pd.DataFrame(columns=["obj1xPos","obj1yPos","obj1Dir","obj2xPos","obj2yPos","obj2Dir","objCxPos","objCyPos","count"])
    while i < 100000:
        obj1xPos,obj1yPos,obj1Dir,obj2xPos,obj2yPos,obj2Dir = generateInitialCoordinate()
        objCxPos,objCyPos,count = objectNoEqual(obj1xPos,obj1yPos,obj1Dir,obj2xPos,obj2yPos,obj2Dir)
      
        initinalCoordinate_Df = initinalCoordinate_Df.append({"obj1xPos":obj1xPos,"obj1yPos":obj1yPos,"obj1Dir":obj1Dir,"obj2xPos":obj2xPos,
                                                              "obj2yPos":obj2yPos,"obj2Dir":obj2Dir,"objCxPos":objCxPos,"objCyPos":objCyPos,"count": count}, ignore_index=True)

        i = i+1
    print("Start Time",start_time)
    print("End time", t.time()- start_time)
    
    return initinalCoordinate_Df


# In[16]:


# Time model took to generate the records

initinalCoordinate_Df = main() #10000 points


# In[23]:


initinalCoordinate_Df.head()


# In[32]:


# Initial collision records in dataset

collision_data = initinalCoordinate_Df.loc[initinalCoordinate_Df['count'] != 20]
print(len(collision_data))


# In[33]:


# To resolve class bias issue, duplicating the collision dataset

initinalCoordinate_Df = initinalCoordinate_Df.append([collision_data] * 7 , ignore_index=True)
print(len(initinalCoordinate_Df))

initinalCoordinate_Df.to_csv('/Users/supriyasardana/Library/Mobile Documents/com~apple~CloudDocs/GeorgeMason/DAEN 690/Dataset/data1.csv',index=False)


# In[34]:


initinalCoordinate_Df[initinalCoordinate_Df.duplicated()]


# In[ ]:




