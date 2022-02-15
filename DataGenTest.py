# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 19:35:16 2022

@author: umaen
"""
import random
import pandas as pd
import tensorflow as tf

df = pd.DataFrame(columns=['Obj1XPos','Obj1YPos','Obj1Dir','Obj2XPos','Obj2YPos',
                           'Obj2Dir','CollisionX','CollisionY','Outcome'])

print("Obj1X  Obj1Y  Obj1Dir  Obj2X  Obj2Y  Obj2Dir  ColX  ColY  Count")
for i in range(20000):
    obj1xPos = random.randrange(0,9);
    obj1yPos = random.randrange(0,9);
    obj1Dir = random.randrange(0,7);
    obj2xPos = random.randrange(0,9);
    obj2yPos = random.randrange(0,9);
    obj2Dir = random.randrange(0,7);
    ini1x=obj1xPos
    ini1y=obj1yPos
    ini1dir=obj1Dir
    ini2x=obj2xPos
    ini2y=obj2yPos
    ini2dir=obj2Dir
    count = 0
    while(obj1xPos != obj2xPos or obj1yPos!=obj2yPos) and count<20:
        count=count+1;
        """Obj1"""
        if (obj1xPos > 1 and obj1xPos < 10 and obj1yPos > 1 and obj1yPos < 10):
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
        elif (obj1xPos == 1 and obj1yPos == 1):
            obj1xPos == obj1xPos + 1
            obj1yPos == obj1yPos + 1
            obj1Dir = 3
        elif (obj1xPos == 10 and obj1yPos == 1):
            obj1xPos = obj1xPos - 1
            obj1yPos = obj1yPos + 1
            obj1Dir = 5
        elif (obj1xPos == 1 and obj1yPos == 10):
            obj1xPos = obj1xPos + 1
            obj1yPos = obj1yPos - 1
            obj1Dir = 1
        elif (obj1xPos == 10 and obj1yPos == 10):
            obj1xPos = obj1xPos - 1
            obj1yPos = obj1yPos - 1
            obj1Dir = 7        
        elif (obj1xPos == 1):
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
        elif (obj1xPos == 10):
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
        elif (obj1yPos == 1):
            if (obj1Dir == 0):
                obj1yPos = obj1yPos - 1
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
            """ End of inside if loop"""

        elif (obj1yPos == 10):
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
        
        
        """Start of obj 2"""
        if (obj2xPos > 1 and obj2xPos < 10 and obj2yPos > 1 and obj2yPos < 10):
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
        elif (obj2xPos == 1 and obj2yPos == 1):
            obj2xPos = obj2xPos + 1
            obj2yPos = obj2yPos + 1
            obj2Dir = 3

        elif (obj2xPos == 10 and obj2yPos == 1):
            obj2xPos = obj2xPos - 1
            obj2yPos = obj2yPos + 1
            obj2Dir = 5

        elif (obj2xPos == 1 and obj2yPos == 10):
            obj2xPos = obj2xPos + 1
            obj2yPos = obj2yPos - 1
            obj2Dir = 1

        elif (obj2xPos == 10 and obj2yPos == 10):
            obj2xPos = obj2xPos - 1
            obj2yPos = obj2yPos - 1
            obj2Dir = 7

        elif (obj2xPos == 1):
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
        elif (obj2xPos == 10):
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
        elif (obj2yPos == 1):
            if (obj2Dir == 0):
                obj2yPos = obj2yPos - 1
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
        elif (obj2yPos == 10):
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
            """End of inside if loop """
        """End of obj 2 loop """
    print("  ",ini1x,"    ",ini1y,"    ",ini1dir,"    ",ini2x,"    ",ini2y,"    ",ini2dir,
          "    ",obj1xPos,"    ",obj1yPos,"    ",count)
    if (count == 20):
        outcome = 0
    else:
        outcome = 1
    ini1x=pd.to_numeric(ini1x)
    df = df.append({'Obj1XPos':ini1x,'Obj1YPos':ini1y,'Obj1Dir':ini1dir,'Obj2XPos':ini2x,
                    'Obj2YPos':ini2y,'Obj2Dir':ini2dir,'CollisionX':obj1xPos,
                    'CollisionY':obj1yPos,'Outcome':outcome},ignore_index=True)

df = df.apply(pd.to_numeric)
print(df.describe())

dup = df.duplicated()

"""
df_final = df[~dup]

X = df_final.iloc[:,:6].values
Y = df_final.iloc[:,-1].values

#Splitting dataset into training and testing dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

ann.fit(X_train,Y_train,batch_size=32,epochs = 100)
print(ann.predict())


done=df[df.columns[8]]==1

"""