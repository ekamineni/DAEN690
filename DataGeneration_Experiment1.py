
"""
Created on 3/21/2022
 @author Sonal Dashora
 """

import pandas as pd
from random import randint

class Particle:
    def __init__(self,x,dir,bounce):
        self.x = x
        self.dir = dir
        self.bounce = 0

    def move(self, bounce=0):
        #### in the Matrix Scenarios
        if self.x > 0 and self.x < 9:
            if self.dir==0:
               self.x -= 1

            elif self.dir==1:
                self.x += 1

        ####covering all the corners
        elif self.x == 0:
            if self.dir==0:
                self.bounce += 1
                self.x += 1
                self.dir = 1

            elif self.dir==1:
                self.x += 1

        elif self.x == 9:
            if self.dir == 0:
                self.x -= 1

            elif self.dir == 1:
                self.bounce += 1
                self.x -= 1
                self.dir = 0

def simulation(obj1,counter=0):
    while counter <= 6:
        counter += 1
        obj1.move()
    return obj1


def generateInitialposition():
    obj1x = randint(0, 9)
    ####16 disctinct cases ###
    #obj1x = randint(0, 7)
    obj1dir = randint(0, 1)  # '0-l, 1-R
    obj1bounce = 0
    obj1 = Particle(obj1x,obj1dir,obj1bounce)
    return obj1

def main():
    obj1 = generateInitialposition()
    #print(obj1.x, obj1.dir)
    start_pos = [obj1.x,obj1.dir]
    #counter = 0
    objf  = simulation(obj1)
    #print(objf,'inside main')
    end_pos = [objf.x, objf.bounce]

    return start_pos + end_pos

from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.model_selection import train_test_split
ros = RandomOverSampler(random_state=42)

if __name__ == "__main__":
    n = 500 #number of datapoints to be simulated
    dataset_list = []
    for i in range(n):
        dataset_points = main()
        dataset_list.append(dataset_points)

    dataset = pd.DataFrame(dataset_list,
        columns=["objixPos", "objiDir", "objfxPos", "bounce"])
    dataset.drop_duplicates(inplace = True)


    """Experiment 1.1 Dataset generation"""

    x = dataset[["objixPos", "objiDir",'bounce']]
    y = dataset["objfxPos"]
    # # fit predictor and target variablex_ros,
    x_ros, y_ros = ros.fit_resample(x, y)

    print('exp1.1: Original dataset shape', Counter(y))
    print('exp1.1: Resample dataset shape', Counter(y_ros))

    #dataset = x_ros.insert(2,'objfxPos',y_ros)
    dataset_11 =pd.concat([x_ros,y_ros],axis = 1,ignore_index=True)
    dataset_11.columns = ["objixPos", "objiDir", "bounce", "objfxPos"]
    dataset_11 = dataset_11.reindex(columns = ["objixPos", "objiDir","objfxPos","bounce"])
    dataset_11 = dataset_11.append([dataset] * 5, ignore_index=True)

    dataset_11.to_csv('../output/exp1.1_dataset.csv', encoding='utf-8', index= False)


    """ Experiment 1.2 Dataset generation"""
    dataset.to_csv('../output/exp1.2_fulldata_test.csv', encoding='utf-8', index=False)
    train_data, test_data = train_test_split(dataset, train_size=0.70, random_state=42)
    x = train_data[["objixPos", "objiDir", 'bounce']]
    y = train_data["objfxPos"]
    # # fit predictor and target variablex_ros,
    x_ros, y_ros = ros.fit_resample(x, y)

    print('exp1.2: Original dataset shape', Counter(y))
    print('exp1.2: Resample dataset shape', Counter(y_ros))


    dataset_1_2 = pd.concat([x_ros, y_ros], axis=1, ignore_index=True)
    dataset_1_2.columns = ["objixPos", "objiDir", "bounce", "objfxPos"]
    dataset_1_2 = dataset_1_2.reindex(columns=["objixPos", "objiDir","objfxPos","bounce"])


    dataset_1_2 = dataset_1_2.append([dataset_1_2] * 5, ignore_index=True)

    dataset_1_2.to_csv('../output/exp1.2_datasets/exp1.2_traindata_70_30.csv', encoding='utf-8', index=False)
    test_data.to_csv('../output/exp1.2_datasets/exp1.2_testdata_70_30.csv', encoding='utf-8', index=False)
