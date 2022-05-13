# Deep Learning for System Test and Evaluation:
## _The Last Markdown Editor, Eswar Sesha Sai

[![N|Solid](https://upload.wikimedia.org/wikipedia/commons/5/53/George_Mason_University_logo.svg)]()

The emergent behavior of the system as a result of the interaction of behaviorally complex sub-systems is what defines complex systems. Although individual subsystems may be tested and certified for high levels of reliability, it is possible that the system's emergent behavior permits it to migrate into a dangerous operating mode. Even if all subsystems are functioning normally and no equipment has failed, this can still emerge. The system of systems of multiple components,where each component has a complex behavior. When components interact, the systems can exhibit emergent behavior. The problem area identifies the necessity for thorough testing of agent combinations that could lead to scenarios with extremely dangerous outcomes. Validation testing requires identification of all the emergent behavior and specially those related to hazardous conditions. Since, it is not possible to test this system with all possible conditions that lead to hazardous events/conditions, engineers used digital twin models of the system of systems interaction.

# Install
This project requires Python and the following Python libraries installed:

NumPy
Pandas
matplotlib
scikit-learn
tensorflow
Dense
Sequential
You will also need to have software installed to run and execute a Jupyter Notebook.

If you do not have Python installed yet, it is highly recommended that you install the Anaconda distribution of Python, which already has the above packages and more included.

## Code
Template code is provided in the .ipynb notebook file where each experimenet is named with the corresponding experiment number. You will also be required to use the the exp_data_example1.csv where example1 is the experiment name dataset file to complete your work. While some code has already been implemented to get you started, you will need to implement additional functionality when requested to successfully complete the project. If you are interested in see the the visualizations of the experiments please use the excel file to get the visulizations as the excel is programmed to get the visulizations of the experiments when you input the values manually to the excel sheet. 

## Run
In a terminal or command window, navigate to the top-level project directory DAEN690 (that contains this README) and run one of the following commands:

ipython notebook experiment4_1.py
or

jupyter notebook experiment4_1.py
or open with Jupyter Lab

jupyter lab
This will open the Jupyter Notebook software and project file in your browser.

## Data
The modified dataset consistes of  1M data records with an equal number of collision and non collision records that makes a balanced dataset.This dataset is a modified version of the Collision dataset found on the Git Repository.

## Field Descriptions:
Field Name and Description
obj1xPos Initial x-coordinate of object1
obj1yPos Initial y-coordinate of object1
obj1Dir Direction in which object 1 is traveling. There are 8 directions viz - 0-N, 1-NE, 2-E, 3-SE, 4-S,5-SW, 6 - W, 7 - NW
obj2xPos Initial x-coordinate of object2
obj2yPos Initial y-coordinate of object2
obj2Dir Direction in which object 2 is traveling. There are 8 directions viz - 0-N, 1-NE, 2-E, 3-SE, 4-S,5-SW, 6 - W, 7 - NW
objCxPos X-coordinate of the collision point of object1 and object 2 ​
objCyPos Y-coordinate of the collision point of object1 and object 2 ​
loop Time units after which the objects will collide​

