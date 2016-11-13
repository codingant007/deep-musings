# How to Train

1. Place the testdata.mat and traindata.mat files in a folder `data` adjacent to all the python code files
2. `cd` into source directory and `python train.py`
3. The predicted labels will be saved in `testY.txt`. It is a pickle file, so read the testY array appropriately.

# What the code does

train.py has some lists for the hyperparameters. Initialize the lists with the different values you wish to experiment on and start training.

The program trains the model for each of the combinations of hyperparameter values and saves the cost in each of the case to `costs/costs_<name based on hyperparameters>` so that later you can choose the best hyperparameter values set based on the result obtained.