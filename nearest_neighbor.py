import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import numpy as np
from sklearn import neighbors
from tools import load_data, constrain_interval, concatenate_sensors, z_score

# load and process training data
train_x, train_y = load_data(range(1,9))
train_x = constrain_interval(train_x, 0, 0.5)
train_x = concatenate_sensors(train_x)
train_x = z_score(train_x)

valid_x, valid_y = load_data(range(9,17))
valid_x = constrain_interval(valid_x, 0, 0.5)
valid_x = concatenate_sensors(valid_x)
valid_x = z_score(valid_x)

n_neighbors = 15

clf = neighbors.KNeighborsClassifier(n_neighbors, 'distance')
clf.fit(train_x, train_y)

print clf.score(valid_x, valid_y)