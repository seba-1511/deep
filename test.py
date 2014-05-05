__author__ = 'gabrielpereyra'

import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.io import loadmat

# load train x and y
train_data = loadmat('train_01_06/train_subject01.mat', squeeze_me=True)
train_x = train_data['X']
train_y = train_data['y']

# load valid x and y
valid_data = loadmat('train_01_06/train_subject02.mat', squeeze_me=True)
valid_x = valid_data['X']
valid_y = valid_data['y']

# unroll arrays
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2])
valid_x = valid_x.reshape(valid_x.shape[0], valid_x.shape[1] * valid_x.shape[2])

# normalize
train_x -= train_x.mean(0)
train_x = np.nan_to_num(train_x / train_x.std(0))

valid_x -= valid_x.mean(0)
valid_x = np.nan_to_num(valid_x / valid_x.std(0))

# fit logistic regression
clf = LogisticRegression(random_state=0)
clf.fit(train_x, train_y)

diff = clf.predict(valid_x)[0:2] - valid_y[0:2]
print diff
print (len(valid_y) - np.count_nonzero(diff)) / float(len(valid_y))