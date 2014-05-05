__author__ = 'gabrielpereyra'

def create_features(train_x, tmin, tmax, sfreq, tmin_orginal =-0.5):
    beginning = np.round((tmin - tmin_original)*sfreq).astype(np.int)
    end = np.round((tmax-tmin_original)*sfreq).astype(np.int)
    train_x = train_x[:,:,beginning:end].copy()

    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2])

    train_x -= train_x.mean(0)
    train_x = np.nan_to_num(train_x / train_x.std(0))

    return train_x

import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.io import loadmat

train_file = open('train_01_06/train_subject01.mat')
valid_file = open('train_01_06/train_subject02.mat')

train_data = loadmat(train_file, squeeze_me=True)
valid_data = loadmat(valid_file, squeeze_me=True)

train_x = train_data['X']
valid_x = valid_data['X']
train_y = train_data['y']
valid_y = valid_data['y']

sfreq = train_data['sfreq']
tmin_original = train_data['tmin']

tmin = 0.0
tmax = 0.500

train_x = create_features(train_x, tmin, tmax, tmin_original)
valid_x = create_features(valid_x, tmin, tmax, tmin_original)

train_x = np.vstack(train_x)
valid_x = np.vstack(train_x)

clf = LogisticRegression(random_state=0)

clf.fit(train_x, train_y)


predict_y = clf.predict(valid_x)

print predict_y