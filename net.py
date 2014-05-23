import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
from scipy.io import loadmat
from scipy.stats.mstats import zscore
import numpy as np
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from matplotlib import pyplot as plt

def load_data(subject_range, start=0, end=375, test=False):

    train_x = []
    train_y = []

    for subject in subject_range:

        if test:
            filename = 'test_17_23/test_subject%02d.mat' % subject
        else:
            filename = 'train_01_16/train_subject%02d.mat' % subject

        data = loadmat(filename, squeeze_me=True)

        if test:
            y = data['Id']
        else:
            y = data['y']

        x = data['X']
        x = zscore(x)
        x = x[:,:,start:end].copy()

        train_x.append(x)
        train_y.append(y)

    train_x = np.vstack(train_x)
    train_y = np.concatenate(train_y)

    return train_x, train_y

if __name__ == "__main__":

    # load data
    train_x, train_y = load_data(range(1,2),125,225)
    valid_x, valid_y = load_data(range(9,10),125,225)

    # concatenate sensors
    x = train_x.reshape(train_x.shape[0], train_x.shape[1]*train_x.shape[2])
    v = valid_x.reshape(valid_x.shape[0], valid_x.shape[1]*valid_x.shape[2])

    # validate
    clf = svm.LinearSVC()
    print "Cross validation (no convolution)", cross_val_score(clf, x, train_y, cv=2, scoring='accuracy')
    clf.fit(x, train_y)
    print "New subject (no convolution)", clf.score(v, valid_y)

    # create convolutional filter
    conv = np.ones(10)
    conv = conv / float(10)

    # convolve each sensor
    for i in range(train_x.shape[0]):
        for j in range(train_x.shape[1]):
            train_x[i][j] = np.convolve(train_x[i][j], conv, 'same')
            valid_x[i][j] = np.convolve(valid_x[i][j], conv, 'same')

    # concatenate sensors
    x = train_x.reshape(train_x.shape[0], train_x.shape[1]*train_x.shape[2])
    v = valid_x.reshape(valid_x.shape[0], valid_x.shape[1]*valid_x.shape[2])

    # validate
    print "Cross validation (convolution)", cross_val_score(clf, x, train_y, cv=2, scoring='accuracy')
    clf.fit(x, train_y)
    print "New subject (convolution)", clf.score(v, valid_y)
