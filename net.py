import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
from scipy.io import loadmat
from scipy.stats.mstats import zscore
import numpy as np
from sklearn import svm, neural_network
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
    train_x, train_y = load_data(range(1,2))
    test_x, test_y = load_data(range(17,18), test=True)

    # concatenate sensors
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1]*train_x.shape[2])
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1]*test_x.shape[2])

    # shorten for testing
    train_x = train_x[:10].copy()
    test_x = test_x[:10].copy()


    # create export file
    f = open('train_transduction.dat','w')

    # write labeled data
    for trial in range(len(train_x)):

        print "Writing labeled trial #", trial, "/", len(train_x)

        if train_y[trial] == 1:
            line = "1"
        else:
            line = "-1"

        for feature in range(len(train_x[trial])):
            line += " " + str(feature+1) + ":" + str(train_x[trial][feature])

        print >> f, line

    # write unlabeled data
    for trial in range(len(test_x)):

        print "Writing unlabeled trial #", trial, "/", len(train_x)

        line = "0"

        for feature in range(len(test_x[trial])):
            line += " " + str(feature+1) + ":" + str(train_x[trial][feature])

        print >> f, line

    f.close()
