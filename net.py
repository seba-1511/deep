import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import pybrain
from scipy.io import loadmat
from scipy.stats.mstats import zscore
import numpy as np
from sklearn import svm

def load_data(subject_range, start=0, end=375):

    train_x = []
    train_y = []

    for subject in subject_range:

        filename = 'train_01_16/train_subject%02d.mat' % subject
        data = loadmat(filename, squeeze_me=True)
        x = data['X']
        y = data['y']
        x = zscore(x)
        x = x[:,:,start:end].copy()

        train_x.append(x)
        train_y.append(y)

    train_x = np.vstack(train_x)
    train_y = np.concatenate(train_y)

    return train_x, train_y

if __name__ == "__main__":

    x, y = load_data(range(1,8),125,225)
    train_x = x[:300]
    train_y = y[:300]

    valid_x = x[300:]
    valid_y = y[300:]

    train_x = train_x.reshape(train_x.shape[0],train_x.shape[1]*train_x.shape[2])
    valid_x = valid_x.reshape(valid_x.shape[0],valid_x.shape[1]*valid_x.shape[2])

    linear_svm = svm.LinearSVC()
    linear_svm.fit(train_x, train_y)

    print "LinearSVC:", linear_svm.score(valid_x, valid_y)
