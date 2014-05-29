import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
from scipy.io import loadmat
from scipy.stats.mstats import zscore
import numpy as np
import pylab as pl
from sklearn import svm, decomposition

def load_data(subject_range, start=0, end=375, test=False):

    # list of all subjects
    x = []
    y = []

    # load subject data
    for subject in subject_range:

        # open train/test file
        if test:
            filename = 'test_17_23/test_subject%02d.mat' % subject
        else:
            filename = 'train_01_16/train_subject%02d.mat' % subject

        # load
        data = loadmat(filename, squeeze_me=True)

        # test = id/train = labels
        if test:
            subject_y = data['Id']
        else:
            subject_y = data['y']

        # z score and resize
        subject_x = data['X']
        subject_x = zscore(subject_x)
        subject_x = subject_x[:,:,start:end].copy()

        # append to subject list
        x.append(subject_x)
        y.append(subject_y)

    # make numpy arrays
    x = np.vstack(x)
    y = np.concatenate(y)

    return x, y

if __name__ == "__main__":

    # load train/valid data
    train_x, train_y = load_data(range(3,5))
    valid_x, valid_y = load_data(range(12,14))

    """
    # concatenate sensors
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1]*train_x.shape[2])
    valid_x = valid_x.reshape(valid_x.shape[0], valid_x.shape[1]*valid_x.shape[2])
    """

    # first sensor
    sensor = train_x[0][0]

    # fit ICA
    ica = decomposition.FastICA()
    S_ = ica.fit(sensor).transform(sensor)

    # plot sensor
    pl.figure()
    pl.subplot(2,1,1)
    pl.plot(sensor)

    # plot ICA
    pl.subplot(2,1,2)
    pl.plot(S_)
