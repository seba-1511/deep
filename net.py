import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
from scipy.io import loadmat
from scipy.stats.mstats import zscore
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn import svm

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

    # load train data
    train_x, train_y = load_data(range(2,3))
    valid_x, valid_y = load_data(range(9,10))

    # ica using one trial
    ica = decomposition.FastICA(n_components=25)
    S_ = ica.fit_transform(train_x[0].T)

    # reshape train ica
    train_ica_x = []
    for trial in range(len(train_x)):
        train_ica_x.append(train_x[trial].T)

    # reshape valid ica
    valid_ica_x = []
    for trial in range(len(valid_x)):
        valid_ica_x.append(valid_x[trial].T)

    # convert to numpy
    train_ica_x = np.array(train_ica_x)
    train_ica_x = np.vstack(train_ica_x)
    valid_ica_x = np.array(valid_ica_x)
    valid_ica_x = np.vstack(valid_ica_x)

    # fit/transform
    train_S = ica.fit_transform(train_ica_x)
    valid_S = ica.transform(valid_ica_x)

    # reshape train ica
    train_unmixed_x = []
    for i in range(0, len(train_S), 375):
        ica_signals = np.array(train_S[i:i+375])
        train_unmixed_x.append(ica_signals.T)
    train_unmixed_x = np.array(train_unmixed_x)

    # reshape valid ica
    valid_unmixed_x = []
    for i in range(0, len(valid_S), 375):
        ica_signals = np.array(valid_S[i:i+375])
        valid_unmixed_x.append(ica_signals.T)
    valid_unmixed_x = np.array(valid_unmixed_x)

    # concatenate sensors
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1]*train_x.shape[2])
    valid_x = valid_x.reshape(valid_x.shape[0], valid_x.shape[1]*valid_x.shape[2])
    train_unmixed_x = train_unmixed_x.reshape(train_unmixed_x.shape[0], train_unmixed_x.shape[1]*train_unmixed_x.shape[2])
    valid_unmixed_x = valid_unmixed_x.reshape(valid_unmixed_x.shape[0], valid_unmixed_x.shape[1]*valid_unmixed_x.shape[2])

    # train svm on original
    clf = svm.LinearSVC()
    clf.fit(train_x, train_y)
    print clf.score(valid_x, valid_y)

    # train on unmixed signal
    clf.fit(train_unmixed_x, train_y)
    print clf.score(valid_unmixed_x, valid_y)