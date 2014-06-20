import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
from scipy.io import loadmat
from scipy.stats.mstats import zscore
import numpy as np
import pylab as plt
from sklearn import svm, cluster
import random
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from pylab import specgram

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

    # load training data
    X, y = load_data(range(1,2))

    cv = 2

    print "Computing cross-validated accuracy for each channel."
    clf = LogisticRegression(random_state=0)
    score_channel = np.zeros(X.shape[1])
    for channel in range(X.shape[1]):
        print "Channel #", channel, "score:",
        X_channel = X[:,channel,:].copy()
        scores = cross_val_score(clf, X_channel, y, cv=cv, scoring='accuracy')
        score_channel[channel] = scores.mean()
        print score_channel[channel]

    best_channels = np.argsort(score_channel)

    print "Best channel #", best_channels[-1], "Accuracy:", score_channel[best_channels[-1]]
    X_best_face = X[:,best_channels[-1],:][y==1].mean(0)
    X_best_scramble = X[:,best_channels[-1],:][y==0].mean(0)

    plt.plot(X_best_face, 'r-')
    plt.plot(X_best_scramble, 'b-')
    plt.savefig('best_sensor_raw.png')

    print "Worst channel #", best_channels[0], "Accuracy:", score_channel[best_channels[0]]
    X_worst_face = X[:,best_channels[0],:][y==1].mean(0)
    X_worst_scramble = X[:,best_channels[0],:][y==0].mean(0)

    plt.plot(X_worst_face, 'r-')
    plt.plot(X_worst_scramble, 'b-')
    plt.savefig('worst_sensor_raw.png')

    # fft best face frequency
    X_best_face_frequency = abs(np.fft.rfft(X_best_face))
    X_best_scramble_frequency = abs(np.fft.rfft(X_best_scramble))

    # fft worst face frequency
    X_worst_face_frequency = abs(np.fft.rfft(X_worst_face))
    X_worst_scramble_frequency = abs(np.fft.rfft(X_worst_scramble))

    # plot best face frequency
    plt.plot(X_best_face_frequency, 'r-')
    plt.plot(X_best_scramble_frequency, 'b-')
    plt.savefig('best_sensor_frequency.png')

    # plot worst face frequency
    plt.plot(X_worst_face_frequency, 'r-')
    plt.plot(X_worst_scramble_frequency, 'b-')
    plt.savefig('worst_sensor_frequency.png')

    # specgram of best face
    plt.clf()
    X_best_face_specgram = specgram(X_best_face)
    plt.savefig('X_best_face_specgram.png')

    # specgram of best scramble
    plt.clf()
    X_best_scramble_specgram = specgram(X_best_scramble)
    plt.savefig('X_best_scramble_specgram.png')

    # specgram of worst face
    plt.clf()
    X_worst_face_specgram = specgram(X_worst_face)
    plt.savefig('X_worst_face_specgram.png')

    # specgram of worst scramble
    plt.clf()
    X_worst_scramble_specgram = specgram(X_worst_scramble)
    plt.savefig('X_worst_scramble_specgram.png')