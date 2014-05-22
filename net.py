import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
from scipy.io import loadmat
from scipy.stats.mstats import zscore
import numpy as np
from sklearn import svm
from sklearn.cross_validation import cross_val_score

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

    # create svm and score arrays
    channel_svm = svm.LinearSVC()
    score_channel = np.zeros(train_x.shape[1])
    score_channel_sums = np.zeros(train_x.shape[1])

    # cross-validate each sensor
    for channel in range(train_x.shape[1]):
        scores = cross_val_score(channel_svm, train_x[:,channel], train_y, cv=2, scoring='accuracy')
        score_channel[channel] = scores.mean()
        print "Channel #", channel, "Score:", score_channel[channel]

    print np.sort(score_channel)

    # cross validate dropping worst sensor
    for channel in range(train_x.shape[1]):
        x = train_x[:,channel:,:]
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        scores = cross_val_score(channel_svm, x, train_y, cv=2, scoring='accuracy')
        score_channel_sums[channel] = scores.mean()
        print "Train_x", x.shape, "Score:", score_channel_sums[channel]

    plt.plot(score_channel)
    plt.plot(score_channel_sums)
    plt.show()