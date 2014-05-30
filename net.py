import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
from scipy.io import loadmat
from scipy.stats.mstats import zscore
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt

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
    train_x, train_y = load_data(range(1,2))

    # ica using one trial
    ica = decomposition.FastICA(n_components=25)
    S_ = ica.fit_transform(train_x[0].T)

    # ica using all trials
    ica_x = []
    for trial in range(len(train_x)):
        ica_x.append(train_x[trial].T)

    # convert to numpy and stack
    ica_x = np.array(ica_x)
    ica_x = np.vstack(ica_x)

    # fit ica
    S_ = ica.fit_transform(ica_x)
    print S_.shape
    print ica_x.shape

    # shapes
    print ica_x[:375,0].shape
    print ica_x[:375,0]
    print S_[:375,0].shape
    print S_[:375,0]

    # normalize ica
    S_ = zscore(S_)

    # plot orginal and ica
    plt.plot(ica_x[:375,278], 'r-')
    plt.plot(S_[:375,1], 'b-')
    plt.show()