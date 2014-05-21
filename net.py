import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
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

    stacked_svms = []

    for i in range(1,9):

        print "Training svm #", i

        train_x, train_y = load_data(range(i,i+1),125,225)
        train_x = train_x.reshape(train_x.shape[0],train_x.shape[1]*train_x.shape[2])

        linear_svm = svm.SVC(probability=True, kernel='linear')
        linear_svm.fit(train_x, train_y)

        stacked_svms.append(linear_svm)

    valid_x, valid_y = load_data(range(9,10),125,225)
    valid_x = valid_x.reshape(valid_x.shape[0],valid_x.shape[1]*valid_x.shape[2])

    for i in range(len(stacked_svms)):

        print "svm #", i+1, "score", stacked_svms[i].score(valid_x, valid_y)



