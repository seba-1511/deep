import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import numpy as np
from scipy.io import loadmat

def create_features(x, tmin, tmax, sfreq, tmin_orginal=-0.5):
    beginning = np.round((tmin - tmin_orginal)*sfreq).astype(np.int)
    end       = np.round((tmax - tmin_orginal)*sfreq).astype(np.int)
    x = x[:,:,beginning:end].copy()

    x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

    x -= x.mean(0)
    x = np.nan_to_num(x/x.std(0))

    return x

def load_data(subject_range = range(1,17), tmin=0.0, tmax=0.5, test=False):

    x = []
    y = []

    for subject in subject_range:

        if(test):
            filename = 'test_17_23/test_subject%02d.mat' % subject
            data = loadmat(filename, squeeze_me=True)
            subject_y = data['Id']
        else:
            filename = 'train_01_16/train_subject%02d.mat' % subject
            data = loadmat(filename, squeeze_me=True)
            subject_y = data['y']

        subject_x = data['X']
        sfreq = data['sfreq']
        tmin_original = data['tmin']

        subject_x = create_features(x, tmin, tmax, sfreq, tmin_original)

        train_x.append(subject_x)
        train_y.append(subject_y)

    X = np.vstack(X)
    Y = np.concatenate(Y)

    return X, Y