import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import numpy as np
from scipy.io import loadmat
from pylab import plot, show

def load_data(subject_range = range(1,17), test=False):
    """
    Read data from either train_01_16 or test_17_23 (if test=True)

    @param subject_range = range(first, last subject)
    @param test = True/False since test data has different y labels

    return x, y numpy arrays
    """
    x = []
    y = []

    for subject in subject_range:

        if(test):
            filename = 'test_17_23/test_subject%02d.mat' % (subject + 16) # so 1 = 17 (first subject)
            print "Loading: ", filename
            data = loadmat(filename, squeeze_me=True)
            subject_y = data['Id']
        else:
            filename = 'train_01_16/train_subject%02d.mat' % subject
            print "Loading: ", filename
            data = loadmat(filename, squeeze_me=True)
            subject_y = data['y']

        subject_x = data['X']
        x.append(subject_x)
        y.append(subject_y)

    x = np.vstack(x)
    y = np.concatenate(y)

    return x, y

def constrain_interval(x, tmin, tmax, sfreq = 250):
    beginning = np.round((tmin + 0.5)*sfreq).astype(np.int)
    end       = np.round((tmax + 0.5)*sfreq).astype(np.int)
    x = x[:,:,beginning:end].copy()
    return x

def concatenate_sensors(x):
    x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
    return x

def z_score(x):
    x -= x.mean(0)
    x = np.nan_to_num(x/x.std(0))
    return x

def plot_series(series_x, sfreq=.004):

    x_axis = [(-0.5 + sfreq * i) for i in range(375)]
    plot(x_axis,x[0][0])
    show()

if __name__ == "__main__":

    x, y = load_data(range(1,2))
    x = constrain_interval(x, 0, .5)
    print x.shape
    x = concatenate_sensors(x)
    print x.shape
    x = z_score(x)
    print x

