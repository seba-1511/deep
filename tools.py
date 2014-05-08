import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import numpy as np
from scipy.io import loadmat
from pylab import plot, show

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
        sfreq = data['sfreq']
        tmin_original = data['tmin']

        subject_x = create_features(subject_x, tmin, tmax, sfreq, tmin_original)

        x.append(subject_x)
        y.append(subject_y)

    x = np.vstack(x)
    y = np.concatenate(y)

    return x, y

def view_data():

    x, y = load_data(range(1,2))

    sfreq = 1.0/250.0

    axis_x = [(-0.5 + sfreq * i) for i in range(375)]

    plot(axis_x,x[0][375*3:375*4])

    show()


if __name__ == "__main__":

    view_data()

