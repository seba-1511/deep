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

def butterfly_plot(trial_x, sfreq=.04):
    """
    plots one trial (306 sensors) in one chart
    @param trial_x = 306x375 numpy array
    @param sfreq = frequency (default is 1/25)
    """

    # change so range is fixed -5 through 1 for varying frequencies
    x_axis = [(-0.5 + sfreq * i) for i in range(375)]

    for sensor in trial_x:
        plot(x_axis,sensor)
    show()

def create_submission_file(ids, prediction):
    filename_submission = "submission.csv"
    print "Creating submission file", filename_submission
    f = open(filename_submission, "w")
    print >> f, "Id,Prediction"
    for i in range(len(prediction)):
        print >> f, str(ids[i]) + "," + str(prediction[i])
    f.close()

if __name__ == "__main__":

    x, y = load_data(range(1,2))

    butterfly_plot(x[2])
