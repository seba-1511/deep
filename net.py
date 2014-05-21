import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
from scipy.io import loadmat
from scipy.stats.mstats import zscore
import numpy as np
from sklearn import svm

def load_data(subject_range, start=0, end=375, test=False):

    train_x = []
    train_y = []

    for subject in subject_range:

        if test:
            filename = 'train_01_16/train_subject%02d.mat' % subject
            data = loadmat(filename, squeeze_me=True)
            x = data['X']
            y = data['y']
        else:
            filename = 'test_17_23/test_subject%02d.mat' % subject
            data = loadmat(filename, squeeze_me=True)
            x = data['X']
            y = data['Id']
        x = zscore(x)
        x = x[:,:,start:end].copy()

        train_x.append(x)
        train_y.append(y)

    train_x = np.vstack(train_x)
    train_y = np.concatenate(train_y)

    return train_x, train_y

if __name__ == "__main__":

    stacked_svms = []

    for i in range(1,10):

        print "Training subject #", i

        train_x, train_y = load_data(range(i,i+1),125,225)
        train_x = train_x.reshape(train_x.shape[0],train_x.shape[1]*train_x.shape[2])


        # make smaller for testing
        train_x = train_x[:10,:].copy()
        train_y = train_y[:10].copy()


        linear_svm = svm.SVC(probability=True, kernel='linear')
        linear_svm.fit(train_x, train_y)

        stacked_svms.append(linear_svm)


    total_correct = 0
    total_trials = 0


    for i in range(10,14):

    




        print "Validating subject #", i

        valid_x, valid_y = load_data(range(i,i+1),125,225)
        valid_x = valid_x.reshape(valid_x.shape[0],valid_x.shape[1]*valid_x.shape[2])

        """
        #make smaller for testing
        valid_x = valid_x[:50,:].copy()
        valid_y = valid_y[:50].copy()
        """

        correct = 0

        for i in range(len(valid_x)):

            print "Trial #", i

            probs = []

            for j in range(len(stacked_svms)):
                probs.append(stacked_svms[j].predict_proba(valid_x[i])[0])

            # stack probabilities of classes
            # rows = different svms
            # cols = probability of col # class
            # take max col as prediction (link to how to get max index)
            # http://stackoverflow.com/questions/3584243/python-get-the-position-of-the-biggest-item-in-a-numpy-array

            print "Label", valid_y[i]
            probs = np.vstack(probs)
            print probs
            prediction = np.unravel_index(probs.argmax(), probs.shape)[1]
            print prediction

            if prediction == valid_y[i]:

                print "Correct!"

                correct += 1

        total_correct += correct
        total_trials += len(valid_y)

        print correct
        print len(valid_y)
        print float(correct) / len(valid_y)

    print total_correct
    print total_trials
    print float(total_correct) / total_trials
