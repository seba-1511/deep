import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
from scipy.io import loadmat
from scipy.stats.mstats import zscore
import numpy as np
import pylab as pl
from sklearn import svm, decomposition

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

    """
    # load train/valid data
    train_x, train_y = load_data(range(3,5))
    valid_x, valid_y = load_data(range(12,14))

    # concatenate sensors
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1]*train_x.shape[2])
    valid_x = valid_x.reshape(valid_x.shape[0], valid_x.shape[1]*valid_x.shape[2])
    """

    # generate sample data
    rng = np.random.RandomState(42)
    S = rng.standard_t(1.5, size=(20000,2))
    S[:,0] *= 2.

    # mix data
    A = np.array([[1,1], [0,2]])

    X = np.dot(S, A.T)

    pca = decomposition.PCA()
    S_pca_ = pca.fit(X).transform(X)

    ica = decomposition.FastICA(random_state=rng)
    S_ica_ = ica.fit(X).transform(X)

    S_ica_ /= S_ica_.std(axis=0)

    #plot results
    def plot_samples(S, axis_list=None):
        pl.scatter(S[:,0], S[:,1], s=2, marker='o', linewidths=0, zorder=10)
        if axis_list is not None:
            colors = [(0,0.6,0), (0.6,0,0)]
            for color, axis in zip(colors, axis_list):
                axis /= axis.std()
                x_axis, y_axis = axis
                # trick to get legend to work
                pl.plot(0.1*x_axis, 0.1 * y_axis, linewidth=2, color=color)
                pl.quiver(0,0,x_axis, y_axis,zorder=11,width=0.01,scale=6,color=color)

        pl.hlines(0,-3,3)
        pl.vlines(0,-3,3)
        pl.xlim(-3,3)
        pl.ylim(-3,3)
        pl.xlabel('x')
        pl.ylabel('y')

    pl.subplot(2,2,1)
    plot_samples(S / S.std())
    pl.title('True Independent Sources')

    axis_list = [pca.components_.T, ica.mixing_]
    pl.subplot(2,2,2)
    plot_samples(X / np.std(X), axis_list=axis_list)
    pl.legend(['PCA','ICA'], loc='upper left')
    pl.title('Observations')

    pl.subplot(2, 2, 3)
    plot_samples(S_pca_ / np.std(S_pca_, axis=0))
    pl.title('PCA scores')

    pl.subplot(2, 2, 4)
    plot_samples(S_ica_ / np.std(S_ica_))
    pl.title('ICA estimated sources')

    pl.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.26)

    pl.show()