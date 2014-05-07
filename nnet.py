import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import numpy as np

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer

from scipy.io import loadmat

def create_features(x, tmin, tmax, sfreq, tmin_orginal=-0.5):
    beginning = np.round((tmin-tmin_orginal)*sfreq).astype(np.int)
    end       = np.round((tmax-tmin_orginal)*sfreq).astype(np.int)
    x = x[:,:,beginning:end].copy()
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    x -= x.mean(0)
    x = np.nan_to_num(x / x.std(0))
    return x

train_subjects = range(1,4)
train_x = []
train_y = []

tmin = 0.0
tmax = 0.500

for subject in train_subjects:
    filename = 'train_01_16/train_subject%02d.mat' % subject
    print 'Loading', filename
    data = loadmat(filename, squeeze_me=True)
    subject_x = data['X']
    subject_y = data['y']
    tmin_original = data['tmin']
    sfreq = data['sfreq']
    subject_x = create_features(subject_x, tmin, tmax, sfreq)
    train_x.append(subject_x)
    train_y.append(subject_y)

train_x = np.vstack(train_x)
train_y = np.concatenate(train_y)

print "Trainset:", train_x.shape

net = buildNetwork(38250,3,1, bias=True, hiddenclass=TanhLayer)
ds = SupervisedDataSet(38250,1)

for inp, target in zip(train_x, train_y):
    ds.addSample(inp, target)

trainer = BackpropTrainer(net, ds)

trainer.trainUntilConvergence(verbose=True)

data = loadmat('train_01_16/train_subject16.mat')
valid_x = data['X']
valid_y = data['y']
tmin_original = data['tmin']
sfreq = data['sfreq']

valid_x = create_features(valid_x, tmin, tmax, sfreq, tmin_original)
valid_x = np.vstack(valid_x)

predict_y = []

for sample in valid_x:
    predict_y.append(net.activate(sample))

predict_y = predict_y > .5

diff = predict_y - valid_y
print (len(valid_y) - np.count_nonzero(diff)) / float(len(valid_y))