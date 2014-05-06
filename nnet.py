import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import numpy as np

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer

from scipy.io import loadmat

train_subjects = range(1,1)
train_x = []
train_y = []

for subject in train_subjects:
    filename = 'train_01_16/train_subject%02d.mat' % subject
    print 'Loading', filename
    data = loadmat(filename, squeeze_me=True)
    subject_x = data['X']
    subject_y = data['y']
    train_x.append(subject_x)
    train_y.append(subject_y)



net = buildNetwork(2,3,1, bias=True, hiddenclass=TanhLayer)
ds = SupervisedDataSet(2,1)

ds.addSample((0,0),(0,))
ds.addSample((0,1),(1,))
ds.addSample((1,0),(1,))
ds.addSample((1,1),(0,))

trainer = BackpropTrainer(net, ds)

print trainer.train()
print trainer.trainUntilConvergence()
