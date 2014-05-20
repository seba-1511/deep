import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import numpy as np
import matplotlib.pyplot as plt
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.validation import Validator
from net import load_data

train_x, train_y = load_data(range(1,2))
valid_x, valid_y = load_data(range(9,10))

# scale 0-1
train_x = (train_x - np.amin(train_x))/(np.amax(train_x)-np.amin(train_x))
valid_x = ((valid_x - np.amin(valid_x))/(np.amax(valid_x)-np.amin(valid_x)))

# use first 10 sensors
train_x = train_x[:,:10,:].copy()
valid_x = valid_x[:,:10,:].copy()

# convolution (before concatenation)
new_x = []

conv_len = 10
conv = np.ones(conv_len) / conv_len

for trial in range(train_x.shape[0]):
    
    sensors_x = []
    
    for sensor in range(train_x.shape[1]):
                 
        conv_x = np.convolve(train_x[trial][sensor], conv,'valid')

        # max pooling
        pool_x = []
        pool_len = 25
        for i in range(0,375,pool_len):
            pool_x.append(np.amax(conv_x[i:i+pool_len]))
        
        sensors_x.append(pool_x)
    new_x.append(sensors_x)
    
train_x = np.array(new_x)

print train_x.shape
  
  
# validation convolution 
  
new_x = []

conv_len = 10
conv = np.ones(conv_len) / conv_len

for trial in range(valid_x.shape[0]):
    
    sensors_x = []
    
    for sensor in range(valid_x.shape[1]):
                 
        conv_x = np.convolve(valid_x[trial][sensor], conv,'valid')

        # max pooling
        pool_x = []
        pool_len = 25
        for i in range(0,375,pool_len):
            pool_x.append(np.amax(conv_x[i:i+pool_len]))
        
        sensors_x.append(pool_x)
    new_x.append(sensors_x)
    
valid_x = np.array(new_x)

print valid_x.shape
  
# concatenate sensors
train_x = train_x.reshape(train_x.shape[0],train_x.shape[1]*train_x.shape[2])
valid_x = valid_x.reshape(valid_x.shape[0],valid_x.shape[1]*valid_x.shape[2])
  
# build net and dataset
net = buildNetwork(150, 100, 1)
train_ds = SupervisedDataSet(150,1)
for trial, result in zip(train_x, train_y):
    train_ds.addSample(trial, result)

# train on dataset
trainer = BackpropTrainer(net, train_ds, verbose=True)
print trainer.trainUntilConvergence(verbose=True)