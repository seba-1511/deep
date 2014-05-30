import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")

import numpy as np
import theano
import csv


# open file and discard header
f = open('kaggle_digits/train.csv', 'r')
f.readline()

# parse into numpy array
d = [line.split(',') for line in f.read().split()]
d = np.array(d)

# separate labels and examples
y = d[:,0]
x = d[:,1:]

print y.shape
print x.shape