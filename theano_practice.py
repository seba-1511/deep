# path to modules
import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")

# main modules
import numpy as np
import theano

# open file and discard header
f = open('kaggle_digits/train.csv', 'r')
f.readline()

# parse into numpy array
d = [line.split(',') for line in f.read().split()]
d = np.array(d)
d = d.astype(np.float)

# separate labels and examples
train_y = d[:,0]
train_x = d[:,1:]

# symbolic variables
x = theano.tensor.matrix('x')
y = theano.tensor.vector('y')
w = theano.shared(np.random.randn(train_x.shape[1]), name='w')
b = theano.shared(0., name='b')

# print initial model
print "initial model:"
print w.get_value(), b.get_value()

# construct expression graph
p_1 = 1 / (1 + theano.tensor.exp(-theano.tensor.dot(x,w)-b))
prediction = p_1 > 0.5
xent = -y * theano.tensor.log(p_1)-(1-y)*theano.tensor.log(1-p_1)
cost = xent.mean() + 0.01 * (w**2).sum()
gw, gb = theano.tensor.grad(cost, [w,b])

# compile
train = theano.function(
    inputs=[x,y],
    outputs=[prediction, xent],
    updates=((w, w-0.1*gw),(b,b-0.1*gb))
)
predict = theano.function(inputs=[x], outputs=prediction)

# train
for i in range(10000):
    pred, err = train(train_x, train_y)

print 'final model:'
print w.get_value(), b.get_value()
print 'target values for D:', train_y
print 'prediction on D:', predict(train_x)