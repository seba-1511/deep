import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")

import numpy
import theano

rng = numpy.random

N = 400
feats = 784
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
training_steps = 1000

# declare symbolic variables
x = theano.tensor.matrix('x')
y = theano.tensor.vector('y')
w = theano.shared(rng.randn(feats), name='w')
b = theano.shared(0., name='w')
print 'Initial model:'
print w.get_value(), b.get_value()

# construct expression graph
p_1 = 1 / (1 + theano.tensor.exp(-theano.tensor.dot(x,w)-b))
prediction = p_1 > 0.5
xent = -y * theano.tensor.log(p_1) - (1-y) * theano.tensor.log(1-p_1)
cost = xent.mean() + 0.01 * (w**2).sum()
gw, gb = theano.tensor.grad(cost, [w,b])

# compile
train = theano.function(inputs=[x,y], outputs=[prediction,xent],
                        updates=(((w,w-0.1*gw),(b,b-0.1*gb))))
predict = theano.function(inputs=[x], outputs=prediction)

# train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

print "final model:"
print w.get_value(), b.get_value()
print "target values for D:", D[1]
print "prediction on D:", predict(D[0])