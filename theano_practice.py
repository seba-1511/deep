# path to modules
import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")

# main modules
import numpy
import theano

class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):

        # initialize weights (n_int x n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                               name='W', borrow=True)

        # initialize baises b (n_out)
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # class-membership probabilities
        self.p_y_given_x = theano.tensor.nnet.softmax(
            theano.tensor.dot(input, self.W) + self.b)

        # paramaters
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):

        return -theano.tensor.mean(theano.tensor.log(
            self.p_y_given_x)[theano.tensor.arange(y.shape[0]), y])

    def errors(self, y):

        # check if y has the same dimensions of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                           ('y', target.type, 'y_pred', self.y_pred.type))

        # check if y os of the correct datatype
        if y.dtype.startswith('int'):
            return theano.tensor.mean(theano.tensor.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def load_data(dataset):

    # open file and discard header
    f = open(dataset)
    f.readline()

    # parse training data and convert to int
    data = [line.split(',') for line in f.readlines()]
    data = numpy.array(data, dtype=int)

    # set train/valid/test split
    num_train = len(data) * .5
    num_valid = len(data) * .25

    # split data
    train_set = data[:num_train]
    valid_set = data[num_train:num_train+num_valid]
    test_set  = data[num_train+num_valid:]

    # separate labels
    train_set_x = train_set[:,1:]
    train_set_y = train_set[:,:1]
    valid_set_x = valid_set[:,1:]
    valid_set_y = valid_set[:,:1]
    test_set_x  =  test_set[:,1:]
    test_set_y  =  test_set[:,:1]

    # return as list of x,y pairs
    return [[train_set_x, train_set_y],
            [valid_set_x, valid_set_y],
            [test_set_x , test_set_y]]

def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset='kaggle_digits/train.csv', batch_size=600):

    # load datasets
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x , test_set_y  = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches  =  test_set_x.get_value(borrow=True).shape[0] / batch_size

    print '... building the model'

    # allocate symbolic variables
    index = theano.tensor.lscalar()
    x = theano.tensor.matrix('x')
    y = theano.tensor.ivector('y')

    # construct logistic regression class
    classifier = LogisticRegression(input=x, n_in=28*28, n_out=10)

    # construct cost model
    cost = classifier.negative_log_likelihood(y)

    # compile theano test function
    test_model = theano.function(inputs=[index],
                                 outputs=classifier.errors(y),
                                 givens={
                                     x: test_set_x[index*batch_size:(index+1)*batch_size],
                                     y: test_set_y[index*batch_size:(index+1)*batch_size]
                                 })

    # compile theano validate function
    validate_model = theano.function(inputs=[index],
                                 outputs=classifier.errors(y),
                                 givens={
                                    x: valid_set_x[index*batch_size:(index+1)*batch_size],
                                    y: valid_set_y[index*batch_size:(index+1)*batch_size]
                                 })

    # compute the gradient cost with respect to theta = (W,b)
    g_W = theano.tensor.grad(cost=cost, wrt=classifier.W)
    g_b = theano.tensor.grad(cost=cost, wrt=classifier.b)

    # specify how to update the parameters of the model
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compile theano train function
    train_model = theano.function(inputs=[index],
                                  outputs=cost,
                                  givens={
                                      x: train_set_x[index*batch_size:(index+1)*batch_size],
                                      y: train_set_y[index*batch_size:(index+1)*batch_size]
                                  })

    print '... training the model'

    # early stopping parameters
    patience = 5000
    patience_increase = 2
    improvement_threshold = .995

    # validation parameters
    validation_frequency = min(n_train_batches, patience / 2)

    # best scores
    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.

    # set clock
    start_time = time.clock()

    # looping parameters
    done_looping = False
    epoch = 0

    while(epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                    for i in xrange(n_valid_batches)]
                this_validatioin_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index+1, n_train_batchs,
                      this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss

                    # test it on the test set
                    test_losses = [test_model(i)
                                    for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('    epoch %i, minibatch %i/%i, test error of best'
                          ' model %f %%') %
                          (epoch, minibatch_index+1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
          'with test performance %f %%') %
            (best_validation_loss * 100., test_score * 100.))

    print 'The code ran for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))

if __name__ == '__main__':

    datasets = load_data('kaggle_digits/train.csv')
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x , test_set_y  = datasets[2]

    print train_set_x.shape
    print train_set_y.shape

    print valid_set_x.shape
    print valid_set_y.shape

    print test_set_x.shape
    print test_set_y.shape