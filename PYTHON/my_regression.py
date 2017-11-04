import cPickle
import numpy
import theano
import theano.tensor as T

class Regression(object):
	def __init__(self, input, n_in, n_out):
		self.W=theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
		self.b=theano.shared(value=numpy.zeros((n_out, 1), dtype=theano.config.floatX), name='b', borrow=True)
		self.y_perd=[T.dot(input, self.W)+self.b]
		self.params=[self.W, self.b]
	def errors(self, y):
		return T.sum(T.sum((y-self.y_perd)**2, axis=1))
	
def load_data(dataset):
    print '... loading data'

    f = file(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.
	
    def shared_dataset(data_xy, borrow=True):
        # Function that loads the dataset into shared variables
		# The reason of using shared variables is to reduce memory read time
		data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, shared_y

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
	

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

def optimization(learning_rate=0.1, n_epochs=10, input_data='full_data.save', batch_size=10):
    datasets=load_data(input_data)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
	
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print '... building the model'
    index=T.lscalar() #index for the minibatch
	
    x=T.fmatrix('x') # input data
    y=T.fvector('y') # output
	
    classifier=Regression(input=x, n_in=4, n_out=1)
    cost=classifier.errors(y)
	
    test_model = theano.function(inputs=[index], outputs=classifier.errors(y), givens={x: test_set_x[index * batch_size: (index + 1) * batch_size], y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function(inputs=[index], outputs=classifier.errors(y), givens={x: valid_set_x[index * batch_size: (index + 1) * batch_size], y: valid_set_y[index * batch_size: (index + 1) * batch_size]})
	
	# compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)
	
	# compute the update parameters for the weigths
    updates = [(classifier.W, classifier.W - learning_rate * g_W), (classifier.b, classifier.b - learning_rate * g_b)]

	#retrain the model with the new weigths
    train_model = theano.function(inputs=[index], outputs=cost, updates=updates, givens={x: train_set_x[index * batch_size: (index + 1) * batch_size], y: train_set_y[index * batch_size: (index + 1) * batch_size]})
 
    print '... training the model'
    patience = 5000
    patience_increase=2
	
    improvement_threashold=0.995
    validation_frequency = min(n_train_batches, patience/2)
    best_validation_loss = numpy.inf
    test_score = 0.0
    done_looping = False
    epoch = 0
    while(epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print ('epoch %i, minibatch %i/%i, validation error %f %%' % (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

            if this_validation_loss < best_validation_loss:
                if this_validation_loss < best_validation_loss * improvement_threashold:
					patience = max(patience, iter * patience_increase)
                best_validation_loss = this_validation_loss
                test_losses = [test_model(i) for i in xrange(n_test_batches)]
                test_score = numpy.mean(test_losses)
                print(('     epoch %i, minibatch %i/%i, test error of best model %f %%') % (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))
            if patience <= iter:
                done_looping = True
                break
    print(('Optimitaion complete with best validation score of %f %%,' 'with test performance %f %%') % (best_validation_loss * 100., test_score * 100.))
    """			
	
	
if __name__ == '__main__':
    optimization()