import numpy as np
import random
import six.moves.cPickle as pickle
import scipy.io as sio

import theano
import theano.tensor as T

def _shared_dataset(data_xy, borrow=True):
	data_x, data_y = data_xy
	shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX),borrow=borrow)
	shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX),borrow=borrow)
	return shared_x, T.cast(shared_y, 'int32')

def _shared_X(data_x, borrow=True):
	return theano.shared(np.asarray(data_x, dtype=theano.config.floatX),borrow=borrow)
	

def load_data(train_file='data/traindata.mat',test_file='data/testdata.mat',validation_fraction=0.2):
	train_data = sio.loadmat('data/traindata.mat')
	test_data = sio.loadmat('data/testdata.mat')

	trainX = train_data['trainX']
	trainY = train_data['trainY']
	testX = test_data['testX']

	val_data_size = int(validation_fraction*len(trainY))

	train_data = zip(trainX,trainY)
	random.shuffle(train_data)

	val_data = train_data[:val_data_size]
	train_data = train_data[val_data_size:]

	trainX = [tup[0] for tup in train_data]
	trainY = [tup[1] for tup in train_data]
	
	valX = [tup[0] for tup in val_data]
	valY = [tup[1] for tup in val_data]

	return (_shared_dataset((trainX,trainY)),_shared_dataset((valX,valY)),_shared_X(testX))


