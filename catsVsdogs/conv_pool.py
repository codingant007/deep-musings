from __future__ import print_function

import os
import sys
import timeit
import six.moves.cPickle as pickle

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from mlp import HiddenLayer

srng = RandomStreams() 

class ConvPoolLayer(object):
	"""Pool Layer of a convolutional network """

	def __init__(self, rng, input, filter_shape, image_shape, 
					poolsize=(2, 2), activation=T.nnet.relu, 
					dropout=0, W=None, b=None):

		assert image_shape[1] == filter_shape[1]
		self.input = input

		if W is None:
			# there are "num input feature maps * filter height * filter width"
			# inputs to each hidden unit
			fan_in = np.prod(filter_shape[1:])
			# each unit in the lower layer receives a gradient from:
			# "num output feature maps * filter height * filter width" /
			#   pooling size
			fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
					   np.prod(poolsize))
			# initialize weights with random weights
			W_bound = np.sqrt(6. / (fan_in + fan_out))
			W = theano.shared(
				np.asarray(
					rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
					dtype=theano.config.floatX
				),
				name='W',
				borrow=True
			)

		if b is None:
			# the bias is a 1D tensor -- one bias per output feature map
			b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)
		
		self.W = W
		self.b = b

		# convolve input feature maps with filters
		conv_out = conv2d(
			input=input,
			filters=self.W,
			filter_shape=filter_shape,
			input_shape=image_shape
		)

		# pool each feature map individually, using maxpooling
		if poolsize[0] > 1:
			pooled_out = pool.pool_2d(
				input=conv_out,
				ds=poolsize,
				ignore_border=True
			)
		else:
			pooled_out = conv_out

		# dropout
		if dropout > 0.0:
			retain_prob = 1 - dropout
			pooled_out *= srng.binomial(pooled_out.shape, p=retain_prob, dtype=theano.config.floatX)
			pooled_out /= retain_prob

		# add the bias term. Since the bias is a vector (1D array), we first
		# reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
		# thus be broadcasted across mini-batches and feature map
		# width & height
		self.output = T.nnet.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

		# store parameters of this layer
		self.params = [self.W, self.b]

		# keep track of model input
		self.input = input

def convPoolLayerTest(input, filter_shape, image_shape, W, b,
					poolsize=(2, 2), activation=T.nnet.relu):
		
	assert image_shape[1] == filter_shape[1]

	# convolve input feature maps with filters
	conv_out = conv2d(
		input=input,
		filters=W,
		filter_shape=filter_shape,
		input_shape=image_shape
	)

	# pool each feature map individually, using maxpooling
	if poolsize[0] > 1:
		pooled_out = pool.pool_2d(
			input=conv_out,
			ds=poolsize,
			ignore_border=True
		)
	else:
		pooled_out = conv_out

	return T.nnet.relu(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))