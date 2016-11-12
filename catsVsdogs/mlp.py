"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

	f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

	- textbooks: "Pattern Recognition and Machine Learning" -
				 Christopher M. Bishop, section 5

"""

from __future__ import print_function

__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


from logistic_regression import LogisticRegression


# Train hidden layer
class HiddenLayer(object):
	def __init__(self, rng, input, n_in, n_out, W=None, b=None,
				 activation=T.nnet.relu, dropout=0):


		self.input = input

		if W is None:
			W_values = numpy.asarray(
				rng.uniform(
					low=-numpy.sqrt(6. / (n_in + n_out)),
					high=numpy.sqrt(6. / (n_in + n_out)),
					size=(n_in, n_out)
				),
				dtype=theano.config.floatX
			)
			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4

			W = theano.shared(value=W_values, name='W', borrow=True)

		if b is None:
			b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)

		self.W = W
		self.b = b

		lin_output = T.dot(input, self.W) + self.b
		out = (
			lin_output if activation is None
			else activation(lin_output)
		)

		# Applying dropout
		if dropout > 0.0:
			retain_prob = 1 - dropout
			srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
			out *= srng.binomial(size=out.shape, p=retain_prob, dtype=theano.config.floatX, n=1)
			#out /= retain_prob

		self.output = out
		# parameters of the model
		self.params = [self.W, self.b]

# Test hidden layer
def hiddenLayerTest(input, W, b, activation=T.nnet.relu):

	lin_output = T.dot(input, W) + b
	out = (
		lin_output if activation is None
		else activation(lin_output)
	)

	return out