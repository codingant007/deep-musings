
__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


class LogisticRegression(object):
	
	def __init__(self, input, n_in, n_out, W=None, b=None):
		
		if W is None:
			W = theano.shared(
				value=numpy.zeros(
					(n_in, n_out),
					dtype=theano.config.floatX
				),
				name='W',
				borrow=True
			)
		# initialize the biases b as a vector of n_out 0s
		if b is None:
			b = theano.shared(
				value=numpy.zeros(
					(n_out,),
					dtype=theano.config.floatX
				),
				name='b',
				borrow=True
			)
		self.W = W
		self.b = b

		# symbolic expression for computing the matrix of class-membership
		# probabilities
		# Where:
		# W is a matrix where column-k represent the separation hyperplane for
		# class-k
		# x is a matrix where row-j  represents input training sample-j
		# b is a vector where element-k represent the free parameter of
		# hyperplane-k
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

		# symbolic description of how to compute prediction as class whose
		# probability is maximal
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)

		# parameters of the model
		self.params = [self.W, self.b]

		# keep track of model input
		self.input = input



	def get_probability_distribution(self):
		return self.p_y_given_x

	def negative_log_likelihood(self, y):
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

	def hinge_cost(self, y, delta=0.1, outputs=T.arange(26)):

		c = self.p_y_given_x[T.arange(y.shape[0]), y]
		p_y_given_x_temp = T.sort(self.p_y_given_x)
		a = p_y_given_x_temp[:,-1]
		b = p_y_given_x_temp[:,-2]
		cost = T.mean(T.maximum(0, T.sub(2*c, T.add(a,b)) + delta))
		return cost

def l2_regularization(l, W):
	return l * ((W**2).sum())



def l1_regularization(l,W):
	return l * (abs(W).sum())


def logisticRegressionTest(input, W, b):
	# Logistic layer
	p_y_given_x = T.nnet.softmax(T.dot(input, W) + b)
	return T.argmax(p_y_given_x, axis=1)



def classificationErrors(y_pred, y):
	# check if y has same dimension of y_pred
	if y.ndim != y_pred.ndim:
		raise TypeError(
			'y should have the same shape as self.y_pred',
			('y', y.type, 'y_pred', y_pred.type)
		)
	# check if y is of the correct datatype
	if y.dtype.startswith('int'):
		# the T.neq operator returns a vector of 0s and 1s, where 1
		# represents a mistake in prediction
		return T.mean(T.neq(y_pred, y))
	else:
		raise NotImplementedError()