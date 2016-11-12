
import six.moves.cPickle as pickle

import numpy as np
import theano

def _make_shared(data, borrow=True):
	shared_data = theano.shared(np.asarray(data,
	 dtype=theano.config.floatX),
	borrow=borrow)
	return shared_data

class ParamDataManager:

	def getParamAddressBase(self):
		return str(self.image_shape + [self.channels] + self.nkerns + 
		self.filter_shapes + self.hidden_layer + [self.outputs] + self.pools + self.dropouts + 
		[self.momentum] + [self.learning_rate] + [self.n_epochs] + [self.batch_size])

	def getParamAddress(self):
			return "params/param_" + self.getParamAddressBase()


	def loadData(self, address=None):
		print "Loading parameters from " + address
		if address == None:
			address = self.getParamAddress()
		params = []
		paramData = np.load(address)
		shapes = paramData['shapes']
		data = paramData['data']
		for a in data:
			params = params + [_make_shared(a)]
		return params, shapes


	def saveData(self, params, address=None):
		if address == None:
			address = self.getParamAddress()
		print "Saving parameters to " + address
		shapes = []
		data = []
		for a in params:
			b = a.get_value()
			shapes = shapes + [b.shape]
			data = data + [b]
		paramData = {}
		paramData['data'] = data
		paramData['shapes'] = shapes
		file = open(address,'wb')
		pickle.dump(paramData, file, pickle.HIGHEST_PROTOCOL)
		file.close()
		print "Parameters saved in " + address


	def __init__(self, image_shape, channels, nkerns, filter_shapes, hidden_layer, outputs, pools, dropouts, momentum, learning_rate, n_epochs, batch_size):
		self.image_shape, self.channels, self.nkerns, self.filter_shapes, self.hidden_layer, self.outputs, self.pools, self.dropouts, self.momentum, self.learning_rate, self.n_epochs, self.batch_size = image_shape, channels, nkerns, filter_shapes, hidden_layer, outputs, pools, dropouts, momentum, learning_rate, n_epochs, batch_size


