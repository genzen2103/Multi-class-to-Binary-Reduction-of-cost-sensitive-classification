import numpy as np 
import math


def sigmoid_activation(x,deriv=False):

	if deriv==True:
		return np.multiply(x,1.0-x)
	else:
		return (1.0 / (1.0 + np.exp(-x)))

def tanh_activation(x,deriv=False):
	if deriv==True:
		return (1.0- np.multiply(x,x))
	else:
		return np.tanh(x)

def unit_gain_activation(x):

	return x


class Weighted_FeedForward_MultiClass_NN:
	'''Implements Class FeedForward_MultiClass_NN'''
	__activation_functions__={
		'sigmoid':sigmoid_activation,
		'tanh':tanh_activation,
		'unit_gain':unit_gain_activation
	}

	def __init__(self):
		self.network = [] #Stores Dim and Activations for each layer
		self.weights = [] #Stores weight matrices between layers

	def add_Layer(self,layer_dim,layer_activation='unit_gain'):

		self.network.append([layer_dim,layer_activation])

	def compile(self):
		if len(self.network)<2:
			print "ERROR:FeedForward_MultiClass_NN:compile: Add more layers"
		for i in range(0,len(self.network)-1):
			#Weights initialised in [-1,1] interval
			self.weights.append(2*np.random.random((self.network[i][0],self.network[i+1][0]))-1)

	def fit(self,features,outputs,**fit_params):
		''' Trains the NN using training samples expects epochs and batch size as added inputs'''
		n,d=features.shape
		print n,d,outputs.shape,fit_params['weights'].shape
		cost_wts=fit_params['weights']
		#errors=[]
		for e in xrange( fit_params['epochs'] ):
			#Shuffle inputs
			perm = np.random.permutation(n)
			features = features[perm]
			outputs = outputs[perm]
			cost_wts = cost_wts[perm]

			updates= [0.0]*len(self.weights)

			for b in xrange(0,n,fit_params['batch_size']):
			#for b in xrange(0,n):
				#Segment based on batch size
				if b+fit_params['batch_size']>=n:
					current_features = np.copy( features[b:n] )
					current_ops = np.copy( outputs[b:n] )
					current_wts = np.copy( cost_wts[b:n] )
				else:
					current_features = np.copy( features[ b:b+fit_params['batch_size']  ] )
					current_ops = np.copy( outputs[ b:b+fit_params['batch_size']  ] )
					current_wts = np.copy( cost_wts[ b:b+fit_params['batch_size']  ] )


				# Forward Pass : 
				states= []
				states.append(current_features)
				for i in range (len(self.weights)):
					current_features =self.__activation_functions__[self.network[i+1][1]]( np.dot ( current_features , self.weights[i]) )
					states.append(current_features)


				error = (current_ops-current_features)

				if (e%500	==0 and b==0):
					print "Epoch %d : Error=%f" %( e, np.mean(np.abs(error)) )

				if ( np.mean(np.abs(error)) <= fit_params['tolerance'] ):
					return

				# BackProp :
				deltas=[]
				for i in range(len(self.network)-1,0,-1):
					delta = np.multiply( error, self.__activation_functions__[self.network[i][1]](states[i],True) )
					deltas.insert(0,delta)
					error=delta.dot(self.weights[i-1].T)	

				#  Save Updates 
				for i in range(0,len(self.weights)):
					updates[i]+= fit_params['learning_rate'] * states[i].T.dot(deltas[i])
			
			# Apply Updates
			for i in range(0,len(self.weights)):
				self.weights[i]+= updates[i]



	def predict(self,features):
		current_features=np.copy(features)
		for i in range (len(self.weights)):
			current_features =self.__activation_functions__[self.network[i+1][1]]( np.dot ( current_features , self.weights[i]) )
		return np.matrix(current_features)

