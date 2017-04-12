import numpy as np 
import math
import matplotlib.pyplot  as plt

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


class FeedForward_MultiClass_NN:
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

	def compile(self,loss_function='sqaure_error'):
		if len(self.network)<2:
			print "ERROR:FeedForward_MultiClass_NN:compile: Add more layers"
		for i in range(0,len(self.network)-1):
			#Weights initialised in [-1,1] interval
			self.weights.append(2*np.random.random((self.network[i][0],self.network[i+1][0]))-1)
		self.loss = loss_function

	def fit(self,features,outputs,**fit_params):
		''' Trains the NN using training samples expects epochs and batch size as added inputs'''
		n,d=features.shape
		#errors=[]
		for e in xrange( fit_params['epochs'] ):
			#Shuffle inputs
			perm = np.random.permutation(n)
			features = features[perm]
			outputs = outputs[perm]

			updates= [0.0]*len(self.weights)

			for b in xrange(0,n,fit_params['batch_size']):
			#for b in xrange(0,n):
				#Segment based on batch size
				if b+fit_params['batch_size']>=n:
					current_features = np.copy( features[b:n] )
					current_ops = np.copy( outputs[b:n] )
				else:
					current_features = np.copy( features[ b:b+fit_params['batch_size']  ] )
					current_ops = np.copy( outputs[ b:b+fit_params['batch_size']  ] )
				#current_features=features[b]
				#current_ops=outputs[b]

				# Forward Pass : 
				states= []
				states.append(current_features)
				for i in range (len(self.weights)):
					current_features =self.__activation_functions__[self.network[i+1][1]]( np.dot ( current_features , self.weights[i]) )
					states.append(current_features)


				error = current_ops-current_features

				# if abs( error.mean() ) < fit_params['tolerance']:
				# 	return
				#if (e%1000==0 and b==0):
					#print "Epoch %d : Error=%f" %( e, np.mean(np.abs(error)) )

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

			#print 'Epoch:%d Mean Error:%f'%(e,mean_err)
		#plt.plot(errors)
		#plt.show()

	def predict(self,features):
		current_features=np.copy(features)
		for i in range (len(self.weights)):
			current_features =self.__activation_functions__[self.network[i+1][1]]( np.dot ( current_features , self.weights[i]) )
		return np.matrix(current_features)

# X =  np.matrix ( [ 
# 	[0,0,6,16,6,0,0,0,0,2,16,14,15,0,0,0,0,9,13,0,14,3,0,0,0,4,6,1,16,5,0,0,0,0,0,8,16,3,0,0,0,0,3,16,14,0,2,0,0,0,13,16,16,16,15,0,0,0,7,14,12,12,12,1],
# 	[0,0,7,12,6,2,0,0,0,0,16,16,13,14,1,0,0,9,16,11,3,0,0,0,0,8,16,16,16,4,0,0,0,1,2,0,6,12,0,0,0,0,0,0,7,12,0,0,0,0,6,9,16,6,0,0,0,0,5,16,9,0,0,0],    
# 	[0,0,6,12,14,2,0,0,0,1,16,14,13,11,0,0,0,0,5,3,9,13,0,0,0,0,3,9,16,8,0,0,0,0,4,12,12,15,7,0,0,0,0,0,0,9,14,0,0,0,4,8,6,12,14,1,0,0,6,16,16,15,4,0]
# 	 ] ,float)

# y =  np.matrix ( [ [1,0,0],  [0,0,1] ,[ 0,1,0] ] ,float)


# np.random.seed(1)


# n,d = X.shape
# n,c = y.shape

# clf = FeedForward_MultiClass_NN()

# clf.add_Layer(d)
# clf.add_Layer(100,'sigmoid')
# clf.add_Layer(c,'sigmoid')

# clf.compile()

# clf.fit(X,y,epochs=60000,batch_size=2,tolerance=0.01)

# preds = np.around(clf.predict(X))
# print preds


