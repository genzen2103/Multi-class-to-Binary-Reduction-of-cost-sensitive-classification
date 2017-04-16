import numpy as np 
from sklearn.base import BaseEstimator

class Simple_Perceptron(BaseEstimator):
	def fit(self,X,y,**fit_params):
		weights=fit_params["weights"]
		self.d = fit_params["ndim"]
		self.weight_vec = np.zeros(self.d,float)
		self.bias = 0.0
		self.iterations=0
		self.convergence_at_epoch=0

		for epoch in xrange(1,fit_params["epochs"]+1):
			wrong_samples = len(X)
			for i in xrange(0,len(X)):
				discriminant = y[i]*(np.dot(self.weight_vec,X[i])+self.bias )
				if discriminant<=0:
					self.weight_vec = self.weight_vec + y[i]*X[i]*weights[i]
					self.bias = self.bias + y[i]*weights[i]
					self.iterations+=1
				else:
					wrong_samples-=1
			else:
				if wrong_samples==0:
					self.convergence_at_epoch=epoch
					break
		self.W=self.weight_vec
		self.b=self.bias


	def predict(self,input_samples):
		'''Predicts class label corresponding to input sample'''
		return [ np.sign(self.weight_vec.dot(sample)+self.bias) for sample in input_samples ]

