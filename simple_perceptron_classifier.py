import numpy as np 
from sklearn.base import BaseEstimator

class Simple_Perceptron(BaseEstimator):
	def fit(self,input_samples,class_labels,**fit_params):
		weights=fit_params["weights"]
		self.d = fit_params["ndim"]
		self.weight_vec = np.zeros(self.d,float)
		self.bias = 0.0

		for epoch in xrange(1,fit_params["epochs"]+1):
			#print epoch,self.weight_vec,self.bias
			for i in xrange(0,len(input_samples)):
				discriminant = class_labels[i]*(np.dot(self.weight_vec,input_samples[i])+self.bias )
				if discriminant<=0:
					self.weight_vec = self.weight_vec + class_labels[i]*input_samples[i]*weights[i]
					self.bias = self.bias + class_labels[i]*weights[i]


	def predict(self,input_samples):
		'''Predicts class label corresponding to input sample'''
		return [ np.sign(self.weight_vec.dot(sample)+self.bias) for sample in input_samples ]

