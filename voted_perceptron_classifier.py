import numpy as np 
from sklearn.base import BaseEstimator

class Voted_Perceptron(BaseEstimator):
	def fit(self,input_samples,class_labels,**fit_params):
		weights=fit_params["weights"]
		self.d = fit_params["ndim"]
		self.weight_vec = np.zeros(self.d,float)
		self.bias = 0.0
		self.survival_set=[]
		self.sucess_count=1

		for itr in xrange(1,fit_params["epochs"]+1):
			for i in xrange(0,len(input_samples)):
				discriminator = class_labels[i] * ( np.dot(self.weight_vec,input_samples[i]) + self.bias )
				if discriminator <=0:
					self.survival_set.append([self.weight_vec,self.bias,self.sucess_count])
					self.weight_vec = self.weight_vec + class_labels[i]*input_samples[i]*weights[i]
					self.bias = self.bias + class_labels[i]*weights[i]
					self.sucess_count=1
				else:
					self.sucess_count+=1
		else:
			self.survival_set.append([self.weight_vec,self.bias,self.sucess_count])


	def predict(self,input_samples):
		'''Predicts class label corresponding to input sample'''
		return [ np.sign( sum( [ item[2]*np.sign(np.dot(item[0],j)+item[1]) for item in self.survival_set ] ) ) for j in input_samples ]
