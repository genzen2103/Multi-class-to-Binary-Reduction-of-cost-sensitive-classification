import numpy as np 
from sklearn.base import BaseEstimator
from numpy import linalg

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))
	
class Kernel_Perceptron(BaseEstimator):

	def __init__(self, kernel=gaussian_kernel, T=1):
		self.kernel = kernel
		self.T = T

	def fit(self, X, y,**fit_params):
		n_samples, n_features = X.shape
		self.sigma = fit_params['sigma']
		self.alpha = np.zeros(n_samples, dtype=np.float64)

		# Gram matrix
		K = np.zeros((n_samples, n_samples))
		for i in range(n_samples):
			for j in range(n_samples):
				K[i,j] = self.kernel(X[i], X[j],self.sigma)

		for t in range(self.T):
			for i in range(n_samples):
				if np.sign(np.sum(K[:,i] * self.alpha * y)) != y[i]:
					self.alpha[i] += 1.0

		for i  in range(n_samples):
			self.alpha[i]*=fit_params['weights'][i]
		# support vectors
		sv = self.alpha > 1e-5
		ind = np.arange(len(self.alpha))[sv]
		self.alpha = self.alpha[sv]
		self.sv = X[sv]
		self.sv_y = y[sv]


	def project(self, X):
		y_predict = np.zeros(len(X))
		for i in range(len(X)):
			s = 0
			for a, sv_y, sv in zip(self.alpha, self.sv_y, self.sv):
				s += a * sv_y * self.kernel(X[i], sv,self.sigma)
			y_predict[i] = s
		return y_predict

	def predict(self, X):
		X = np.atleast_2d(X)
		n_samples, n_features = X.shape
		return np.sign(self.project(X))


