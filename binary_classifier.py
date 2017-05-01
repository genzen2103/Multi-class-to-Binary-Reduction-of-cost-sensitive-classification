import numpy as np 
from sklearn import svm
from sklearn.model_selection import cross_val_score
from simple_perceptron_classifier import Simple_Perceptron 
from voted_perceptron_classifier import Voted_Perceptron 
from kernel_perceptron_classifier import Kernel_Perceptron,gaussian_kernel

class Binary_Classifier:
	def __init__(self,input_vectors,labels,wts,epoch,kcv):
		data_dim = len(input_vectors[0])	
		classes = np.unique(labels)
		if len(classes)!=2:
			print "ERROR:Binary_Classifier:init: Labels have non binary data"
			return
		class_labels = np.array([ 1 if labels[i]==classes[0] else -1 for i in range(len(labels))])	

		#Voted
		#bcf = Voted_Perceptron()
		#bcf.fit(input_vectors,class_labels,epochs=epoch,ndim=data_dim,weights=wts)
		# scores = cross_val_score(estimator=bcf, X=input_vectors, y=class_labels, cv=kcv,scoring="accuracy",fit_params={"weights":wts})
		# self.accuracy=scores.mean()
		
		#Simple
		bcf = Simple_Perceptron()
		bcf.fit(input_vectors,class_labels,epochs=epoch,ndim=data_dim,weights=wts)
		#scores = cross_val_score(estimator=bcf, X=input_vectors, y=class_labels, cv=kcv,scoring="accuracy",fit_params={"weights":wts,"ndim":data_dim,"epochs":epoch})
		#self.accuracy=scores.mean()

		#Kernel
		#bcf=Kernel_Perceptron(kernel=gaussian_kernel,T=epoch )
		#bcf.fit(input_vectors,labels,weights=wts,sigma=5.0)
		
		#SVM
		#bcf = svm.SVC()
		#bcf.fit(input_vectors,class_labels,sample_weight=wts)
		

		self.classes=classes
		self.bcf=bcf


