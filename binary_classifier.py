import numpy as np 
from sklearn.model_selection import cross_val_score
from simple_perceptron_classifier import Simple_Perceptron 

class Binary_Classifier:
	def __init__(self,input_vectors,labels,wts,epoch,kcv):
		data_dim = len(input_vectors[0])	
		classes = np.unique(labels)
		if len(classes)!=2:
			print "ERROR:Binary_Classifier:init: Labels have non binary data"
			return
		class_labels = np.array([ 1 if labels[i]==classes[0] else -1 for i in range(len(labels))])	
		bcf = Simple_Perceptron()
		bcf.fit(input_vectors,class_labels,epochs=epoch,ndim=data_dim,weights=wts)
		scores = cross_val_score(estimator=bcf, X=input_vectors, y=class_labels, cv=kcv,scoring="accuracy",fit_params={"epochs":epoch,"ndim":data_dim,"weights":wts})
		#print("Accuracy: %f" % (scores.mean()))
		self.accuracy=scores.mean()
		self.classes=classes
		self.bcf=bcf

