import numpy as np 
from sklearn.base import BaseEstimator
from binary_classifier import Binary_Classifier 
from collections import Counter
import cPickle
from multiprocessing import Pool


class CSOVO_Classifier(BaseEstimator):
	def __init__(self,clf_type):
		self.classifier_list=[]
		self.classifier_type=clf_type

	def fit(self,features,labels,**fit_params):
		for i in range(fit_params['numclass']):
			for j in range(fit_params['numclass']):
				if i<j:
					C,y,w=[],[],[]
					for k in xrange(len(labels)):
						if labels[k]==i or labels[k]==j:
							C.append(features[k])
							y.append(labels[k])
							w.append(fit_params['weights'][k])
					#print "Classifier",i,j
					clf =  Binary_Classifier(np.array(C),np.array(y),np.array(w),fit_params['epochs'],fit_params['cross_val_fold'])
					print "Classifier(%d,%d) Accuracy: %f"%(clf.classes[0],clf.classes[1],clf.accuracy)
					self.classifier_list.append(clf)

	def predict(self,input_samples):
		preds=[]
		for clf in self.classifier_list:
			res = clf.bcf.predict(input_samples)
			print "Classifier(%d,%d) Prediction done"%(clf.classes[0],clf.classes[1])
			map_res=[clf.classes[0] if res[i]==1 else clf.classes[1] for i in range(len(res)) ]
			preds.append(map_res)

		final_pred=[]
		for i in range(len(input_samples)):
			votes=[]
			for j in range(len(self.classifier_list)):
				votes.append(preds[j][i])
			count=Counter(votes)
			final_pred.append(count.most_common()[0][0])
		return final_pred

	def save(self,filename):
		with open(filename, 'wb') as fid:
			cPickle.dump(self.classifier_list, fid)  
	
	def load(self,filename):
		with open(filename, 'rb') as fid:
			self.classifier_list = cPickle.load(fid) 

