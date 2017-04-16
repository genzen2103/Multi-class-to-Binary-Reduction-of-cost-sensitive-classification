import numpy as np 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from csovo_classifier import CSOVO_Classifier
import os.path

def argmin(i,j,Ci,Cj):
    if Ci<Cj:
        return i;
    else:
        return j; 

def integration_param(minm, C_i, C):
    Range = np.arange(minm, C_i, 0.2)
    sum = 0

    for t in Range:
        count = 0

        for c in C:
            if c <= t:
                count += 1

        sum += 1.0/count
    return sum

def weightedAllPairs(X, Y, C):
    noOfClasses = len(C)
    returnList = []
    xList, labelList, weightList = [], [], []

    for i in range(noOfClasses):

        for j in range(noOfClasses):

            if(j > i):
                minm = min(C)
                xList.append(X)
                labelList.append(argmin(i,j,C[i],C[j]))
                weightList.append(abs(integration_param(minm, C[i], C) - integration_param(minm, C[j], C)))

    return [xList, labelList, weightList]


def costToWeight(X,Y,C):
	#print C
	k,xl,ll,wl=len(C),[],[],[]

	for i in range(k):
		for j in range(k):
			if i<j:
				xl.append(X)
				if C[i]<C[j]:
					ll.append(i)
				else:
					ll.append(j)
				wl.append( abs(C[i]-C[j]) )
	return [xl,ll,wl]


if __name__=="__main__":

	
	dataset=load_digits()

	print "Digits dataset loaded"
	
	X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.80, random_state=21)
	
	K,Features,labels,wts=len(dataset.target_names),[],[],[]

	print X_train.shape

	filename='csovo_model.pkl'

	clf_obj = CSOVO_Classifier('Simple_Perceptron')

	#Un comment when testing
	# if os.path.isfile(filename):
	# 	os.remove(filename)

	if os.path.isfile(filename):
		clf_obj.load(filename)
		print 'Model loaded from file ',filename
	else:

		for i in range(len(y_train)):
			#cost=np.random.random(K)
			#cost=np.ones(K)
			cost=[]
			for j in range(K):
				cost.append(float(abs(j-y_train[i])))
			f,l,w=costToWeight(X_train[i],y_train[i],cost)
			Features.extend(f)
			labels.extend(l)
			wts.extend(w)

		print "Training Phase: Samples",len(labels)

		clf_obj.fit(Features,labels,weights=wts,numclass=K,epochs=25,cross_val_fold=10)

		clf_obj.save(filename)

		print "Model Saved to ",filename


	pred_labels=clf_obj.predict(X_test)

	print "\nCSOVO Accuracy: %f"%(sum([1 if pred_labels[i]==y_test[i] else 0 for i in range(len(pred_labels))] )/float(len(pred_labels)) )





	



		
