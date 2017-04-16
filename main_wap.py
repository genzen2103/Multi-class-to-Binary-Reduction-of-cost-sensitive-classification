import numpy as np 
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from wap_classifier import WAP_Classifier
import os.path


def compute_v_cost_wap(cost, K):
	V=[0.0]*K
	costNow = cost
	vNow = [0.0]*K
	sorted(costNow)

	for i in xrange(K-1,0,-1):
		vNow[i] = (costNow[i] - costNow[i-1])*1.0

		vNow[0] = 0

	for i in xrange(1,K):
		vNow[i] /= i
		vNow[i] += vNow[i-1]

	for i in xrange(K):
		for k in xrange(K):
			if cost[i] == costNow[k]:
				V[i] = vNow[k]
				break

	return V;


if __name__=="__main__":

	
	dataset=load_digits()

	print "Digits dataset loaded"


	X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.25, random_state=21)
	
	K=len(dataset.target_names)
	
	temp = [ y_train[i] for i in xrange(len(y_train))]

	#cost_matrix = [ [ float( abs(i-j) ) for j in xrange(K) ] for i in xrange(K) ]
	cost_matrix = [ [ np.random.uniform( 0.0,2000.00 * ( temp.count(j)/float( temp.count(i) ) ) ) for j in xrange(K) ] for i in xrange(K) ]
	for i in range(K):
		cost_matrix[i][i]=0.0


	filename='csovo_model.pkl'

	clf_obj = WAP_Classifier('Simple_Perceptron')

	#Un comment when testing
	if os.path.isfile(filename):
	 	os.remove(filename)

	if os.path.isfile(filename):
		clf_obj.load(filename)
		print 'Model loaded from file ',filename
	else:
		sample_costs=[]
		sample_vcosts=[]
		for i in range(len(y_train)):
			cost=[]
			for j in range(K):
				cost.append(cost_matrix[ y_train[i] ][j] )		
			sample_costs.append(cost)
			sample_vcosts.append(compute_v_cost_wap(cost, K))


		print "Training Phase: Samples",len(y_train)

		clf_obj.fit(X_train,y_train,costs=sample_costs,vcosts=sample_vcosts,numclass=K,epochs=50,cross_val_fold=10)

		clf_obj.save(filename)

		print "Model Saved to ",filename


	pred_labels=clf_obj.predict(X_test)
	acc=0.0
	max_cost=max([max(i) for i in cost_matrix])
	for i in range(len(pred_labels)):
		#max_cost = max(cost_matrix[y_test[i]])
		miss_cost = cost_matrix[y_test[i]][pred_labels[i]]
		acc+=(max_cost-miss_cost)/float(max_cost)
		#print max_cost,miss_cost,(max_cost-miss_cost),(max_cost-miss_cost)/float(max_cost)
	print "WAP Weighted Accuracy:",acc/float(len(pred_labels))
	print "WAP Simple Accuracy:",sum([1 if pred_labels[i]==y_test[i] else 0  for i in range(len(pred_labels))])/float(len(pred_labels))