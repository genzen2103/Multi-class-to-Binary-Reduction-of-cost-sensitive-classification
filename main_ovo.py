import numpy as np 
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from csovo_classifier import CSOVO_Classifier
from loadData import load_zoo_data,load_yeast_data,load_vowel_data,load_vehicle_data,load_glass_data
import os.path
import math


if __name__=="__main__":

	print "\nChoose a dataset which you want to use:\n 1: Digits\n 2: Iris\n 3: breast-cancer\n 4: Vowels\n 5: Vehicles\n 6: Glass\n 7: Yeast\n 8: Zoo\n\n Enter your option: \n"
	dataset_option = input()
	if dataset_option == 1:
		dataset = load_digits()
		target = dataset.target
		data = dataset.data
		classes = dataset.target_names
		K = len(classes)
		temp = [target[i] for i in xrange(len(target))]
	elif dataset_option == 2:
		dataset = load_iris()
		target = dataset.target
		data = dataset.data
		classes = dataset.target_names
		K = len(classes)
		temp = [target[i] for i in xrange(len(target))]
	elif dataset_option == 3:
		dataset = load_breast_cancer()
		target = dataset.target
		data = dataset.data
		classes = dataset.target_names
		K = len(classes)
		temp = [target[i] for i in xrange(len(target))]
	elif dataset_option == 4:
		data, target, classes = load_vowel_data('vowel-context.data')
		K = len(classes.keys())
		temp = [target[i] for i in xrange(len(target))]
	elif dataset_option == 5:
		data, target, classes = load_vehicle_data('vehicle.dat')
		K = len(classes.keys())
		temp = [target[i] for i in xrange(len(target))]
	elif dataset_option == 6:
		data, target, classes = load_glass_data('glass.data')
		K = len(classes.keys())
		temp = [target[i] for i in xrange(len(target))]
	elif dataset_option == 7:
		data, target, classes = load_yeast_data('yeast.data')
		K = len(classes.keys())
		temp = [target[i] for i in xrange(len(target))]
	else:
		data, target, classes = load_zoo_data('zoo.data')
		K = len(classes.keys())
		temp = [target[i] for i in xrange(len(target))]

	cost_matrix = [ [ np.random.uniform( 0.0,2000.00 * ( temp.count(i)/float( temp.count(j) ) ) ) for j in xrange(K) ] for i in xrange(K) ]
	for i in range(K):
		cost_matrix[i][i]=0.0

	filename='ovo_model.pkl'

	clf_obj = CSOVO_Classifier('Simple_Perceptron')

	max_iterations = 20

	sacc_l,cacc_l,scost_l=[],[],[]

	for iteration in range(max_iterations):
		print "\nIteration:",iteration
		X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=iteration)

		#Un comment when testing
		if os.path.isfile(filename):
			os.remove(filename)

		if os.path.isfile(filename):
			clf_obj.load(filename)
			print 'Model loaded from file ',filename
		else:
			sample_costs=[ [ 0.0 if j==y_train[i] else 1.0 for j in xrange(K) ] for i in xrange(len(y_train)) ]
			print "\nTraining Phase: Samples",len(y_train)

			clf_obj.fit(X_train,y_train,costs=sample_costs,numclass=K,epochs=50,cross_val_fold=10)

			clf_obj.save(filename)

			#print "Model Saved to ",filename
		print "\nMaking Predictions"
		pred_labels=clf_obj.predict(X_test)
		full_matrix_max_cost=max([max(i) for i in cost_matrix])
		cost_acc = sum ( [ ( full_matrix_max_cost-cost_matrix[y_test[i]][pred_labels[i]] )/ float(full_matrix_max_cost) for i in xrange(len(pred_labels))  ] )/float(len(pred_labels))
		simp_acc = sum([1 if pred_labels[i]==y_test[i] else 0  for i in range(len(pred_labels))])/float(len(pred_labels))
		sum_cost = sum ( [ cost_matrix[y_test[i]][pred_labels[i]]  for i in xrange(len(pred_labels))  ] )/float( len(pred_labels) )
		print "\nStatistics:"
		print simp_acc,cost_acc,sum_cost

		sacc_l.append(simp_acc)
		cacc_l.append(cost_acc)
		scost_l.append(sum_cost)

	scost_l=np.array(scost_l)
	print "OVO Weighted Accuracy:",sum(cacc_l)/float(len(cacc_l))
	print "OVO Simple Accuracy:",sum(sacc_l)/float(len(sacc_l))
	print "OVO Mean Cost:",np.mean(scost_l)
	print "OVO Cost SD :",np.std(scost_l)