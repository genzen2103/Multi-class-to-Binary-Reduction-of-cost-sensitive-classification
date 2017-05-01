from weighted_multiclass_NN import Weighted_FeedForward_MultiClass_NN,sigmoid_activation,unit_gain_activation
import numpy as np 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.datasets import load_breast_cancer
from loadData import load_zoo_data,load_yeast_data,load_vowel_data,load_glass_data,load_vehicle_data
from sklearn.preprocessing  import normalize


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

	print "Dataset loaded"
		
	X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=21)
		
	Features,labels,wts=[],[],[]

	temp = [ y_train[i] for i in xrange(len(y_train))]

	#cost_matrix = [ [ float( abs(i-j) ) for j in xrange(K) ] for i in xrange(K) ]
	cost_matrix = [ [ np.random.uniform( 0.0,2000.00 * ( temp.count(j)/float( temp.count(i) ) ) ) for j in xrange(K) ] for i in xrange(K) ]
	for i in range(K):
		cost_matrix[i][i]=0.0
	
	#print cost_matrix
	cost_matrix=[ [ cost_matrix[i][j]/200.0 for i in range(K) ] for j in range(K) ]
	#print cost_matrix


	sample_costs=[ [ cost_matrix[ y_train[i] ][j]  for j in xrange(K) ] for i in xrange(len(y_train)) ]

	max_cost=max([max(i) for i in cost_matrix])

	for i in xrange(len(y_train)):
		X,Y,W = [],[],[]
		for k in xrange(K):
			X.append(X_train[i])
			Y.append(y_train[i])
			W.append( max_cost-sample_costs[ y_train[i] ][k] + 1.0)
		Features.extend(X)
		labels.extend(Y)
		wts.extend(W)

	labels=[ [ 1 if labels[i]==k else 0 for k in range(K) ] for i in range(len(labels)) ]
	labels = np.matrix(labels)
	Features=np.matrix(Features)
	np.random.seed(1)

	n,d = Features.shape
	c = K
	nh = max(2,n/((d+c)*100))
	print "Samples:",n," Dimentions:",d," Hidden units:",nh," op units:",c

	clf = Weighted_FeedForward_MultiClass_NN()

	clf.add_Layer(d)

	clf.add_Layer(nh,'sigmoid')

	clf.add_Layer(c,'sigmoid')

	clf.compile()

	#print "Training Phase"

	clf.fit(Features,labels,epochs=1000,batch_size=K,tolerance=0.0001,learning_rate=0.0001,weights=np.matrix( [ [i] for i in wts] ) )

	print "Testing Phase"

	pred_labels = clf.predict(X_test)
	#print preds
	
	pred_labels = [ np.argmax(pred_labels[i]) for i in range(len(y_test))]
	#print preds
	
	acc=0.0
	sumcost=0.0
	full_matrix_max_cost=max([max(i) for i in cost_matrix])
	for i in range(len(pred_labels)):
		#max_cost = max(cost_matrix[y_test[i]]);
		miss_cost = cost_matrix[y_test[i]][pred_labels[i]]
		acc+=(full_matrix_max_cost-miss_cost)/float(full_matrix_max_cost)
		sumcost+=miss_cost;
		#print i,"misscost",miss_cost
		#print max_cost,(max_cost-miss_cost)/float(max_cost)

	print "CWNN Weighted Accuracy:",acc/float(len(pred_labels))
	print "CWNN Simple Accuracy:",sum([1 if pred_labels[i]==y_test[i] else 0  for i in range(len(pred_labels))])/float(len(pred_labels))
	mean_cost=sumcost/float( len(pred_labels) )
	print "CWNN Mean Cost:",mean_cost
