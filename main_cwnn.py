from weighted_multiclass_NN import Weighted_FeedForward_MultiClass_NN,sigmoid_activation,unit_gain_activation
import numpy as np 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing  import normalize


if __name__=="__main__":

	dataset=load_digits()

	print "Digits dataset loaded"
		
	X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.25, random_state=21)
		
	K,Features,labels,wts=len(dataset.target_names),[],[],[]

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
	nh,c = 20,K
	print "Samples:",n

	clf = Weighted_FeedForward_MultiClass_NN()

	clf.add_Layer(d)

	clf.add_Layer(nh,'sigmoid')

	clf.add_Layer(c,'sigmoid')

	clf.compile()

	#print "Training Phase"

	clf.fit(Features,labels,epochs=1000,batch_size=K,tolerance=0.0001,learning_rate=0.0001,weights=np.matrix( [ [i] for i in wts] ) )

	print "Testing Phase"

	preds = clf.predict(X_test)
	#print preds
	
	preds = [ np.argmax(preds[i]) for i in range(len(y_test))]
	#print preds
	
	acc=0.0
	for i in range(len(preds)):
		#print y_test[i],preds[i]
		miss_cost = cost_matrix[ y_test[i] ][ preds[i] ]
		acc+=(max_cost-miss_cost)/float(max_cost)
	print "CWNN Weighted Accuracy:",acc/float(len(preds))
	print "CWNN Simple Accuracy:",sum([1 if preds[i]==y_test[i] else 0  for i in range(len(preds))])/float(len(preds))




