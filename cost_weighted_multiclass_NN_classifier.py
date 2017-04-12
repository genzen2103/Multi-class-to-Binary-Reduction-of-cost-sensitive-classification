from weighted_multiclass_NN import Weighted_FeedForward_MultiClass_NN,sigmoid_activation,unit_gain_activation
import numpy as np 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits


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
		
	X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.93, random_state=21)
		
	K,Features,labels,wts=len(dataset.target_names),[],[],[]

	print X_train.shape

	for i in range(len(y_train)):
		cost=[]
		for j in range(K):
			cost.append(float(abs(j-y_train[i])))
		f,l,w=costToWeight(X_train[i],y_train[i],cost)
		Features.extend(f)
		labels.extend(l)
		wts.extend(w)


	np.random.seed(1)

	n,d = X_train.shape
	c=1

	print "Samples:",n

	Accs =[]

	for nh  in range(8,9,1):

		clf = Weighted_FeedForward_MultiClass_NN()

		clf.add_Layer(d)

		clf.add_Layer(nh,'sigmoid')

		clf.add_Layer(c,'sigmoid')

		clf.compile()

		#print "Training Phase"

		clf.fit(np.matrix(Features),np.matrix([[i] for i in labels]),epochs=6000,batch_size=20,tolerance=0.0001,learning_rate=0.0001,weights=np.matrix( [ [i] for i in wts] ) )

		print "Testing Phase"

		preds = clf.predict(X_test)
			
		sr= sum([1 if preds[i]==y_test[i] else 0 for i in range(len(preds))])/float( len(preds))

		print "%d,%0.4f"%(nh,sr)

		Accs.append(sr)

	plt.plot(range(5,20,1),Accs)
	plt.show()




