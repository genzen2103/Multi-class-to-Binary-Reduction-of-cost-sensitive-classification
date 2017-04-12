from FeedForward_Muilticlass_NN import FeedForward_MultiClass_NN,sigmoid_activation,unit_gain_activation
from preprocess_data import extract_data_file,compress_image
import numpy as np 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

#print 'Extracting Training from File:'
digits=np.array([6,8,9])
X_train,y_train = extract_data_file('optdigits-orig.tra',digits,32,4)

#print 'Extracting Test Data:'
X_test,y_test = extract_data_file('optdigits-orig.test',digits,32,4)



np.random.seed(1)

n,d = X_train.shape
n,c = y_train.shape

print "Samples:",n
Accs =[]

for nh  in range(5,20,1):

	#print 'Creating Network (nh=%d):'%nh


	#nh = n//(7*(d+c))

	#print "n=%d nh=%d"%(n,nh)

	clf = FeedForward_MultiClass_NN()

	clf.add_Layer(d)

	clf.add_Layer(nh,'sigmoid')

	clf.add_Layer(c,'sigmoid')

	clf.compile()

	#print "Training Phase"

	clf.fit(X_train,y_train,epochs=6000,batch_size=20,tolerance=0.0001,learning_rate=0.0001)

	#print "Testing Phase"

	preds = clf.predict(X_test)

	labels=[]
	for i in preds:
		labels.append(digits[ np.argmax(i) ] )

	actual_labels = np.dot(y_test,digits).T


		
	sr= sum([1 if labels[i]==actual_labels[i] else 0 for i in range(len(actual_labels))])/float( len(actual_labels))

	#print "Acc=",sr

	print "%d,%0.4f"%(nh,sr)

	Accs.append(sr)

plt.plot(range(5,20,1),Accs)
plt.show()




