# for drawing learning curve
import sys, os
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
import config

if __name__ == "__main__":

	pkl_data_label_file = config.PKL_DATA_LABEL_FILE
	classifier_file = config.PKL_CLASSIFIER_FILE

	try:
		(data, labels) = joblib.load(pkl_data_label_file)
		clf = joblib.load(classifier_file)
	except Exception, e:			
		raise e
	else:
		pass


	# just check randomly 20% of the data set
	X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

	# calcurate confusion_matrix
	y_true = []
	y_pred = []	
	for X in X_test:
		pred = clf.predict(X)
		y_pred.append(int(pred)) 
	cnt = len(y_pred)
	y_true = y_test[:cnt]
	res = confusion_matrix(y_true, y_pred)

	print '='*50
	print '# Confusion matrix (vertical: actual, horizontal: prediction)'
	print '='*50

	print res