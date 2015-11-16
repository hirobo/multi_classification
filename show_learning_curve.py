# for drawing learning curve
import sys, os
import numpy as np
from sklearn.externals import joblib
from sklearn.learning_curve import learning_curve
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

if __name__ == "__main__":

	pkl_data_label_file = sys.argv[1]
	classifier_file = sys.argv[2]
	result_file = sys.argv[3]

	try:
		(data, labels) = joblib.load(pkl_data_label_file)
		clf = joblib.load(classifier_file)
	except Exception, e:			
		raise e
	else:
		pass

	# currently this process will not end(?) so, I will comment out this term.	
	# print "draw learning curve"
	# train_sizes, train_scores, valid_scores = learning_curve(
	# 	clf, data, labels, train_sizes=np.linspace(0.1, 1.0, 5), cv=5)	
	# fig = plt.figure()	
	# plt.plot(train_sizes, train_scores.mean(axis=1), label="training scores")
	# plt.plot(train_sizes, valid_scores.mean(axis=1), label="test scores")
	# plt.legend(loc="best")
	# plt.show()
	# fig.savefig(result_file)

