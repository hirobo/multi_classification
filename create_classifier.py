import sys, os, cv2
from sklearn.externals import joblib
import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn import cross_validation, grid_search
from sklearn.metrics import classification_report
from lib.utils import mkdir_p
from lib.feature import HogFeature, HogSizeFeature
import config


if __name__ == "__main__":
	if len(sys.argv) < 1:
		print "USAGE: python create_classifier.py"
		sys.exit(0)

	feature_type = config.FEATURE_TYPE
	feature = None
	pkl_estimator_file = config.PKL_CLASSIFIER_FILE
	pkl_data_label_file = config.PKL_DATA_LABEL_FILE	
	algorithm_type = "svm" # use random forest?

	# load data label file and params file
	try:
		(data, labels) = joblib.load(pkl_data_label_file)
		(mu, sigma) = joblib.load(pkl_data_label_file)
	except Exception, e:			
		raise e
	else:
		pass

	# create Feature Class instance	
	if feature_type == "HogFeature":
		feature = HogFeature(mu, sigma)
	else:
		feature = HogSizeFeature(mu, sigma)


	# separate data to train set and test set(8:2 as train:test))
	X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)

	# Find the best parameters in this range.
	tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
	                     'C': [1, 10, 100, 1000]},
	                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

	print '='*50
	print("# Tuning hyper-parameters for F1-score")
	print '='*50

	clf = grid_search.GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5,
					   scoring='f1_weighted')
	clf.fit(X_train, y_train)

	print("Best parameters set found on development set:")
	print(clf.best_params_)

	print("Grid scores on development set:")

	for params, mean_score, scores in clf.grid_scores_:
		print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))


	print("Detailed classification report:")
	print("The model is trained on the full development set.")
	print("The scores are computed on the full evaluation set.")
	y_true, y_pred = y_test, clf.predict(X_test)
	print(classification_report(y_true, y_pred))

	# use the best estimator as classifier
	clf = clf.best_estimator_

	# dump as file
	s = joblib.dump(clf, pkl_estimator_file, compress=3)