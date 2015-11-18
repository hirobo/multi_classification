import sys, os, cv2
import numpy as np
from sklearn.externals import joblib
from lib.feature import HogFeature, HogSizeFeature
import config

RESULT_DIR = config.RESULT_DIR
SAVE_FAILURES = config.SAVE_FAILURES
#SAVE_FAILURES = True

if __name__ == "__main__":

	if len(sys.argv) < 4:
		print "USAGE: python test_classify.py <src(a file path or folder)> <label>"
		sys.exit(0)

	feature_type = config.FEATURE_TYPE
	feature = None

	src = sys.argv[1]
	label = int(sys.argv[2])

	classifier_file = config.PKL_CLASSIFIER_FILE
	params_file = config.PKL_PARAMS_FILE

	# load data for estimator
	try:
		clf = joblib.load(classifier_file)
		(mu, sigma) = joblib.load(params_file)
		print clf
	except Exception, e:			
		raise e
	else:
		pass

	if feature_type	== "HogFeature":
		feature = HogFeature(mu, sigma)
	else:
		feature = HogSizeFeature(mu, sigma)

	print "classify for label %d with %s"%(label, classifier_file)

	cnt = 0
	failure = 0
	if os.path.isdir(src):
		for filename in os.listdir(src):
			path = os.path.join(src, filename)
			img = cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
			if img is not None:
				feature_value = feature.calc(img)
				label_predict = int(clf.predict(feature_value))
				cnt += 1
				if label_predict is not label:
					failure += 1
					#print "%d:%s"%(label_predict, path)
					if SAVE_FAILURES is True:
						result_path = os.path.join(RESULT_DIR, "%d_%s"%(label_predict,filename))
						cv2.imwrite(result_path, img)

		score = (cnt - failure)* 100/cnt
		print "score: %d (cnt, failure) = (%d, %d)"%(score, cnt, failure)

	else:
		img = cv2.imread(src, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		if img is not None:
			feature_value = feature.calc(img)
			label_predict = int(clf.predict(feature_value))	
			print "%d:%s"%(label_predict, src)
