import os
THIS_DIR = os.path.dirname(os.path.realpath(__file__))

CATEGORY_CLASS_DICT = {
	"data/examples/ballerinas": 0, 
	"data/examples/boots": 1, 
	"data/examples/heels": 2, 
	"data/examples/sandals": 3, 
	"data/examples/sneakers": 4
}

FEATURE_TYPE = "HogSizeFeature"
LIMIT_NUM = 300

#PKL_DATA_LABEL_FILE_DIR = "pkl/data_label"
#PKL_CLASSIFIER_FILE_DIR = "pkl/estimator"
PKL_DIR = "pkl"
PKL_DATA_LABEL_FILE = os.path.join(PKL_DIR, "estimator_%s.pkl"%FEATURE_TYPE)
PKL_CLASSIFIER_FILE = os.path.join(PKL_DIR, "data_label_%s.pkl"%FEATURE_TYPE)
PKL_PARAMS_FILE = os.path.join(PKL_DIR, "params_%s.pkl"%FEATURE_TYPE)

##########################
# just need for development 
##########################
# for test_classify.py
RESULT_DIR = "result"
SAVE_FAILURES = False