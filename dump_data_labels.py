import sys, os
from lib.feature import HogFeature, HogSizeFeature
import config

if __name__ == "__main__":
	if len(sys.argv) < 1:
		print "USAGE: python dump_data_labels.py"
		sys.exit(0)

	feature_type = config.FEATURE_TYPE
	feature = None

	if feature_type == "HogFeature":
		feature = HogFeature()
	else:
		feature = HogSizeFeature()

	# dump dataset
	data, labels = feature.create_data_labels(
		class_dict = config.CATEGORY_CLASS_DICT, 
		limit_num = config.LIMIT_NUM, 
		pkl_data_label_path = config.PKL_DATA_LABEL_FILE,
		pkl_params_path = config.PKL_PARAMS_FILE)