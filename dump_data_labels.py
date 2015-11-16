import sys, os
from lib.feature import HogFeature, HogSizeFeature
import config

if __name__ == "__main__":
	if len(sys.argv) < 1:
		print "USAGE: python dump_data_labels.py <option: feature_type(HogFeature or HogSizeFeature)"
		sys.exit(0)

	feature_type = config.FEATURE_TYPE
	feature = None

	if len(sys.argv) >= 2:
		feature_type = sys.argv[1]

	if feature_type == "HogFeature":
		feature = HogFeature()
	else:
		feature = HogSizeFeature()

	# dump dataset
	file_name = "data_label_%s.pkl"%feature.__class__.__name__
	pkl_data_label_path = os.path.join(config.PKL_DATA_LABEL_FILE_DIR, file_name)	
	data, labels = feature.create_data_labels(
		class_dict = config.CATEGORY_CLASS_DICT, 
		limit_num = config.LIMIT_NUM, 
		pkl_data_label_path = pkl_data_label_path)