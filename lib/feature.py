import sys, os, cv2
import numpy as np
from skimage.feature import hog
from sklearn.externals import joblib

SIZE = 64
HOG_PARAMS = dict(
	orientations=8, 
	pixels_per_cell=(8, 8)
)

class Feature(object):

	def __init__(self):
		self.feature_data_size = None

	def calc(self, img):
		pass

	def create_data_labels(self, class_dict, limit_num, pkl_data_label_path = None):
		if self.feature_data_size is None:
			raise NotImplementedError("Feature is super class.")

		class_num = len(class_dict)
		labels_length = limit_num*class_num
		data = np.zeros((labels_length, self.feature_data_size))
		labels = np.zeros((labels_length))
		num = 0

		for class_dir, label in class_dict.iteritems():
			print "%d:%s"%(label, class_dir)
			cnt = 0
			for filename in os.listdir(class_dir):
				path = os.path.join(class_dir, filename)
				img = cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
				if img is not None:
					data[num] = self.calc(img)
					labels[num] = label
					num += 1
					cnt += 1
				if cnt >= limit_num:
					break
		if pkl_data_label_path is None:			
			return data, labels
		else:
			print "dump as pkl file: %s"%pkl_data_label_path		
			joblib.dump((data, labels), pkl_data_label_path, compress=3)
			return data, labels


class HogFeature(Feature):

	def __init__(self):
		Feature.__init__(self)
		hog_data_size = HOG_PARAMS['orientations']*HOG_PARAMS['pixels_per_cell'][0]*HOG_PARAMS['pixels_per_cell'][1]
		self.feature_data_size = hog_data_size


	def calc(self, img):
		(height, width) = img.shape[:2]
		s_img = cv2.resize(img,(SIZE, SIZE))
		fd, hog_image = hog(s_img, orientations=HOG_PARAMS['orientations'], pixels_per_cell=HOG_PARAMS['pixels_per_cell'],
							cells_per_block=(1, 1), visualise=True)
		return fd

class HogSizeFeature(Feature):

	def __init__(self):
		Feature.__init__(self)
		hog_data_size = HOG_PARAMS['orientations']*HOG_PARAMS['pixels_per_cell'][0]*HOG_PARAMS['pixels_per_cell'][1]
		additional_data_size = 1
		self.feature_data_size = hog_data_size + additional_data_size

	def calc(self, img):
		(height, width) = img.shape[:2]
		s_img = cv2.resize(img,(SIZE, SIZE))
		fd, hog_image = hog(s_img, orientations=HOG_PARAMS['orientations'], pixels_per_cell=HOG_PARAMS['pixels_per_cell'],
							cells_per_block=(1, 1), visualise=True)
		ratio = 1.0*width/height
		return np.append(fd, ratio)
