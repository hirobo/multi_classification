import sys, os, cv2
import numpy as np
from skimage.feature import hog
from sklearn.externals import joblib
from sklearn import preprocessing

SIZE = 64
HOG_PARAMS = dict(
	orientations=8, 
	pixels_per_cell=(8, 8)
)

class Feature(object):

	def __init__(self, mu = None, sigma = None):
		self.feature_data_size = None
		self.mu = mu
		self.sigma = sigma

	def calc(self, img):
		pass

	def create_data_labels(self, class_dict, limit_num, pkl_data_label_path = None, pkl_params_path = None):
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
		# normalize
		# scaler = preprocessing.StandardScaler().fit(data)
		# data_norm = scaler.transform(data)

		# normalize
		mu = np.mean(data, axis=0);
		sigma = np.std(data, axis=0);
		(m, n) = data.shape
		data_norm = data - np.dot(np.ones((m, n)), np.diag(mu)) # normalise by mean 
		data_norm = np.dot(data_norm, np.linalg.inv(np.diag(sigma))) # normalise by std

		if pkl_data_label_path is None:			
			return data_norm, labels
			#return data_norm, labels
		else:
			print "dump as pkl file: %s"%pkl_data_label_path
			print "dump as pkl file: %s"%pkl_params_path

			joblib.dump((data_norm, labels), pkl_data_label_path, compress=3)
			joblib.dump((mu, sigma), pkl_params_path, compress=3)
			return data_norm, labels
			#joblib.dump((data_norm, labels), pkl_data_label_path, compress=3)
			#return data_norm, labels


class HogFeature(Feature):

	def __init__(self, mu = None, sigma = None):
		Feature.__init__(self, mu, sigma)
		hog_data_size = HOG_PARAMS['orientations']*HOG_PARAMS['pixels_per_cell'][0]*HOG_PARAMS['pixels_per_cell'][1]
		self.feature_data_size = hog_data_size


	def calc(self, img):
		(height, width) = img.shape[:2]
		s_img = cv2.resize(img,(SIZE, SIZE))
		fd, hog_image = hog(s_img, orientations=HOG_PARAMS['orientations'], pixels_per_cell=HOG_PARAMS['pixels_per_cell'],
							cells_per_block=(1, 1), visualise=True)
		return fd

class HogSizeFeature(Feature):

	def __init__(self, mu = None, sigma = None):
		Feature.__init__(self, mu, sigma)
		hog_data_size = HOG_PARAMS['orientations']*HOG_PARAMS['pixels_per_cell'][0]*HOG_PARAMS['pixels_per_cell'][1]
		additional_data_size = 1
		self.feature_data_size = hog_data_size + additional_data_size
		self.mu = np.zeros(self.feature_data_size)
		self.sigma = np.ones(self.feature_data_size)

	def calc(self, img):
		(height, width) = img.shape[:2]
		s_img = cv2.resize(img,(SIZE, SIZE))
		fd, hog_image = hog(s_img, orientations=HOG_PARAMS['orientations'], pixels_per_cell=HOG_PARAMS['pixels_per_cell'],
							cells_per_block=(1, 1), visualise=True)
		ratio = 1.0*width/height
		#return np.append(fd, ratio)
		feature = np.append(fd, ratio)
		#return feature

		feature_norm = feature - self.mu # normalise by mean 
		feature_norm = feature_norm / self.sigma
		return feature_norm
