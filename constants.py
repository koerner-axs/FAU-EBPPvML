IMAGE_SIZE = 512
BASE_DIR = 'F:\\Machine Learning\\FAU - EBPPvML\\'

# Controls for the gradient vectorizer algorithm.
GV_A_KERNEL_SIZE = 8
GV_A_STRIDE = 4
GV_B_KERNEL_SIZE = 7
GV_B_STRIDE = 1

# This controls the effect strength of the brightness mitigation
# algorithm in the image preprocessor module. Adjust only after
# a certain new value was tested on a lot of images with high
# variance.
IP_BGM_EFFECT_STRENGTH = 0.15
#IP_BGM_EFFECT_STRENGTH = 0.05

# An enum over the possible classifications for each pixel.
# Do not change.
SEGM_FLT_BCKGRND = 0
SEGM_FLT_POROUS = 1
SEGM_FLT_BULDGE = 2
SEGM_FLT_GOODLYR = 3
# The certainty level which has to be reached by the segmenter for
# the fault at any pixel to be accepted. This sets a lower bound
# to the predicted values. Increase to filter more for noise.
# Decrease to allow for higher sensitivity.
SEGM_FLT_DELTA_THRHLD = 0.05
SEGM_FLT_MIN_ABS_THRHLD = 0.6


import numpy as np
import h5py
import os


def hdf5_get(dsetname, file):
	with h5py.File(file, 'r') as f:
		data = f[dsetname].value
	return data

# Deprecated
import pickle
def _get_pickle(file):
	with open(file, 'rb') as file:
		data = pickle.load(file)
	return data

# Gathering point for all the dataset loading needs of all modules.
def get_data(name):
	assert isinstance(name, str)

	if name == 'SEGM_INP':
		return _get_pickle('F:\\Machine Learning\\FAU - Image Classification\\datasets\\segmentation\\Labeled Dataset 1\\inputs\\trainInputs.pickle')
	if name == 'SEGM_LBL':
		return _get_pickle('F:\\Machine Learning\\FAU - Image Classification\\datasets\\segmentation\\Labeled Dataset 1\\labels\\trainLabels.pickle')
	if name == 'SEGM_EXT1_INP':
		return _get_pickle('F:\\Machine Learning\\FAU - Image Classification\\datasets\\segmentation\\Labeled Dataset 2\\inputs\\trainInputs.pickle')
	if name == 'SEGM_EXT1_LBL':
		return _get_pickle('F:\\Machine Learning\\FAU - Image Classification\\datasets\\segmentation\\Labeled Dataset 2\\labels\\trainLabels.pickle')
	if name == 'SEGM_EXT2_INP':
		return hdf5_get('inputs', 'F:\\Machine Learning\\FAU - Image Classification\\datasets\\segmentation\\Labeled Dataset 3\\inputs\\trainInputs.hdf5')
	if name == 'SEGM_EXT2_LBL':
		return hdf5_get('labels', 'F:\\Machine Learning\\FAU - Image Classification\\datasets\\segmentation\\Labeled Dataset 3\\labels\\trainLabels.hdf5')
	if name == 'SEGM_EXT3_INP':
		return hdf5_get('inputs', 'F:\\Machine Learning\\FAU - Image Classification\\datasets\\segmentation\\Labeled Dataset 4\\inputs\\trainInputs.hdf5')
	if name == 'SEGM_EXT3_LBL':
		return hdf5_get('labels', 'F:\\Machine Learning\\FAU - Image Classification\\datasets\\segmentation\\Labeled Dataset 4\\labels\\trainLabels.hdf5')
	if name == 'SEGM_TST1_INP':
		return hdf5_get('inputs', 'F:\\Machine Learning\\FAU - Image Classification\\datasets\\segmentation\\Test Dataset\\inputs\\testInputs.hdf5')

	raise ValueError('The dataset with the name ' + name + ' does not exist.')


# For testing only...
if __name__=='__main__':
	# d = get_data('SEGM_LBL')
	# import cv2
	# for dx in d[0:5]:
	# 	cv2.imshow('lul', dx)
	# 	cv2.waitKey(0)
	# 	cv2.destroyAllWindows()

	img = get_data('SEGM_TST1_INP')[0]
	print(img.shape)
	import gradient_vectorizer_gpu as prep
	prep.compute_gradient_vector(img)