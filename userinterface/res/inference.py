## EBPPvML Resource Files Version 1.0

import os, cv2
from res.constants import IMAGE_SIZE, timeit
import res.image_preprocessor as ip
import res.fully_convolutional_network_quadout as fcn
import tensorflow as tf
import numpy as np

def init(weights_file):
	global model
	model = fcn.FCNNetwork()
	model.load_weights(weights_file)

def getFilename(layer_id):
	return 'image'+str(layer_id)+'.bmp'

@timeit
def predict(directory, layer_id):
	filename = getFilename(layer_id)
	full_filename = os.path.join(directory, filename)
	new_filename = os.path.join(directory, 'proc_' + filename)
	result_filename = os.path.join(directory, 'results/segm_' + filename)

	# Load and preprocess image
	image = ip.load_single(full_filename)
	image = image.reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 1))

	# Predict
	prediction = model.predict(image)
	prediction = prediction[0, :, :, :]
	prediction *= 255.0

	# Store result
	#cv2.imwrite(result_filename, prediction)

	# Rename input file
	#os.rename(full_filename, new_filename)

	return prediction

@timeit
def predict_batch(directory, batch):
	# Load and preprocess image
	images = np.empty(shape=(len(batch), IMAGE_SIZE, IMAGE_SIZE, 1))
	for index, layer_id in enumerate(batch):
		file = getFilename(layer_id)
		full_filename = os.path.join(directory, file)
		images[index] = ip.load_single(full_filename).reshape((IMAGE_SIZE, IMAGE_SIZE, 1))
	images = images.reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 1))

	# Predict
	predictions = model.predict(images)
	predictions *= 255.0

	# Store result
	for index, file in enumerate(batch):
		pass
		# file = getFilename(batch)
		#cv2.imwrite(os.path.join(directory, 'results/segm_'+file), predictions[index,:,:,0:3])
		# Rename input file
		#os.rename(os.path.join(directory, file), os.path.join(directory, 'proc_'+file))

	return predictions