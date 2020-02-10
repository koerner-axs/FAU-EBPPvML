import numpy as np
import os
import cv2
import random
import pickle

import h5py

from constants import IMAGE_SIZE

def flip_and_rotate(image, numChannels, targetList):
	image = np.array(image).reshape(IMAGE_SIZE, IMAGE_SIZE, numChannels)

	currentList = [image, np.fliplr(image),
				   np.rot90(image), np.fliplr(np.rot90(image)),
				   np.rot90(image, k=2), np.fliplr(np.rot90(image, k=2)),
				   np.rot90(image, k=3), np.fliplr(np.rot90(image, k=3))]

	targetList.extend(currentList)

# Load the labels.
if True:
	directory = 'F:\\Machine Learning\\FAU - Image Classification\\datasets\\segmentation\\Labeled Dataset 4\\labels\\'

	print('Loading labels from directory: ' + directory)
	count = 0
	labels = list()
	dirFiles = list(os.listdir(directory))
	dirFiles = list(filter(lambda x: x.endswith('.bmp'), dirFiles))
	dirFiles.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
	for dirEntry in dirFiles:
		print(dirEntry)
		try:
			image = cv2.imread(os.path.join(directory, dirEntry), cv2.IMREAD_COLOR)
			image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

			#labels.append(image)
			flip_and_rotate(image, 3, labels)
		except Exception as e:
			print('Skipping file due to exception thrown: ' + dirEntry)
			print('Due to: ' + str(e))
			continue

		count += 1

	print('Loaded {0} label images!'.format(count))

	labels = np.array(labels).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
	labels = labels / 255.0

	hdf5Path = os.path.join(directory, 'trainLabels.hdf5')
	print('Writing label images to: ' + hdf5Path)

	with h5py.File(hdf5Path, 'w') as f:
		dset = f.create_dataset('labels', data=labels)


# Load the inputs
if True:
	directory = 'F:\\Machine Learning\\FAU - Image Classification\\datasets\\segmentation\\Labeled Dataset 4\\inputs\\'

	print('Loading inputs from directory: ' + directory)
	count = 0
	inputs = list()
	dirFiles = list(os.listdir(directory))
	dirFiles = list(filter(lambda x: x.endswith('.bmp'), dirFiles))
	dirFiles.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
	for dirEntry in dirFiles:
		print(dirEntry)
		try:
			image = cv2.imread(os.path.join(directory, dirEntry), cv2.IMREAD_GRAYSCALE)
			image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

			#inputs.append(image)
			flip_and_rotate(image, 1, inputs)
		except Exception as e:
			print('Skipping file due to exception thrown: ' + dirEntry)
			print('Due to: ' + str(e))
			continue

		count += 1

	print('Loaded {0} input images!'.format(count))

	inputs = np.array(inputs).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
	inputs = inputs / 255.0

	hdf5Path = os.path.join(directory, 'trainInputs.hdf5')
	print('Writing input images to: ' + hdf5Path)

	with h5py.File(hdf5Path, 'w') as f:
		d2set = f.create_dataset('inputs', data=inputs)