## EBPPvML Resource Files Version 1.0

from res.constants import IMAGE_SIZE
import numpy as np
import h5py, cv2, os
from os.path import join as join_path

# Returns a list of all the files in the given directory whose name ends
# with the given file extension. This list is sorted by the numbers in the
# filenames. Will fail if any file with the given file extension does not
# have any digits in its name.
def _directory_iterator(directory, file_extension):
	# Get all filenames in the directory that have the right file extension.
	dirFiles = list(os.listdir(directory))
	names = [name for name in dirFiles if name.endswith(file_extension)]

	# Sort the filenames ascending by all the digits they contain.
	try:
		names.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
	except ValueError as e:
		print('A file in the given directory appears to not contain digits in its name.\n\tPlease remove the file or change its extension if it is not supposed to be included in the dataset.')
		print(e)
		exit(1)

	# Make absolute paths from the filenames.
	paths = [join_path(directory, name) for name in names]
	return paths

# Load all the image files given by their absolute path and return them as
# a numpy array.
def _load_files(paths, n_channels):
	if n_channels == 1:
		cv2_channels = cv2.IMREAD_GRAYSCALE
	elif n_channels == 3:
		cv2_channels = cv2.IMREAD_COLOR
	else:
		raise ValueError('The argument n_channels of function _load_files must be either 1 or 3, i.e. only grayscale and rgb images are supported.')

	# Target uninitialized numpy array.
	images = np.empty(shape=(len(paths), IMAGE_SIZE, IMAGE_SIZE, n_channels), dtype=float)

	# Load and resize images, then store them in the numpy array.
	for index, path in enumerate(paths):
		try:
			image = cv2.imread(path, cv2_channels)
			image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
		except Exception as e:
			print('The following image file could not be loaded with n_channels={0}: {1}'.format(n_channels, path))
			print(e)
			exit(1)
		images[index] = image.reshape((512, 512, n_channels))

	return images

# Apply the normalization technique to the input images.
def _apply_input_normalization(data):
	# Currently, we do not zero-center the images nor do we adjust the
	# standard deviation. We just set the value range to [0.0; 1.0].
	return data / 255.0

# Set the value range of the label images.
def _apply_label_value_range(data):
	# Set the value range to [0.0; 1.0].
	return data / 255.0

# Builds an HDF5-file containing the inputs and labels (or just the inputs)
# that are loaded from the directory given.
def build_dataset(directory, load_inputs, load_labels, inputs=None, labels=None, filename='dataset'):
	directory = join_path(BASE_DIR, directory)
	dset_path = join_path(directory, filename + '.hdf5')

	with h5py.File(dset_path, 'w') as hdf5_file:
		if load_inputs:
			## Load the inputs.
			names = _directory_iterator(join_path(directory, 'inputs'), '.bmp')
			inputs = _load_files(names, n_channels=1)
			inputs = _apply_input_normalization(inputs)
			hdf5_file.create_dataset('inputs', data=inputs)
		else:
			## Save the given inputs to the dataset file.
			hdf5_file.create_dataset('inputs', data=inputs)

		if load_labels:
			## Load the labels.
			names = _directory_iterator(join_path(directory, 'labels'), '.bmp')
			labels = _load_files(names, n_channels=3)
			labels = _apply_label_value_range(labels)
			hdf5_file.create_dataset('labels', data=labels)
		else:
			## Save the given labels to the dataset file.
			hdf5_file.create_dataset('labels', data=labels)

	return dset_path

# Augments the dataset by rotating and flipping the inputs and labels if
# present.
def _apply_dataset_augmentation(dataset):
	for name in dataset.keys():
		array = dataset[name]
		aug_array = np.empty(shape=(8, *array.shape))

		if len(array.shape) == 4:
			aug_array[0] = array
			array = array[:,:,::-1,:]
			aug_array[1] = array
			array = np.rot90(array, axes=(1, 2))
			aug_array[2] = array
			array = array[:,::-1,:,:]
			aug_array[3] = array
			array = np.rot90(array, axes=(1, 2))
			aug_array[4] = array
			array = array[:,:,::-1,:]
			aug_array[5] = array
			array = np.rot90(array, axes=(1, 2))
			aug_array[6] = array
			array = array[:,::-1,:,:]
			aug_array[7] = array
		else:
			# This array is not of the required shape for
			# the augmentation operation. Simply copy the
			# data.
			aug_array[0] = aug_array[1] = aug_array[2] = aug_array[3] = aug_array[4] = aug_array[5] = aug_array[6] = aug_array[7] = array

		dataset[name] = aug_array.reshape((-1, *array.shape[1:]))

# Load the inputs and labels (if they are present) from the given
# HDF5-file. Returns a dictionary.
def load_dataset(relative_path, do_data_aug):
	path = join_path(BASE_DIR, relative_path)
	path = join_path(path, 'dataset.hdf5')
	dataset = dict()

	with h5py.File(path, 'r') as hdf5_file:
		if 'inputs' in hdf5_file.keys():
			dataset['inputs'] = hdf5_file['inputs'][()]
		if 'labels' in hdf5_file.keys():
			dataset['labels'] = hdf5_file['labels'][()]

	if do_data_aug:
		_apply_dataset_augmentation(dataset)

	return dataset