from constants import BASE_DIR, IMAGE_SIZE
from constants import GV_A_KERNEL_SIZE as A_KERNEL_SIZE
from constants import GV_A_STRIDE as A_STRIDE
from constants import GV_B_KERNEL_SIZE as B_KERNEL_SIZE
from constants import GV_B_STRIDE as B_STRIDE
from constants import IP_BGM_EFFECT_STRENGTH as BGM_EFFECT_STRENGTH
from constants import hdf5_get
from dataset import _apply_input_normalization

import tensorflow as tf
import numpy as np
import h5py
import cv2
from math import floor, ceil, exp, sqrt


def compute_gradient_vector_cpu(image):
	image = image.reshape((IMAGE_SIZE, IMAGE_SIZE))

	## Compute averaged brightness map.
	brightness_map_size = floor((IMAGE_SIZE - A_KERNEL_SIZE) / float(A_STRIDE)) + 1
	image_averaged = np.zeros(shape=(brightness_map_size, brightness_map_size))

	indexU, indexV = 0, 0
	for x in range(0, IMAGE_SIZE - A_KERNEL_SIZE + 1, A_STRIDE):
		indexV = 0
		for y in range(0, IMAGE_SIZE - A_KERNEL_SIZE + 1, A_STRIDE):
			avg = 0.0
			for c in range(A_KERNEL_SIZE):
				avg += sum(image[x:x+A_KERNEL_SIZE, y+c])
			image_averaged[indexU, indexV] = avg / (A_KERNEL_SIZE**2)
			indexV += 1
		indexU += 1


	## Compute gradient vectors.
	B_KERNEL_RADIUS = floor(B_KERNEL_SIZE / 2.0)

	# Average gradient vectors over the entire image.
	gradient_vector = np.array((0.0, 0.0))
	for kernel_x in range(B_KERNEL_RADIUS, brightness_map_size - B_KERNEL_RADIUS, B_STRIDE):
		for kernel_y in range(B_KERNEL_RADIUS, brightness_map_size - B_KERNEL_RADIUS, B_STRIDE):
			# The scan will follow this pattern:
			# .......
			# .###oo.
			# .###oo.
			# .##Xoo.
			# .##ooo.
			# .##ooo.
			# .......
			# Where . signals no scan, X signals the center of the kernel and where # signals
			# which pixels are directly considered in the following.
			# Note how each # has a corresponding o opposite with respect to the center of the
			# kernel. The 2-norm distance from any two such partners is used as a means of
			# weighing the brightness gradient between said two pixels.

			maxYOffset = B_KERNEL_RADIUS
			for offset_x in range(-B_KERNEL_RADIUS, 1):
				if offset_x == 0:
					maxYOffset = -1	# For the last column we only process it half way.
				for offset_y in range(-B_KERNEL_RADIUS, maxYOffset + 1):
					brightness_difference = image_averaged[kernel_x - offset_x, kernel_y - offset_y] - image_averaged[kernel_x + offset_x, kernel_y + offset_y]
					distance_half = sqrt(offset_x**2 + offset_y**2)
					vector_x = offset_x / distance_half * brightness_difference
					vector_y = offset_y / distance_half * brightness_difference

					gradient_vector[0] += vector_x
					gradient_vector[1] += vector_y

	return gradient_vector


def compute_gradient_vectors_gpu(images, gradient_map=False):
	def make_kernel(kernel_size):
		kernel = np.zeros(shape=(kernel_size, kernel_size, 1, 2))

		kernel_radius = floor(kernel_size / 2.0)
		maxYOffset = kernel_radius
		for offset_x in range(-kernel_radius, 1):
			if offset_x == 0:
				maxYOffset = -1 # For the last column we only process it half way.
			for offset_y in range(-kernel_radius, maxYOffset + 1):
				dist_half = sqrt(offset_x**2 + offset_y**2)
				# Filter for gradients on x.
				kernel[offset_x + kernel_radius, offset_y + kernel_radius, 0, 0] = -offset_y / dist_half
				kernel[-offset_x + kernel_radius, -offset_y + kernel_radius, 0, 0] = offset_y / dist_half
				# Filter for gradients on y.
				kernel[offset_x + kernel_radius, offset_y + kernel_radius, 0, 1] = -offset_x / dist_half
				kernel[-offset_x + kernel_radius, -offset_y + kernel_radius, 0, 1] = offset_x / dist_half

		return tf.compat.v1.constant(kernel)

	size_after_avg = IMAGE_SIZE / A_STRIDE
	size_after_conv = size_after_avg / B_STRIDE
	# If this assertion fails: Please choose (IMAGE_SIZE,) A_KERNEL_SIZE, A_STRIDE,
	# B_KERNEL_SIZE and B_STRIDE, so that size_after_avg/_conv is an integer value.
	assert(size_after_avg.is_integer() and size_after_conv.is_integer())

	gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.75, allow_growth=True)
	with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as session:
		# Variable shananigans.
		with tf.compat.v1.variable_scope('PREP_GRAD_FILTER', reuse=tf.compat.v1.AUTO_REUSE):
			kernel = make_kernel(B_KERNEL_SIZE)
			tf_kernel = tf.compat.v1.get_variable(name='grad_vec_filter', initializer=kernel, dtype=tf.float64)

		session.run(tf.compat.v1.global_variables_initializer())

		## Build tensorflow processing pipeline.
		# Image downsampling/averaging layer.
		tensor = tf.nn.avg_pool(images,
			ksize=[1, A_KERNEL_SIZE, A_KERNEL_SIZE, 1],
			strides=[1, A_STRIDE, A_STRIDE, 1],
			padding='SAME', data_format='NHWC', name='AVG_POOLING')

		# Compute layer for brightness gradients.
		tensor = tf.nn.conv2d(tensor, filters=tf_kernel, strides=[1, B_STRIDE, B_STRIDE, 1], padding='SAME', name='CONVOLUTION')

		if not gradient_map:
			# Average the tensor along height and width to retrieve the gradient
			# of the input image.
			tensor = tf.nn.avg_pool(tensor,
				ksize=[1, size_after_conv, size_after_conv, 1],
				strides=[1, 1, 1, 1],
				padding='VALID', data_format='NHWC', name='MEAN')

			# Reduce dimensionality to the batch axis.
			tensor = tf.squeeze(tensor, (1, 2))

		ret = session.run(tensor)

	return ret


def mitigate_brightness_gradient_cpu(image, gradient_vector):
	# Based on the magnitude of the gradient vector the image will now
	# be rebalanced. The overall strength of the effect applied will be a
	# controlled by a sigmoidal function of the magnitude of the gradient
	# vector. The local effect strength will in addition be dependent on
	# the 2-norm distance to the middle of the image.
	gradient_vector_magnitude = sqrt(gradient_vector[0]**2 + gradient_vector[1]**2)
	effect_strength = BGM_EFFECT_STRENGTH / (1.0 + exp(-(gradient_vector_magnitude - 180.0) / 50.0))

	gradient_vector /= gradient_vector_magnitude + 0.00001

	for current_x in range(0, IMAGE_SIZE):
		for current_y in range(0, IMAGE_SIZE):
			normalized_x = 2 * current_x / IMAGE_SIZE - 1.0
			normalized_y = 2 * current_y / IMAGE_SIZE - 1.0
			normalized_x = normalized_x * sqrt(1.0 - normalized_y**2 / 2)
			normalized_y = normalized_y * sqrt(1.0 - normalized_x**2 / 2)
			distance_in_unit_circle = sqrt(normalized_x**2 + normalized_y**2)
			dot_product = gradient_vector[0] * normalized_x + gradient_vector[1] * normalized_y
			local_effect_strength = effect_strength * dot_product

			image[current_x, current_y] = np.clip(image[current_x, current_y] + local_effect_strength, 0.0, 1.0)
	return image


def mitigate_brightness_gradients_cpu(images, gradient_vectors):
	for image, gradient_vector in zip(images, gradient_vectors):
		mitigate_brightness_gradient_cpu(image, gradient_vector)
	return images


def process_images(images, threaded=True):
	gradient_vectors = compute_gradient_vectors_gpu(images)
	if not threaded:
		images = mitigate_brightness_gradients_cpu(images, gradient_vectors)
	else:
		from multiprocessing.dummy import Pool as ThreadPool
		pool = ThreadPool(2)
		print(images.shape)
		def routine(index, image):
			mitigate_brightness_gradient_cpu(image, gradient_vectors[index])
		pool.starmap(routine, enumerate(images))
		pool.close()
		pool.join()
	return images


def process_hdf5(filename):
	images = hdf5_get('inputs', filename)
	return process_images(images)


def process_single(image):
	gradient_vector = compute_gradient_vector_cpu(image)
	images = mitigate_brightness_gradients_cpu([image], [gradient_vector])
	return images[0]


def load_single(filename):
	try:
		image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
		image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
	except Exception as e:
		print('The following image file could not be loaded: {}'.format(filename))
		print(e)
		exit(1)

	image = image.reshape((IMAGE_SIZE, IMAGE_SIZE, 1))
	image = _apply_input_normalization(image)
	return process_single(image)