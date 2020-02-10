## EBPPvML Resource Files Version 1.0

from res.constants import *
from res.dataset import _apply_input_normalization
from math import floor, sqrt, exp
import tensorflow as tf
import numpy as np
import cv2

#@timeit
def compute_gradient_vector_seq(image):
	image = image.reshape((IMAGE_SIZE, IMAGE_SIZE))

	## Compute averaged brightness map.
	brightness_map_size = floor((IMAGE_SIZE - GV_A_KERNEL_SIZE) / float(GV_A_STRIDE)) + 1
	image_averaged = np.zeros(shape=(brightness_map_size, brightness_map_size))

	indexU, indexV = 0, 0
	for x in range(0, IMAGE_SIZE - GV_A_KERNEL_SIZE + 1, GV_A_STRIDE):
		indexV = 0
		for y in range(0, IMAGE_SIZE - GV_A_KERNEL_SIZE + 1, GV_A_STRIDE):
			avg = 0.0
			for c in range(GV_A_KERNEL_SIZE):
				avg += sum(image[x:x+GV_A_KERNEL_SIZE, y+c])
			image_averaged[indexU, indexV] = avg / (GV_A_KERNEL_SIZE**2)
			indexV += 1
		indexU += 1

	## Compute gradient vectors.
	B_KERNEL_RADIUS = floor(GV_B_KERNEL_SIZE / 2.0)

	# Average gradient vectors over the entire image.
	gradient_vector = np.array((0.0, 0.0))
	for kernel_x in range(B_KERNEL_RADIUS, brightness_map_size - B_KERNEL_RADIUS, GV_B_STRIDE):
		for kernel_y in range(B_KERNEL_RADIUS, brightness_map_size - B_KERNEL_RADIUS, GV_B_STRIDE):
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

					gradient_vector[0] += vector_y
					gradient_vector[1] += vector_x

	return gradient_vector / 30000.0 # Calibrated for the tensor algorithm. Why this isn't 512*512, I don't have a clue..

#@timeit
def compute_gradient_vectors_tensor(images, gradient_map=False):
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

	size_after_avg = IMAGE_SIZE / GV_A_STRIDE
	size_after_conv = size_after_avg / GV_B_STRIDE
	# If this assertion fails: Please choose (IMAGE_SIZE,) GV_A_KERNEL_SIZE, GV_A_STRIDE,
	# GV_B_KERNEL_SIZE and GV_B_STRIDE, so that size_after_avg/_conv is an integer value.
	assert(size_after_avg.is_integer() and size_after_conv.is_integer())

	# Variable shananigans.
	with tf.compat.v1.variable_scope('PREP_GRAD_FILTER', reuse=tf.compat.v1.AUTO_REUSE):
		kernel = make_kernel(GV_B_KERNEL_SIZE)
		tf_kernel = tf.compat.v1.get_variable(name='grad_vec_filter', initializer=kernel, dtype=tf.float64)

	## Build tensorflow processing pipeline.
	# Image downsampling/averaging layer.
	tensor = tf.nn.avg_pool(images,
		ksize=[1, GV_A_KERNEL_SIZE, GV_A_KERNEL_SIZE, 1],
		strides=[1, GV_A_STRIDE, GV_A_STRIDE, 1],
		padding='SAME', data_format='NHWC', name='AVG_POOLING')

	# Compute layer for brightness gradients.
	tensor = tf.nn.conv2d(tensor, filters=tf_kernel, strides=[1, GV_B_STRIDE, GV_B_STRIDE, 1], padding='SAME', name='CONVOLUTION')

	if not gradient_map:
		# Average the tensor along height and width to retrieve the gradient
		# of the input image.
		tensor = tf.nn.avg_pool(tensor,
			ksize=[1, size_after_conv, size_after_conv, 1],
			strides=[1, 1, 1, 1],
			padding='VALID', data_format='NHWC', name='MEAN')

		# Reduce dimensionality to the batch axis.
		tensor = tf.squeeze(tensor, (1, 2))

	return tensor

#@timeit
def mitigate_brightness_gradient_tensor(image, gradient_vector):
	gradient_vector_magnitude = sqrt(gradient_vector[0]**2 + gradient_vector[1]**2)
	effect_strength = IP_BGM_EFFECT_STRENGTH / (1.0 + exp(-5.0*(gradient_vector_magnitude - 1.0)))

	gradient_vector /= gradient_vector_magnitude + 0.00001
	x = np.linspace(-1.0, 1.0, IMAGE_SIZE)
	a, b = tf.constant(x), tf.constant(x)
	if GV_BGM_UNIT_CIRCLE:
		a, b = a * tf.sqrt(1.0 - tf.square(b) / 2.0), b * tf.sqrt(1.0 - tf.square(a) / 2.0)
	a, b = x * gradient_vector[0], x * gradient_vector[1]
	a, b = tf.meshgrid(a, b)
	tensor = a + b
	tensor *= effect_strength
	tensor += image.reshape(IMAGE_SIZE, IMAGE_SIZE)
	tensor = tf.clip_by_value(tensor, 0.0, 1.0)

	return tensor.numpy().reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 1))

def load_single(filename):
	try:
		image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
		image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
	except Exception as e:
		print('The following image file could not be loaded: {}'.format(filename))
		print(e)
		raise e

	image = image.reshape((IMAGE_SIZE, IMAGE_SIZE, 1))
	image = _apply_input_normalization(image)

	if GV_ENABLE:
		gradient_vector = compute_gradient_vectors_tensor(np.array(image).reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 1)))[0]
		image = mitigate_brightness_gradient_tensor(image, gradient_vector)
		return image
	else:
		return image