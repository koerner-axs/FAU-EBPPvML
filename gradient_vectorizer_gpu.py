import numpy as np
import tensorflow as tf
import math

from constants import IMAGE_SIZE
from constants import GV_A_KERNEL_SIZE as A_KERNEL_SIZE
from constants import GV_A_STRIDE as A_STRIDE
from constants import GV_B_KERNEL_SIZE as B_KERNEL_SIZE
from constants import GV_B_STRIDE as B_STRIDE


def compute_gradient_vector(image, gradient_map=False):
	def make_kernel(kernel_size):
		kernel = np.zeros(shape=(kernel_size, kernel_size, 1, 2))

		kernel_radius = math.floor(kernel_size / 2.0)
		maxYOffset = kernel_radius
		for offset_x in range(-kernel_radius, 1):
			if offset_x == 0:
				maxYOffset = -1 # For the last column we only process it half way.
			for offset_y in range(-kernel_radius, maxYOffset + 1):
				dist_half = math.sqrt(offset_x**2 + offset_y**2)
				# Filter for gradients on x.
				kernel[offset_x + kernel_radius, offset_y + kernel_radius, 0, 0] = -offset_y / dist_half
				kernel[-offset_x + kernel_radius, -offset_y + kernel_radius, 0, 0] = offset_y / dist_half
				# Filter for gradients on y.
				kernel[offset_x + kernel_radius, offset_y + kernel_radius, 0, 1] = -offset_x / dist_half
				kernel[-offset_x + kernel_radius, -offset_y + kernel_radius, 0, 1] = offset_x / dist_half

		return tf.constant(kernel)

	size_after_avg = IMAGE_SIZE / A_STRIDE
	size_after_conv = size_after_avg / B_STRIDE
	# If this assertion fails: Please choose (IMAGE_SIZE,) A_KERNEL_SIZE, A_STRIDE,
	# B_KERNEL_SIZE and B_STRIDE, so that size_after_avg/_conv is an integer value.
	assert(size_after_avg.is_integer() and size_after_conv.is_integer())

	kernel = make_kernel(B_KERNEL_SIZE)
	session = tf.Session()
	tf_kernel = tf.get_variable(name='grad_vec_filter', initializer=kernel)
	session.run(tf.global_variables_initializer())

	## Build tensorflow processing pipeline.
	tensor = tf.nn.avg_pool(image,
		ksize=[1, A_KERNEL_SIZE, A_KERNEL_SIZE, 1],
		strides=[1, A_STRIDE, A_STRIDE, 1],
		padding='SAME', data_format='NHWC', name='AVG_POOLING')

	tensor = tf.nn.conv2d(tensor, filter=tf_kernel, strides=[1, B_STRIDE, B_STRIDE, 1], padding='SAME', name='CONVOLUTION')

	if not gradient_map:
		tensor = tf.nn.avg_pool(tensor,
			ksize=[1, size_after_conv, size_after_conv, 1],
			strides=[1, 1, 1, 1],
			padding='VALID', data_format='NHWC', name='MEAN')

		tensor = tf.squeeze(tensor)

	return session.run(tensor)


# For testing purposes only...
if __name__=='__main__':
	from constants import get_data
	img = get_data('SEGM_TST1_INP')[8*5]
	img = img.reshape(-1, *img.shape)
	print(img.shape)
	grad_vec = compute_gradient_vector(img, True)

	print(grad_vec.shape)
	dimension = grad_vec.shape[1]
	U=grad_vec[0,:,:,0]
	V=grad_vec[0,:,:,1]

	gradient = (np.sum(grad_vec[0,:,:,0], axis=(0, 1)), np.sum(grad_vec[0,:,:,1], axis=(0, 1)))

	# U=np.transpose(U)
	# V=np.transpose(V)
	X, Y = np.arange(0, dimension, 1), np.arange(0, dimension, 1)

	import matplotlib.pyplot as plt
	img = img.reshape(512, 512)
	plt.subplot(121)
	plt.imshow(img)
	ax = plt.subplot(122)
	#q = ax.quiver(U, V)
	ax.quiver(X, Y, U, V, width=0.001, headwidth=4, headaxislength=2, scale=500, pivot='middle')
	ax.quiver([dimension / 2], [dimension / 2], [gradient[0]], [gradient[1]], width=0.01, headwidth=4, headaxislength=3, scale=500, pivot='middle')
	plt.show()