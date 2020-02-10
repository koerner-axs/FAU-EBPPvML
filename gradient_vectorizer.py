import numpy as np
import os
import pickle
import math
import time

from constants import BASE_DIR
from constants import GV_A_KERNEL_SIZE as A_KERNEL_SIZE
from constants import GV_A_STRIDE as A_STRIDE
from constants import GV_B_KERNEL_SIZE as B_KERNEL_SIZE
from constants import GV_B_STRIDE as B_STRIDE
#from constants import GV_B_WEIGHT_DECAY as B_WEIGHT_DECAY

def compute_gradient_vector(image, image_size):
	image = image.reshape((image_size, image_size))

	## Compute averaged brightness map.
	brightness_map_size = math.floor((image_size - A_KERNEL_SIZE) / float(A_STRIDE)) + 1
	image_averaged = np.zeros(shape=(brightness_map_size, brightness_map_size))

	indexU, indexV = 0, 0
	for x in range(0, image_size - A_KERNEL_SIZE + 1, A_STRIDE):
		indexV = 0
		for y in range(0, image_size - A_KERNEL_SIZE + 1, A_STRIDE):
			avg = 0.0
			for c in range(A_KERNEL_SIZE):
				avg += sum(image[x:x+A_KERNEL_SIZE, y+c])
			image_averaged[indexU, indexV] = avg / (A_KERNEL_SIZE**2)

			indexV += 1
		indexU += 1


	## Compute gradient vectors.
	B_KERNEL_RADIUS = math.floor(B_KERNEL_SIZE / 2.0)

	# # Generate weights.
	# sum_weights = 1.0
	# weights = [1.0]
	# for x in range(B_KERNEL_RADIUS - 1):
	# 	current = weights[-1] * B_WEIGHT_DECAY
	# 	sum_weights += current
	# 	weights.append(current)
	# weights = np.array(weights)
	# weights /= sum_weights

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
					distance_half = math.sqrt(offset_x**2 + offset_y**2)
					vector_x = offset_x / distance_half * brightness_difference
					vector_y = offset_y / distance_half * brightness_difference

					gradient_vector[0] += vector_x
					gradient_vector[1] += vector_y

	# plt.imshow(image)
	# plt.show()
	# plt.imshow(image_averaged)
	# plt.show()
	return gradient_vector


# counter = 0
# for x in range(1):
# 	print('Numero:', x)
# 	img = trainImages[x]
# 	img = img.reshape(img.shape[:2])

# 	## Average pixel brightness
# 	image_size = 200
# 	kernel_size = 8
# 	stride = 4

# 	average = list()
# 	dimension = 0
# 	for x in range(0, image_size - kernel_size + 1, stride):
# 		dimension += 1
# 		for y in range(0, image_size - kernel_size + 1, stride):
# 			avg = 0
# 			for c in range(kernel_size):
# 				avg += sum(img[x:x+kernel_size, y+c])
# 			average.append(avg / kernel_size**2)

# 	image = np.array(average).reshape((dimension, dimension))

# 	fig, ax = plt.subplots()
# 	plt.imshow(image)
# 	plt.show()
# 	#plt.savefig('F:\\Machine Learning\\FAU - Image Classification\\pics\\' + str(counter + 1) + '.png')
# 	counter += 1
# 	plt.close('all')


# 	## Compute gradient vectors
# 	image_size = dimension
# 	kernel_size = 5	# Must not be even
# 	stride = 1
# 	kernel_radius = math.floor(kernel_size / 2.0)

# 	sum_weights = 1.0
# 	weights = [1.0]
# 	for x in range(kernel_radius - 1):
# 		current = weights[-1] * 0.65
# 		sum_weights += current
# 		weights.append(current)

# 	weights = np.array(weights)
# 	weights /= sum_weights

# 	image = image.reshape((image_size, image_size))

# 	x_vector_components = list()
# 	y_vector_components = list()

# 	dimension = 0
# 	for x in range(kernel_radius, image_size - kernel_radius, stride):
# 		dimension += 1
# 		for y in range(kernel_radius, image_size - kernel_radius, stride):
# 			gradient_vector = [0.0, 0.0]

# 			# Horizontal
# 			for offset in range(kernel_radius):
# 				weight = weights[offset]
# 				difference = image[x + offset][y] - image[x - offset][y]
# 				gradient_vector[0] += weight * difference

# 			# Vertical
# 			for offset in range(kernel_radius):
# 				weight = weights[offset]
# 				difference = image[x][y + offset] - image[x][y - offset]
# 				gradient_vector[1] += weight * difference

# 			# Diagonal uphill
# 			for offset in range(kernel_radius):
# 				weight = weights[offset]
# 				difference = image[x + offset][y - offset] - image[x - offset][y + offset]
# 				gradient_vector[0] += weight * difference
# 				gradient_vector[1] -= weight * difference

# 			# Diagonal downhill
# 			for offset in range(kernel_radius):
# 				weight = weights[offset]
# 				difference = image[x + offset][y + offset] - image[x - offset][y - offset]
# 				gradient_vector[0] += weight * difference
# 				gradient_vector[1] += weight * difference

# 			x_vector_components.append(float(gradient_vector[0]))
# 			y_vector_components.append(-float(gradient_vector[1]))

# 	U=np.array(x_vector_components).reshape((dimension, dimension))
# 	V=np.array(y_vector_components).reshape((dimension, dimension))

# 	print(sum(x_vector_components), sum(y_vector_components))

# 	# U=np.transpose(U)
# 	# V=np.transpose(V)

# 	X, Y = np.arange(0, dimension, 1), np.arange(dimension - 1, -1, -1)

# 	# plt.subplot(121)
# 	# plt.imshow(image)
# 	# ax = plt.subplot(122)
# 	# q = ax.quiver(U, V)
# 	fig, ax = plt.subplots()
# 	plt.imshow(img)
# 	plt.show()
# 	#plt.savefig('F:\\Machine Learning\\FAU - Image Classification\\pics\\' + str(counter - 1) + '.png')
# 	counter += 1

# 	fig, ax = plt.subplots()
# 	ax.quiver(X, Y, V, U, width=0.001, headwidth=4, headaxislength=3, scale=50, pivot='middle')
# 	ax.quiver([dimension / 2], [dimension / 2], [sum(y_vector_components)], [sum(x_vector_components)], width=0.001, headwidth=4, headaxislength=3, scale=50, pivot='middle')

# 	plt.show()
# 	#plt.savefig('F:\\Machine Learning\\FAU - Image Classification\\pics\\' + str(counter) + '.png')
# 	counter += 1
# 	plt.close('all')