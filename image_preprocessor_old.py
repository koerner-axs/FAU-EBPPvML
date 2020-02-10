import numpy as np
import os
import pickle
import math
import matplotlib.pyplot as plt

from multiprocessing.dummy import Pool as ThreadPool

import gradient_vectorizer as gv

from constants import BASE_DIR
from constants import IMAGE_SIZE
from constants import IPP_BGM_EFFECT_STRENGTH as BGM_EFFECT_STRENGTH

def routine(index, image):
	image = image.reshape(IMAGE_SIZE, IMAGE_SIZE)

	## Mitigate a possible brightness gradient of the image.
	# Compute the gradient vector.
	gradient_vector = gv.compute_gradient_vector(image, IMAGE_SIZE)

	# Using the brightness gradient vector try to balance the image.
	# img = np.array(image)
	mitigate_brightness_gradient(image, gradient_vector)

	# print('gradient_vector:', gv.compute_gradient_vector(img, IMAGE_SIZE))
	# print('gradient_vector:', gv.compute_gradient_vector(image, IMAGE_SIZE))

	# ax = plt.subplot(121)
	# ax.imshow(img)
	# ax = plt.subplot(122)
	# ax.imshow(image)
	# plt.show()

	## Increase the images contrast.
	# TODO
	print(str(index))
	return None

def run(filename):
	print('Starting prep.')

	with open(filename, 'rb') as file:
		images_resized = pickle.load(file)

	pool = ThreadPool(16)
	images_resized = images_resized.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
	print(images_resized.shape)
	pool.starmap(routine, enumerate(images_resized))
	#results = [pool.apply(routine, (index, image)) for index, image in enumerate(images_resized)]

	pool.close()
	pool.join()

	with open(filename, 'wb') as file:
		pickle.dump(images_resized, file)


def mitigate_brightness_gradient(image, gradient_vector):
	# Based on the magnitude of the image's gradient vector the image will now
	# be rebalanced. The overall strength of the effect applied will be a
	# sigmoidal in the magnitude of the gradient vector.
	# The local effect strength will in addition be dependent on the 2-norm
	# distance to the middle of the image.

	gradient_vector_magnitude = math.sqrt(gradient_vector[0]**2 + gradient_vector[1]**2)
	effect_strength = BGM_EFFECT_STRENGTH / (1.0 + math.exp(-(gradient_vector_magnitude - 180.0) / 50.0))

	gradient_vector /= gradient_vector_magnitude + 0.00001

	for current_x in range(0, IMAGE_SIZE):
		for current_y in range(0, IMAGE_SIZE):
			normalized_x = 2 * current_x / IMAGE_SIZE - 1.0
			normalized_y = 2 * current_y / IMAGE_SIZE - 1.0
			normalized_x = normalized_x * math.sqrt(1.0 - normalized_y**2 / 2)
			normalized_y = normalized_y * math.sqrt(1.0 - normalized_x**2 / 2)
			distance_in_unit_circle = math.sqrt(normalized_x**2 + normalized_y**2)
			dot_product = gradient_vector[0] * normalized_x + gradient_vector[1] * normalized_y
			local_effect_strength = effect_strength * dot_product

			image[current_x, current_y] = np.clip(image[current_x, current_y] + local_effect_strength, 0.0, 1.0)