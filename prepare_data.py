import numpy as np
import os
import cv2
import random
import pickle

from image_preprocessor import process_images, compute_gradient_vectors_gpu, mitigate_brightness_gradient_cpu

from constants import IMAGE_SIZE

def routine(pair):
	img, vec = pair
	img = img.reshape(IMAGE_SIZE, IMAGE_SIZE, 1)
	img = mitigate_brightness_gradient_cpu(img, vec)
	img = np.array(img * 255.0, dtype=np.uint8)
	return img

if __name__=='__main__':
	number = 256
	load_directory = 'F:\\Machine Learning\\FAU - EBPPvML\\datasets\\'
	store_directory = 'F:\\Machine Learning\\FAU - EBPPvML\\datasets\\fast_pipeline\\Dataset 2\\inputs'
	categories = [0, 1, 2]

	# prep.run(os.path.join(directory, 'trainImages-normalized.pickle'))
	# exit()

	# trainImages, trainLabels = list(), list()

	for cid in categories:
		cdir = os.path.join(load_directory, str(cid))	# Path to the directory for the current category

		print('Loading pictures from directory: ' + cdir)
		count = 0
		images = np.empty(shape=(number, IMAGE_SIZE, IMAGE_SIZE, 1))
		for dirEntry in os.listdir(cdir):
			if count >= number:
				break
			try:
				name = 'image{}.bmp'.format(cid * number + count)
				image = cv2.imread(os.path.join(cdir, dirEntry), cv2.IMREAD_GRAYSCALE)
				image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
				cv2.imwrite(os.path.join(store_directory, 'raw\\' + name), image)
				img = np.array(image, dtype=float).reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1) / 255.0
				images[count,:,:,:] = img
			except Exception as e:
				print('Skipping file due to exception thrown: ' + dirEntry)
				print('Due to: ' + str(e))
				pass
			count += 1

		gradient_vectors = compute_gradient_vectors_gpu(images)
		print(gradient_vectors.shape)
		from multiprocessing import Pool as ThreadPool
		pool = ThreadPool(15)
		print(images.shape)

		imgs = pool.map(routine, zip(images, gradient_vectors))
		pool.close()
		pool.join()
		for index, img in enumerate(imgs):
			name = 'norm_image{}.bmp'.format(cid * number + index)
			cv2.imwrite(os.path.join(store_directory, name), img)

		# imgs = process_images(images)
		# imgs = np.array(imgs * 255.0, dtype=np.uint8)
		# for i in range(number):
		# 	name = 'norm_image{}.bmp'.format(cid * number + i)
		# 	cv2.imwrite(os.path.join(store_directory, name), imgs[i])

		print(count)

# trainingData = list(zip(trainImages, trainLabels))
# random.shuffle(trainingData)
# trainImages, trainLabels = zip(*trainingData)

# trainImages = np.array(trainImages).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
# trainImages = trainImages / 255.0

# with open(os.path.join(directory, 'trainImages-base.pickle'), 'wb') as file:
# 	pickle.dump(trainImages, file)

# with open(os.path.join(directory, 'trainLabels-base.pickle'), 'wb') as file:
# 	pickle.dump(list(trainLabels), file)

# with open(os.path.join(directory, 'trainImages-normalized.pickle'), 'wb') as file:
# 	pickle.dump(trainImages, file)

# with open(os.path.join(directory, 'trainLabels-normalized.pickle'), 'wb') as file:
# 	pickle.dump(list(trainLabels), file)

# prep.run(os.path.join(directory, 'trainImages-normalized.pickle'))