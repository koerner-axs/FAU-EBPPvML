if False:
	import os
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import numpy as np
import cv2, os
import time

import fully_convolutional_network_quadout as fcn
import prediction_to_layerfaults as ptlf
from constants import BASE_DIR, IMAGE_SIZE
from dataset import load_dataset


dataset = load_dataset('datasets\\segmentation\\Test Dataset', True)
inputs = dataset['inputs']

TIME = 1566481907
era = 3

fcn = fcn.FCNNetwork()
fcn.load_weights(os.path.join(BASE_DIR, 'models\\quadout\\segm_t{0}_era{1}.tfkem'.format(TIME, era)))
#fcn.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['mae'])

directory = os.path.join(BASE_DIR, 'datasets\\segmentation\\Test Dataset\\outputs\\t{0}\\'.format(str(int(time.time()))))
try:
	os.mkdir(directory)
except:
	pass

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

predictions = fcn.predict(inputs.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1), batch_size=4, verbose=1)
predictions = predictions * 255.0
for index, image in enumerate(predictions):
	#img = ptlf.convert(image, 0.05, 0.6)
	#cv2.imwrite(directory + 'image{0}_c.bmp'.format(x), img)

	cv2.imwrite(directory + 'image{0}.bmp'.format(index), image[:,:,0:3])
	cv2.imwrite(directory + 'image{0}_a.bmp'.format(index), image[:,:,1:4])
	cv2.imwrite(directory + 'image{0}_b.bmp'.format(index), image[:,:,[0, 2, 3]])
	if False:
		cv2.imshow('image' + str(index), image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

# predictions = fcn.predict(inputs.reshape(-1, 512, 512, 1), verbose=1)

# panel = np.zeros(shape=(1536, 1536, 3), dtype=float)
# for x in range(3):
# 	for y in range(3):
# 		panel[x*512:x*512+512, y*512:y*512+512, :] = predictions[3*y+x]

# delta_thrhlds = np.arange(0.25, 0.3, 0.01)
# min_abs_thrhlds = np.arange(0.4, 0.7, 0.05)

# for delta_thrhld in delta_thrhlds:
# 	for min_abs_thrhld in min_abs_thrhlds:
# 		print('Test of delta_thrhld=' + str(delta_thrhld) + ' and min_abs_thrhld=' + str(min_abs_thrhld))

# 		cv2.imwrite(directory + 'image_{0}_{1}.bmp'.format(int(round(delta_thrhld*100)), int(round(min_abs_thrhld*100))), ptlf.convert(panel))

# def routine(input):
# 	if input[1] == 0.4:
# 		print('Starting delta_thrhld=' + str(int(round(input[0]))))
# 	cv2.imwrite(directory + 'image_{0}_{1}.bmp'.format(int(round(input[0]*100)), int(round(input[1]*100))), ptlf.convert(panel, input[0], input[1]))

# runs = ((d, m, np.array(panel)) for d in delta_thrhlds for m in min_abs_thrhlds)
# from multiprocessing.dummy import Pool as ThreadPool
# pool = ThreadPool(16)
# pool.map(routine, runs)
# pool.close()
# pool.join()