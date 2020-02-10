import tensorflow as tf
import numpy as np
import cv2
import os

from constants import BASE_DIR, IMAGE_SIZE
from dataset import load_dataset


# FCN Non-Dilated Tri-Skip Model definition
class FCNNetwork(tf.keras.Model):
	def get_config(self):
		return ''

	def  __init__(self):
		super(FCNNetwork, self).__init__()

		activationFunction = 'selu'

		# Pool 1
		self.pad1 = tf.keras.layers.ZeroPadding2D(padding=((100, 100), (100, 100)), input_shape=(512, 512, 1))
		self.conv1_1 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=64, padding='valid', trainable=False)
		self.relu1_1 = tf.keras.layers.Activation(activationFunction)

		self.conv1_2 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=64, padding='same', trainable=False)
		self.relu1_2 = tf.keras.layers.Activation(activationFunction)

		self.pool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)

		# Pool 2
		self.conv2_1 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=128, padding='same', trainable=False)
		self.relu2_1 = tf.keras.layers.Activation(activationFunction)

		self.conv2_2 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=128, padding='same', trainable=False)
		self.relu2_2 = tf.keras.layers.Activation(activationFunction)

		self.pool2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)

		# Pool 3
		self.conv3_1 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=256, padding='same', trainable=False)
		self.relu3_1 = tf.keras.layers.Activation(activationFunction)

		self.conv3_2 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=256, padding='same', trainable=False)
		self.relu3_2 = tf.keras.layers.Activation(activationFunction)

		self.conv3_3 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=256, padding='same', trainable=False)
		self.relu3_3 = tf.keras.layers.Activation(activationFunction)

		self.pool3 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)

		# Pool 4
		self.conv4_1 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=256, padding='same', trainable=False)
		self.relu4_1 = tf.keras.layers.Activation(activationFunction)

		self.conv4_2 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=256, padding='same', trainable=False)
		self.relu4_2 = tf.keras.layers.Activation(activationFunction)

		self.conv4_3 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=256, padding='same', trainable=False)
		self.relu4_3 = tf.keras.layers.Activation(activationFunction)

		self.pool4 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)

		# Pool 5
		self.conv5_1 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=512, padding='same', trainable=False)
		self.relu5_1 = tf.keras.layers.Activation(activationFunction)

		self.conv5_2 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=512, padding='same', trainable=False)
		self.relu5_2 = tf.keras.layers.Activation(activationFunction)

		self.conv5_3 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=256, padding='same', trainable=False)
		self.relu5_3 = tf.keras.layers.Activation(activationFunction)

		self.pool5 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)


		# ## Phase 1: only pool5 used.
		# # Reduce size along filter dimension.
		# self.conv6_1 = tf.keras.layers.Convolution2D(kernel_size=1, strides=1, filters=16, padding='same', trainable=True)

		# # Linear regression
		# self.linreg = tf.keras.layers.Dense(4, trainable=True)


		## Phase 2: only pool4&5 used.
		# Reduce size along filter dimension.
		self.conv6_1 = tf.keras.layers.Convolution2D(kernel_size=2, strides=2, filters=8, padding='same', trainable=True)
		self.conv6_2 = tf.keras.layers.Convolution2D(kernel_size=1, strides=1, filters=8, padding='same', trainable=True)
		self.eltwise1 = tf.keras.layers.Add()

		# Linear regression
		self.linreg = tf.keras.layers.Dense(4, trainable=True)


	def call(self, inputs, training=False):
		# Pool 1
		pool1 = self.pad1(inputs)
		pool1 = self.relu1_1(self.conv1_1(pool1))
		pool1 = self.relu1_2(self.conv1_2(pool1))
		pool1 = self.pool1(pool1)

		# Pool 2
		pool2 = self.relu2_1(self.conv2_1(pool1))
		pool2 = self.relu2_2(self.conv2_2(pool2))
		pool2 = self.pool2(pool2)

		# Pool 3
		pool3 = self.relu3_1(self.conv3_1(pool2))
		pool3 = self.relu3_2(self.conv3_2(pool3))
		pool3 = self.relu3_3(self.conv3_3(pool3))
		pool3 = self.pool3(pool3)

		# Pool 4
		pool4 = self.relu4_1(self.conv4_1(pool3))
		pool4 = self.relu4_2(self.conv4_2(pool4))
		pool4 = self.relu4_3(self.conv4_3(pool4))
		pool4 = self.pool4(pool4)

		# Pool 5
		pool5 = self.relu5_1(self.conv5_1(pool4))
		pool5 = self.relu5_2(self.conv5_2(pool5))
		pool5 = self.relu5_3(self.conv5_3(pool5))
		pool5 = self.pool5(pool5)


		# ## Phase 1: only pool5 used.
		# # Reduce size along filter dimension.
		# concatenated = self.conv6_1(pool5)

		# # Linear regression
		# return self.linreg(tf.keras.layers.Flatten()(concatenated))


		## Phase 2: only pool4&5 used.
		# Reduce size along filter dimension.
		p4 = self.conv6_1(pool4)
		p5 = self.conv6_2(pool5)
		concatenated = self.eltwise1([p4, p5])

		# Linear regression
		return self.linreg(tf.keras.layers.Flatten()(concatenated))


def nploss(alpha):
	def loss(y_true, y_pred):
		psum = tf.reduce_sum(y_pred)
		norm_y_pred = y_pred / tf.abs(psum) + tf.keras.backend.epsilon()
		norm_y_pred -= y_true
		mse = tf.reduce_mean(norm_y_pred * norm_y_pred)
		return mse + alpha * abs(1 - psum)
	return loss


if __name__=='__main__':
	gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.75, allow_growth=True)
	sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

	import time
	TIME = str(int(time.time()))
	optimizer = 'adam'
	loss = 'mean_absolute_error'
	#loss = nploss(alpha=0.5)

	if True:
		fcn = FCNNetwork()
		fcn.compile(optimizer=optimizer, loss=loss, metrics=['mse', 'mae'])
		erasToTrain = 4

		eraStart, eraEnd = 0, erasToTrain
	else:
		TIME = 1572710112
		eraLeftOff = 2
		erasToTrain = 4

		eraStart = eraLeftOff + 1
		eraEnd = eraStart + erasToTrain

		fcn = FCNNetwork()
		fcn.load_weights(os.path.join(BASE_DIR, 'models\\fast_pipeline\\segm_t{0}_era{1}.tfkem'.format(TIME, eraLeftOff)))
		fcn.compile(optimizer=optimizer, loss=loss, metrics=['mse', 'mae'])


	if True:
		fcn.build(input_shape=(1, 512, 512, 1))
		from tensorflow.keras.layers import Input
		x = Input(shape=(512, 512, 1))
		m = tf.keras.Model(inputs=[x], outputs=fcn.call(x))
		m.summary()
		#exit()


	dataset = load_dataset('datasets\\fast_pipeline\\Dataset 2', True)
	inputs = dataset['inputs']
	labels = dataset['labels']
	
	for era in range(eraStart, eraEnd):
		NAME = 'segm_t{1}_era{0}'.format(era, TIME)

		print('Era:', era)
		callbacks = []
		fcn.fit(inputs, labels, validation_split=0.0, callbacks=callbacks, batch_size=4, epochs=16)

		tf.keras.models.save_model(fcn, os.path.join(BASE_DIR, 'models\\fast_pipeline\\{}.tfkem'.format(NAME)))
		fcn.save_weights(os.path.join(BASE_DIR, 'models\\fast_pipeline\\{}.tfkem'.format(NAME)))

		# Dir = os.path.join(BASE_DIR, 'models\\quadout\\{}\\'.format(NAME))
		# try:
		# 	os.mkdir(Dir)
		# except:
		# 	pass

		# predictions = fcn.predict(inputs[0:720:4].reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1), batch_size=4, verbose=1)
		# for index, prediction in enumerate(predictions):
		# 	img = (prediction.reshape(512, 512, 4)[:,:,0:3]) * 255.0

		# 	try:
		# 		cv2.imwrite(os.path.join(Dir, 'pic{0}.bmp'.format(index*4)), img)
		# 	except Exception as e:
		# 		print(e)

		# 	if False:
		# 		cv2.imshow('image' + str(index*4), img)
		# 		cv2.waitKey(0)
		# 		cv2.destroyAllWindows()