import tensorflow as tf
import cv2
import numpy as np
import os

from constants import get_data
from constants import IMAGE_SIZE

# FCN Model definition
class FCNNetwork(tf.keras.Model):
	def get_config(self):
		return ''

	def  __init__(self):
		super(FCNNetwork, self).__init__()

		# Pool 1
		self.pad1 = tf.keras.layers.ZeroPadding2D(padding=((100, 100), (100, 100)), input_shape=(512, 512, 1))
		self.conv1_1 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=64, padding='valid')
		self.relu1_1 = tf.keras.layers.ReLU()

		self.conv1_2 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=64, padding='same')
		self.relu1_2 = tf.keras.layers.ReLU()

		self.pool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)

		# Pool 2
		self.conv2_1 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=128, padding='same')
		self.relu2_1 = tf.keras.layers.ReLU()

		self.conv2_2 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=128, padding='same')
		self.relu2_2 = tf.keras.layers.ReLU()

		self.pool2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)

		# Pool 3
		self.conv3_1 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=256, padding='same')
		self.relu3_1 = tf.keras.layers.ReLU()

		self.conv3_2 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=256, padding='same')
		self.relu3_2 = tf.keras.layers.ReLU()

		self.conv3_3 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=256, padding='same')
		self.relu3_3 = tf.keras.layers.ReLU()

		self.pool3 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)

		# Pool 4
		self.conv4_1 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=256, padding='same')
		self.relu4_1 = tf.keras.layers.ReLU()

		self.conv4_2 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=256, padding='same')
		self.relu4_2 = tf.keras.layers.ReLU()

		self.conv4_3 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=256, padding='same')
		self.relu4_3 = tf.keras.layers.ReLU()

		self.pool4 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)

		# Pool 5
		self.conv5_1 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=512, padding='same')
		self.relu5_1 = tf.keras.layers.ReLU()

		self.conv5_2 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=512, padding='same')
		self.relu5_2 = tf.keras.layers.ReLU()

		self.conv5_3 = tf.keras.layers.Convolution2D(kernel_size=3, strides=1, filters=256, padding='same')
		self.relu5_3 = tf.keras.layers.ReLU()

		self.pool5 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)

		# Highest-level deconvolution
		self.conv6_1 = tf.keras.layers.Convolution2D(kernel_size=7, strides=1, filters=256, padding='valid')
		self.relu6_1 = tf.keras.layers.ReLU()
		self.dropout6_1 = tf.keras.layers.Dropout(rate=0.5)

		self.conv6_2 = tf.keras.layers.Convolution2D(kernel_size=1, strides=1, filters=256, padding='valid')
		self.relu6_2 = tf.keras.layers.ReLU()
		self.dropout6_2 = tf.keras.layers.Dropout(rate=0.5)

		self.conv6_3 = tf.keras.layers.Convolution2D(kernel_size=1, strides=1, filters=256, padding='valid')
		self.deconv_hl = tf.keras.layers.Convolution2DTranspose(kernel_size=4, strides=2, filters=16)

		# Mid-level deconvolution
		self.conv7_1 = tf.keras.layers.Convolution2D(kernel_size=1, strides=1, filters=16, padding='valid')
		self.crop1 = tf.keras.layers.Cropping2D(cropping=5)

		self.eltwise1 = tf.keras.layers.Add()
		self.deconv_ml = tf.keras.layers.Convolution2DTranspose(kernel_size=4, strides=2, filters=8)

		# Low-level deconvolution
		self.conv8_1 = tf.keras.layers.Convolution2D(kernel_size=1, strides=1, filters=8, padding='valid')
		self.crop2 = tf.keras.layers.Cropping2D(cropping=9)

		self.eltwise2 = tf.keras.layers.Add()
		self.deconv_ll = tf.keras.layers.Convolution2DTranspose(kernel_size=16, strides=8, filters=3)###60

		# Final layers
		self.crop3 = tf.keras.layers.Cropping2D(cropping=28)
		self.softmax = tf.keras.layers.Softmax(axis=-1) # softmax along the channel axis

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

		# Highest-level deconvolution
		deconv_hl = self.relu6_1(self.conv6_1(pool5))
		if training:
			deconv_hl = self.dropout6_1(deconv_hl)

		deconv_hl = self.relu6_2(self.conv6_2(deconv_hl))
		if training:
			deconv_hl = self.dropout6_2(deconv_hl)

		deconv_hl = self.conv6_3(deconv_hl)
		deconv_hl = self.deconv_hl(deconv_hl)

		# Mid-level deconvolution
		deconv_ml = self.conv7_1(pool4)
		deconv_ml = self.crop1(deconv_ml)
		deconv_ml = self.eltwise1([deconv_hl, deconv_ml])
		deconv_ml = self.deconv_ml(deconv_ml)

		# Low-level deconvolution
		deconv_ll = self.conv8_1(pool3)
		deconv_ll = self.crop2(deconv_ll)
		deconv_ll = self.eltwise2([deconv_ml, deconv_ll])
		deconv_ll = self.deconv_ll(deconv_ll)

		# Final layers
		score = self.crop3(deconv_ll)
		score = self.softmax(score)

		return score


# Losses
def focal_loss(alpha=0.25, gamma=2):
	def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
		weight_a = alpha * (1 - y_pred) ** gamma * targets
		weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

		return (tf.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b 

	def loss(y_true, y_pred):
		y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
		logits = tf.log(y_pred / (1 - y_pred))

		loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

		return tf.reduce_mean(loss)

	return loss


if __name__=='__main__':
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

	import time
	TIME = str(int(time.time()))
	tensorboard = tf.keras.callbacks.TensorBoard(log_dir='F:\\Machine Learning\\FAU - Image Classification\\logs\\test\\' + TIME)

	optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.1)

	inputs = get_data('SEGM_EXT3_INP')
	labels = get_data('SEGM_EXT3_LBL')

	# for x, lbl in enumerate(labels[0:720:239]):
	# 	image = np.array(lbl.reshape(512, 512, 3) * 255.0, dtype=np.uint8)
	# 	cv2.imshow('image' + str(x), image)
	# 	cv2.waitKey(0)
	# 	cv2.destroyAllWindows()

	# exit()

	if False:
		fcn = FCNNetwork()
		fcn.compile(optimizer=optimizer, loss='mean_squared_logarithmic_error', metrics=['mae'])
		eraStart, eraEnd = 0, 49
	else:
		TIME = 1564607620
		eraLeftOff = 113
		erasToTrain = 50

		eraStart = eraLeftOff + 1
		eraEnd = eraStart + erasToTrain

		fcn = FCNNetwork()
		fcn.load_weights('F:\\Machine Learning\\FAU - Image Classification\\models\\segm_t{0}_era{1}.tfkem'.format(TIME, eraLeftOff))
		fcn.compile(optimizer=optimizer, loss='mean_squared_logarithmic_error', metrics=['mae'])

	for era in range(eraStart, eraEnd):
		NAME = 'segm_t{1}_era{0}'.format(era, TIME)

		print('Era:', era)
		fcn.fit(inputs, labels, validation_split=0.0, callbacks=[tensorboard], batch_size=4, epochs=4)

		tf.keras.models.save_model(fcn, 'F:\\Machine Learning\\FAU - Image Classification\\models\\{}.tfkem'.format(NAME))
		fcn.save_weights('F:\\Machine Learning\\FAU - Image Classification\\models\\{}.tfkem'.format(NAME))

		Dir = 'F:\\Machine Learning\\FAU - Image Classification\\models\\{}\\'.format(NAME)
		try:
			os.mkdir(Dir)
		except:
			pass

		predictions = fcn.predict(inputs[0:720:4].reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1), batch_size=4, verbose=1)
		for index, prediction in enumerate(predictions):
			img = prediction.reshape(512, 512, 3) * 255.0

			try:
				cv2.imwrite(Dir + 'pic{0}.bmp'.format(index), img)
			except:
				pass

			if False:
				cv2.imshow('image' + str(index), img)
				cv2.waitKey(0)
				cv2.destroyAllWindows()