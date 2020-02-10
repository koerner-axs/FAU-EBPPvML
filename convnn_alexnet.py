import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

directory = 'F:\\Machine Learning\\FAU - Image Classification\\datasets\\'

trainImages = list()
with open(os.path.join(directory, 'trainImages-normalized.pickle'), 'rb') as file:
  trainImages = pickle.load(file)

trainLabels = list()
with open(os.path.join(directory, 'trainLabels-normalized.pickle'), 'rb') as file:
  trainLabels = pickle.load(file)
  trainLabels = np.array(trainLabels)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

from tensorflow.keras.callbacks import TensorBoard

import time
NAME = str(int(time.time()))
tensorboard = TensorBoard(log_dir='F:\\Machine Learning\\FAU - Image Classification\\logs\\test\\' + NAME)

alexnet = Sequential()

# Layer 1
alexnet.add(Conv2D(48, (7, 7), input_shape=(200, 200, 1),
	padding='same', kernel_regularizer=l2(0.0)))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
alexnet.add(Conv2D(24, (3, 3), padding='same'))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
# alexnet.add(ZeroPadding2D((1, 1)))
# alexnet.add(Conv2D(32, (3, 3), padding='same')) # 72
# alexnet.add(BatchNormalization())
# alexnet.add(Activation('relu'))
# alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 4
# alexnet.add(ZeroPadding2D((1, 1)))
# alexnet.add(Conv2D(128, (3, 3), padding='same'))
# alexnet.add(BatchNormalization())
# alexnet.add(Activation('relu'))

# Layer 5
# alexnet.add(ZeroPadding2D((1, 1)))
# alexnet.add(Conv2D(64, (3, 3), padding='same'))
# alexnet.add(BatchNormalization())
# alexnet.add(Activation('relu'))
alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 6
alexnet.add(Flatten())
# alexnet.add(Dense(128))
# alexnet.add(BatchNormalization())
# alexnet.add(Activation('relu'))
# alexnet.add(Dropout(0.5))

# Layer 7
alexnet.add(Dense(16))
alexnet.add(BatchNormalization())
alexnet.add(Activation('sigmoid'))
alexnet.add(Dropout(rate=0.66))

# Layer 8
alexnet.add(Dense(3))
alexnet.add(BatchNormalization())
alexnet.add(Activation('softmax'))

alexnet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

alexnet.fit(trainImages, trainLabels, validation_split=0.3, callbacks=[tensorboard], batch_size=64, epochs=150)

#print(alexnet.evaluate(validationImages, validationLabels))

tf.keras.models.save_model(alexnet, 'F:\\Machine Learning\\FAU - Image Classification\\models\\{}.tfkem'.format(NAME))