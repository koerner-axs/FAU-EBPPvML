import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.666)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

directory = 'F:\\Machine Learning\\FAU - Image Classification\\datasets\\'

trainImages = list()
with open(os.path.join(directory, 'trainImages.pickle'), 'rb') as file:
  trainImages = pickle.load(file)

import matplotlib.pyplot as plt
plt.imshow(trainImages[0].reshape(100, 100))
plt.show()
plt.imshow(trainImages[1].reshape(100, 100))
plt.show()
plt.imshow(trainImages[2].reshape(100, 100))
plt.show()

exit()

trainLabels = list()
with open(os.path.join(directory, 'trainLabels.pickle'), 'rb') as file:
  trainLabels = pickle.load(file)
  trainLabels = np.array(trainLabels)

model = tf.keras.Sequential()

# First convolutional layer
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='valid', strides=1, input_shape=(100, 100, 1), activation='relu'))

# First maximum pooling layer
#model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

# Second convolutinal layer
model.add(Conv2D(filters=24, kernel_size=(3, 3), padding='same', strides=1, input_shape=(33, 33, 8), activation='relu'))

# Second maximum pooling layer
#model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

# Third convolutional layer
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=1, input_shape=(16, 16, 32), activation='relu'))

# Third maximum pooling layer
#model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

# Fourth convolitional layer
#model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=1, input_shape=(0, 0, 72), activation='relu'))

# Fifth convolutional layer
#model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=1, input_shape=(0, 0, 72), activation='relu'))

# Final maximum pooling layer
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

# Flatten layer
model.add(Flatten())

# Dense layers
model.add(Dense(2048, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(2048, activation='relu'))

# Output layer
model.add(Dense(1, activation='softmax'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(trainImages, trainLabels, batch_size=64, epochs=4)