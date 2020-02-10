import os, cv2
import numpy as np
import tensorflow as tf
from constants import IMAGE_SIZE, BASE_DIR
from dataset import load_dataset, build_dataset
from math import ceil
import image_preprocessor as ip
import fully_convolutional_network_quadout as fcn


model_path = os.path.join(BASE_DIR, 'models\\quadout\\segm_t1566481907_era3.tfkem')
model = fcn.FCNNetwork()
model.load_weights(model_path)

dataset = load_dataset('datasets\\fast_pipeline\\Dataset 2', False)
inputs = dataset['inputs']

num_images = inputs.shape[0]
percentages = np.empty([num_images, 4], dtype=float)
batches = 1
batch_size = ceil(num_images / batches)
for index in range(0, batches):
	start, end = index * batch_size, min(num_images, (index + 1) * batch_size)
	p_init = model.predict(inputs[start:end].reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1), batch_size=8, verbose=1)
	print(p_init.shape)

	tf.compat.v1.disable_eager_execution()
	p_plhdr = tf.compat.v1.placeholder(dtype=tf.float32, shape=p_init.shape)
	predictions = tf.compat.v1.get_variable('predictions', p_init.shape)
	with tf.compat.v1.Session() as session:
		session.run(tf.compat.v1.global_variables_initializer())
		session.run(predictions.assign(p_plhdr), {p_plhdr: p_init})

		gsum = tf.math.reduce_sum(predictions, [1, 2])
		psum = tf.math.reduce_sum(gsum, [1])
		psum = tf.expand_dims(psum, 1)
		tensor = tf.math.divide(gsum, psum)
		tensor = session.run(tensor)

	print(tensor.shape)
	print(percentages[start:end,:].shape)
	percentages[start:end,:] = tensor

print(percentages.shape)
print(percentages)

build_dataset('datasets\\fast_pipeline\\Dataset 2', False, False, np.reshape(inputs, (num_images, IMAGE_SIZE, IMAGE_SIZE, 1)), percentages)