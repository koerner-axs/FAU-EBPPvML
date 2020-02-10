## EBPPvML Resource Files Version 1.0

from res.constants import IMAGE_SIZE
from database import *
import numpy as np

def feed(segmentation, layer_id, dbthread):
	# stats = (0.0, 0.0, 0.0, 1.0)
	# stats = np.random.uniform(1.0, 2.0, size=(4))
	# stats /= stats.sum()

	# TODO: Make sure double processing of layers is not neck-breaking.
	#    -> Catch SQL Exception for duplicate keys.

	stats = segmentation.sum(axis=(0, 1))
	maxsum = IMAGE_SIZE * IMAGE_SIZE * 255
	stats = stats / maxsum
	stats[3] = 1.0 - stats[0:3].sum()

	dbthread.insertLayer(layer_id, np.array(segmentation[:, :, 0:3], dtype=np.uint8), *stats)