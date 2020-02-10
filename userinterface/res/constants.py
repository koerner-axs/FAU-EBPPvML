## EBPPvML Resource Files Version 1.0

IMAGE_SIZE = 512
BATCHED_PROCESSING_BATCH_SIZE = 10
BATCHED_PROCESSING_COMMIT_SIZE = 100

# Controls for the gradient vectorizer algorithm.
GV_ENABLE = True
GV_BGM_UNIT_CIRCLE = True
GV_A_KERNEL_SIZE = 8
GV_A_STRIDE = 4
GV_B_KERNEL_SIZE = 7
GV_B_STRIDE = 1

# This controls the effect strength of the brightness mitigation
# algorithm in the image preprocessor module. Adjust only after
# a certain new value was tested on a lot of images with high
# variance.
IP_BGM_EFFECT_STRENGTH = 2.0

# An enum over the possible classifications for each pixel.
# Do not change.
SEGM_FLT_BCKGRND = 0
SEGM_FLT_POROUS = 1
SEGM_FLT_BULDGE = 2
SEGM_FLT_GOODLYR = 3
# The certainty level which has to be reached by the segmenter for
# the fault at any pixel to be accepted. This sets a lower bound
# to the predicted values. Increase to filter more for noise.
# Decrease to allow for higher sensitivity.
SEGM_FLT_DELTA_THRHLD = 0.05
SEGM_FLT_MIN_ABS_THRHLD = 0.6

import time
def timeit(method):
	def timed(*args, **kw):
		ts = time.time()
		result = method(*args, **kw)
		te = time.time()
		print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
		return result
	return timed

def batch(iterable, n=1):
	l = len(iterable)
	for ndx in range(0, l, n):
		yield iterable[ndx:min(ndx + n, l)]