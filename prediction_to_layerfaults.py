import numpy as np

from constants import SEGM_FLT_BCKGRND as FLT_BCKGRND
from constants import SEGM_FLT_POROUS as FLT_POROUS
from constants import SEGM_FLT_BULDGE as FLT_BULDGE
from constants import SEGM_FLT_GOODLYR as FLT_GOODLYR
from constants import SEGM_FLT_DELTA_THRHLD as DELTA_THRHLD
from constants import SEGM_FLT_MIN_ABS_THRHLD as MIN_ABS_THRHLD

def convert(prediction):
	fault_map = np.zeros(shape=(*prediction.shape[0:2], 3), dtype=np.uint8)

	for xpos in range(prediction.shape[0]):
		for ypos in range(prediction.shape[1]):
			channels = prediction[xpos, ypos, :]
			ch = [(FLT_POROUS, channels[0]), (FLT_BULDGE, channels[1]), (FLT_GOODLYR, channels[2])]
			ch.sort(key=lambda x: x[1])

			flt_type, delta = ch[2][0], ch[2][1] - ch[1][1]

			if delta >= DELTA_THRHLD and channels[flt_type - 1] >= MIN_ABS_THRHLD:
				fault_map[xpos, ypos] = ((255, 0, 0), (0, 255, 0), (0, 0, 255))[flt_type - 1]
			# 	fault_map[xpos, ypos] = flt_type
			# else:
			# 	fault_map[xpos, ypos] = FLT_BCKGRND

	return fault_map