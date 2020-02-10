import os, cv2
from constants import IMAGE_SIZE, BASE_DIR
import image_preprocessor as ip
import fully_convolutional_network_quadout as fcn

def predict(filename):
	# Load and preprocess image
	image = ip.load_single(filename)
	image = image.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

	# Load model
	model_path = os.path.join(BASE_DIR, 'models\\quadout\\segm_t1566481907_era3.tfkem')
	#model_path = os.path.join(BASE_DIR, 'models\\cce\\segm_t1566824502_era9.tfkem')
	model = fcn.FCNNetwork()
	model.load_weights(model_path)

	# Predict
	prediction = model.predict(image)
	prediction = prediction[0,:,:,0:3]
	prediction *= 255.0

	# Show result
	cv2.imshow('Prediction for ' + filename, prediction)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imwrite('F:\\Machine Learning\\FAU - EBPPvML\\img.bmp', prediction)


#predict('F:\\Machine Learning\\FAU - EBPPvML\\datasets\\segmentation\\Test Dataset\\inputs\\norm_image1.bmp')
predict('F:\\Machine Learning\\FAU - EBPPvML\\datasets\\segmentation\\Labeled Dataset 4\\inputs\\norm_image49.bmp')
#predict('F:\\Machine Learning\\FAU - EBPPvML\\datasets\\segmentation\\Labeled Dataset 4\\inputs\\norm_image4.bmp')