from pathlib import Path
from PIL import Image

directory = 'F:\\Machine Learning\\FAU - Image Classification\\classified_images\\'
dir2 = 'F:\\Machine Learning\\FAU - Image Classification\\datasets\\'

images = list(Path(directory).glob('*/*'))
images = [str(x) for x in images]

for filename in images[10:]:
	print(filename)
	img = Image.open(filename)
	print(img.format, img.size, img.mode)
	img.show()
	name = filename[len(directory):-4]
	print(dir2 + name)
	img.save(dir2 + name + '.bmp', 'bmp')