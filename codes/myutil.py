import matplotlib.pyplot as plt
import cv2
import numpy as np

def plotImage(images, titles=['Approximation', ' Horizontal detail', 'Vertical detail', 'Diagonal detail']):
	imgcount =  len(images)
	fig = plt.figure(figsize=(5, 5))
	print("wavelet size:",images[0].shape)

	for i, a in enumerate(images):
		if imgcount >= 4:
			ax = fig.add_subplot(imgcount//2, imgcount//2, i + 1)
		else:
			ax = fig.add_subplot(1, imgcount, i + 1)
		ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
		ax.set_title(titles[i], fontsize=10)
		ax.set_xticks([])
		ax.set_yticks([])

	fig.tight_layout()
	plt.show()

def loadImage(fname):
	image = cv2.imread(fname)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = np.float32(image)
	image /= 255
	return image