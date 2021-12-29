from scipy.fftpack import dct, idct
from myutil import loadImage, plotImage

def dct2(block):
	return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')

def idct2(block):
	return idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')

if __name__ == "__main__":
	image = loadImage('babon.png')
	#DCT
	dct_img = dct2(image)
	#IDCT
	idct_image = idct2(dct_img)
	
	plotImage([image, dct_img, idct_image], ["Original", "DCT image", "IDCT image"])

