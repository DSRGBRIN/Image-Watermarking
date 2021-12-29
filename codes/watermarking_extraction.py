from scipy.fftpack import dct, idct
from myutil import loadImage, plotImage
import pywt
import numpy as np
from DCT import dct2, idct2
from DWT import multilevelDWT, reconstructImg
from SVD import decomposeSVD, reconstructSVD
from watermarking_embedding import embedWatermark

def extractWatermark(watermarkedImage, originalImage, alpha=0.1):
	def dwtdctsvd(image):
		#1st level DWT watermarkedImage
		LF, HF = multilevelDWT(image)

		#DCT
		dctImg = dct2(LF[0])

		#SVD image
		u, s, vh, imgshape = decomposeSVD(dctImg)
		return s
	
	s_wi = dwtdctsvd(watermarkedImage)
	s_oi = dwtdctsvd(originalImage)
	#1st level DWT originalImage
	
	s_w = s_wi - s_oi
	s_w = s_w / alpha
	return s_w

def constructExtractedWatermark(originalWatermark, ext_s):
	uw, sw, vhw, imgshape_w = decomposeSVD(originalWatermark)
	watermark = reconstructSVD(uw, ext_s, vhw, imgshape_w[0])	
	return watermark	

	

if __name__ == "__main__":
	image = loadImage('babon.png')
	wimage = loadImage('brinbw.png')
	
	newImage = embedWatermark(image, wimage)
	
	new_s = extractWatermark(newImage, image)
	
	newWimage = constructExtractedWatermark(wimage, new_s)
	
	plotImage([image, newImage, wimage, newWimage], ['Original', 'Watermarked', 'watermark', 'ext.watermark'])
	
	
	
	