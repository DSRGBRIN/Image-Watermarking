from scipy.fftpack import dct, idct
from myutil import loadImage, plotImage
import pywt
import numpy as np
from DCT import dct2, idct2
from DWT import multilevelDWT, reconstructImg
from SVD import decomposeSVD, reconstructSVD


def embedWatermark(image, watermark, alpha = 0.1):
	#1st level DWT
	LF, HF = multilevelDWT(image)

	#DCT
	dctImg = dct2(LF[0])

	#SVD image
	u, s, vh, imgshape = decomposeSVD(dctImg)
	matrix_s = np.diag(s)
	
	#SVD watermark
	def addWatermarkData(s_img, watermark_image, alpha_value=.1):
		#note: the watermark image must has smaller dimension than the quarter of the host image
		u_wi, s_wi, vh_wi, dimensi_wi = decomposeSVD(watermark_image)
		s_watermark = np.pad(s_wi, ((0,s_img.shape[0] - s_wi.shape[0] )), 'constant')
		
		tmp = s_watermark * alpha_value
		new_s = s_img + tmp
		return new_s

	new_s = addWatermarkData(s, watermark, alpha)

	#ISVD using new_s
	newSVD = reconstructSVD(u, new_s, vh, imgshape[0])

	#IDCT
	idctImg = idct2(newSVD)

	#inverse DWT
	newImage  = reconstructImg(idctImg, HF[0])
	return newImage
	
if __name__ == "__main__":
	image = loadImage('babon.png')
	wimage = loadImage('brinbw.png')
	
	newImage = embedWatermark(image, wimage)
	
	plotImage([image, newImage], ['Original', 'Watermarked'])
	
	
	
	
	