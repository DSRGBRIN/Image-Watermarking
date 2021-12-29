from scipy.fftpack import dct, idct
from myutil import loadImage, plotImage
import pywt
import numpy as np
from DCT import dct2, idct2
from DWT import multilevelDWT, reconstructImg
from SVD import decomposeSVD, reconstructSVD
from watermarking_embedding import embedWatermark
from watermarking_extraction import extractWatermark, constructExtractedWatermark
import cv2
from skimage.metrics import structural_similarity

def computeStats(oriImg, compImg,isMultichannel=False):
	allres = []
	img1 = oriImg
	img2 = compImg
	psnr = cv2.PSNR(img1, img2)
	print("PSNR:",psnr)
	allres.append(psnr)
	
	if isMultichannel:
		(score, diff) = structural_similarity(img1, img2, multichannel=True, full=True)
	else:
		(score, diff) = structural_similarity(img1, img2, full=True)
	diff = (diff * 255).astype("uint8")
	print("SSIM:",score)
	allres.append(score)
	
	result = cv2.matchTemplate(img1,img2, cv2.TM_CCOEFF_NORMED)
	print("NC:",result[0][0])
	allres.append(result[0][0])
	return allres

if __name__ == "__main__":
	image = loadImage('babon.png')
	wimage = loadImage('brinbw.png')
	newImage = embedWatermark(image, wimage)
	new_s = extractWatermark(newImage, image)
	newWimage = constructExtractedWatermark(wimage, new_s)
	
	print(wimage.shape)
	print(newWimage.shape)
	
	print("Compare Image:")
	computeStats(image, newImage, True)
	
	print("Compare Watermark:")
	computeStats(wimage, newWimage)
	
	
	
	