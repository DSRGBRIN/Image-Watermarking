from scipy.fftpack import dct, idct
from myutil import loadImage, plotImage
import pywt
import numpy as np

def multilevelDWT(image, N=1, isPlot=False):
	ctr = 0
	result = {}
	HF = []
	LF = []
	data = image
	for i in range(N):
		result[ctr] = pywt.dwt2(data, 'bior1.3')
		data, (LH, HL, HH) = result[ctr]
		if i == 0:
			HF.append( np.array([LH, HL, HH]) )
			LF.append( data )
	if isPlot:
		plotImage([data, LH, HL, HH], ["LL","LH","HL","HH"])
	return LF, HF
	
def reconstructImg(LF, HF, showimg=False):
	temp = np.array([LF, HF])
	img=pywt.idwt2(temp,'bior1.3')
	if showimg:
		plotImage([img], ["reconstructed image"])
	return img

if __name__ == "__main__":
	image = loadImage('babon.png')	
	#DWT
	LF, HF = multilevelDWT(image, 2)
	#IDWT
	idwtImg = reconstructImg(LF[0], HF[0], False)

	plotImage([image, idwtImg], ["Original", "IDWT image"])

