from scipy.fftpack import dct, idct
from myutil import loadImage, plotImage
import pywt
import numpy as np
from DCT import dct2, idct2
from DWT import multilevelDWT, reconstructImg

image = loadImage('babon.png')

#1st level DWT
LF, HF = multilevelDWT(image)

#DCT
dctImg = dct2(LF[0])

#IDCT
idctImg = idct2(dctImg)

#inverse DWT
newImage  = reconstructImg(idctImg, HF[0])

plotImage([image, newImage], ['Original', 'DWT-DCT'])
