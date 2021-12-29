from scipy.fftpack import dct, idct
from myutil import loadImage, plotImage
import pywt
import numpy as np
from DCT import dct2, idct2
from DWT import multilevelDWT, reconstructImg
from SVD import decomposeSVD, reconstructSVD

image = loadImage('babon.png')

#DCT
dctImg = dct2(image)

#SVD
u, s, vh, imgshape = decomposeSVD(dctImg)
matrix_s = np.diag(s)
plotImage([u, matrix_s, vh], ['U matrix', 'S matrix', 'V transpose matrix'])

#ISVD
newSVD = reconstructSVD(u, s, vh, imgshape[0])

#IDCT
idctImg = idct2(newSVD)
plotImage([image, idctImg], ['Original', 'DCT-SVD'])