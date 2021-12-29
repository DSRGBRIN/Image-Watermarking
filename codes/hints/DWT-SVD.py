from scipy.fftpack import dct, idct
from myutil import loadImage, plotImage
import pywt
import numpy as np
from DWT import multilevelDWT, reconstructImg
from SVD import decomposeSVD, reconstructSVD

image = loadImage('babon.png')

#DCT
LF, HF = multilevelDWT(image, 1)

#SVD
u, s, vh, imgshape = decomposeSVD(LF[0])
matrix_s = np.diag(s)
plotImage([u, matrix_s, vh], ['U matrix', 'S matrix', 'V transpose matrix'])

#ISVD
newSVD = reconstructSVD(u, s, vh, imgshape[0])

#IDWT
idwtImg = reconstructImg(newSVD, HF[0], False)

plotImage([image, idwtImg], ['Original', 'DWT-SVD'])