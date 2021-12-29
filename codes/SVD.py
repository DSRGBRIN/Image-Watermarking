from scipy.fftpack import dct, idct
from myutil import loadImage, plotImage
import pywt
import numpy as np

def decomposeSVD(block, isdebug = False):
	u, s, vh = np.linalg.svd(block)
	if isdebug:
		print( "u.shape", u.shape)
		print( "s.shape", s.shape)
		print( "vh.shape",vh.shape)
	return u, s, vh, block.shape

def reconstructSVD(u,s,vh, size, showimg=False):
    reconst_img = np.matrix(u[:, :size]) * np.diag(s[:size]) * np.matrix(vh[:size, :])
    if showimg:
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        plt.axis('off')
        plt.imshow(reconst_img, cmap=plt.get_cmap("gray"))
        plt.savefig('test.png', bbox_inches='tight', transparent=True, pad_inches=0)
        plt.show()
        return reconst_img
    else:
        return reconst_img
if __name__ == "__main__":
	image = loadImage('babon.png')	
	#SVD
	u, s, vh, imgshape = decomposeSVD(image)
	matrix_s = np.diag(s)
	plotImage([u, matrix_s, vh], ['U matrix', 'S matrix', 'V transpose matrix'])
	#ISVD
	newImage = reconstructSVD(u, s, vh, imgshape[0])
	plotImage([image, newImage], ['Original', 'Inverse SVD'])


