import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import fourier_gaussian

#Generate z displacements (in pixel units) to embed an Nx x Ny image 
#in an Nx x Ny x Nz scalar field, both assumed to be periodic.
#The displacement proile will be smooth on the scale of sigma (also in pixel units)
def zDisplacements(Nx, Ny, Nz, sigma):
    noise = [ np.random.randn(Nx, Ny) for i in range(2) ] #Re and Im of complex white noise
    noise = [ gaussian_filter(nPart, sigma, mode='wrap') for nPart in noise ] #Bandwidth limit them
    zFrac = np.angle(noise[0]+1j*noise[1])/(2*np.pi) #periodic fraction displacement in [-0.5,0.5)
    zFrac -= np.floor(zFrac) #wrap to [0,1)
    return np.floor(Nz * zFrac).astype(int)

def convolveTEM(img, Nz, sigma1=7, sigma2=7, binaryThresh=0.0146):
	# doesn't affect AF or VF but ensures the particles are connected
	zPos = zDisplacements(img.shape[0], img.shape[1], Nz, sigma1)
	microStruct = np.zeros((img.shape[0], img.shape[1], Nz))
	for x in range(img.shape[0]):
		for y in range(img.shape[1]):
			microStruct[x,y,zPos[x,y]] = img[x,y]

	# Convolve the constructed microstructure with a Gaussian to give them a shape
	convolveMS = np.fft.ifftn(fourier_gaussian(np.fft.fftn(microStruct), sigma=sigma2)).real
	binaryMS = np.where(convolveMS < binaryThresh, 0, 1) # binarize

	# Change the orientation such the z-dir is along longer direction (Change XYZ to ZXY)
	# TODO use np.stack
	binaryMSrot = np.zeros((Nz, img.shape[0], img.shape[1]))
	for z in range(Nz):
		binaryMSrot[z,...] = binaryMS[...,z]

	return binaryMSrot
