# for formatting the data with a PCA decomposition 
# of the input data

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
import healpy as hp
from sklearn.decomposition import PCA

def gen_rearr(nside):
	# recursive funtion for finding the right 
	# ordering for the nested pixels 
	if (nside==1):
		return np.array([0,1,2,3])
	else:
		smaller = np.reshape(gen_rearr(nside-1),(2**(nside-1),2**(nside-1)))
		npixsmaller = 2**(2*(nside-1))
		top = np.concatenate((smaller,smaller+npixsmaller),axis=1)
		bot = np.concatenate((smaller+2*npixsmaller,smaller+3*npixsmaller),axis=1)
		whole = np.concatenate((top,bot))
		return whole.flatten()


if __name__ == '__main__':
	# "GLOBAL" parameters
	(NU_L,NU_H) = (1,30)
	DO_NU_AVG = False
	NU_AVG = 30
	assert(((NU_H-NU_L + 1)%NU_AVG) ==0)
	MAP_NSIDE = 256
	WINDOW_NSIDE = 4
	NUM_SIMS = 100
	N_COMP_MASK = 3 # number of PCA components to remove
	# resolution of the outgoing window
	NPIX_WINDOW = (MAP_NSIDE/WINDOW_NSIDE)**2
	# actual side length of window
	WINDOW_LENGTH = int(np.sqrt(NPIX_WINDOW))
	rearr = gen_rearr(int(np.log2(MAP_NSIDE/WINDOW_NSIDE)))
	nwinds = hp.nside2npix(WINDOW_NSIDE)
	# "global" string with name to disk directory
	dirstr = "/Volumes/My Passport for Mac/lachlanl/21cm_project/sims"
	# initialize the PCA algorithm
	pca = PCA()
	
	x_out = np.zeros((NUM_SIMS*nwinds,64,64,30))
	for SNUM in np.arange(1,NUM_SIMS + 1):
		# Open the Fits files for foreground and ccosmological signal
		fgd = np.array([fits.getdata("%s/run_fg_s1%03d/fg_%03d.fits"%(dirstr,SNUM,nu+1),1) for nu in range(N_NU)],dtype=np.float64)
		cosmo = np.array([fits.getdata("%s/run_pkEH_s1%03d/cosmo_%03d.fits"%(dirstr,SNUM,nu+1),1) for nu in range(N_NU)],dtype=np.float64)
		# average in frequency bins and transpose
		fgd = np.array([np.mean(i,axis=0) for i in np.split(fgd,NU_AVG)]).T
		cosmo = np.array([np.mean(i,axis=0) for i in np.split(cosmo,NU_AVG)]).T
		# create the observed signal as a sum of the forground and cosmological signal
		obs = fgd + cosmo
		pca.fit(obs)

		obs_pca = pca.transform(obs)
		ind_arr = np.reshape(np.arange(np.prod(obs_pca.shape)),obs_pca.shape)
		mask = np.ones(obs_pca.shape)
		for i in range(N_COMP_MASK,obs_pca.shape[1]):
			mask[ind_arr%obs_pca.shape[1]==i] = 0
		obs_pca = obs_pca*mask
		obs_pca_red = pca.inverse_transform(obs_pca)
		print "Now I'm doing the minimum subtraction..."
		obs_pca_red = obs - obs_pca_red

		# get the array indices in the RING formulation
		inds = np.arange(hp.nside2npix(MAP_NSIDE))
		# transfer these to what they would be in the NESTED formulation
		inds_nest = hp.ring2nest(MAP_NSIDE,inds)
		

		for PIX_SELEC in np.arange(hp.nside2npix(WINDOW_NSIDE)):
			# get the indices of the pxixels which actually are in the larger pixel
			inds_in = np.where((inds_nest/NPIX_WINDOW)==PIX_SELEC)
			to_rearr_inds = inds_nest[inds_in] - PIX_SELEC*NPIX_WINDOW
			to_rearr = obs_pca_red[inds_in]
			to_rearr = (to_rearr[np.argsort(to_rearr_inds)])[rearr]
			to_rearr = np.reshape(to_rearr,(WINDOW_LENGTH,WINDOW_LENGTH,NU_AVG))
			ind = (SNUM-1)*nwinds + PIX_SELEC
			x_out[ind] = to_rearr

	np.save("%s/pca_reduced_nsim%d"%(dirstr,NUM_SIMS),x_out)
	
