# code to look at residuals of power spectrum 
# after being processed by the UNet

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
import healpy as hp

## Some plotting formatting stuff
plt.rc('font', **{'size': 8, 'sans-serif': ['Helvetica'], 'family': 'sans-serif'})                                          
plt.rc("text.latex", preamble=["\\usepackage{helvet}\\usepackage[T1]{fontenc}\\usepackage{sfmath}"])
plt.rc("text", usetex=True)
plt.rc('ps', usedistiller='xpdf')
plt.rc('savefig', **{'dpi': 300})
plt.style.use('dark_background')

if __name__ == '__main__':
	
	N_NU = 30
	# get the spetrum of frequenies covered in units of MHz
	(bn,nu_bot,nu_top,z_bot,z_top) = np.loadtxt("./sim_info/nuTable.txt").T
	nu_arr = ((nu_bot + nu_top)/2.)[:-1]
	nu_arr = np.array([np.mean(i,axis=0) for i in np.split(nu_arr,N_NU)])

	pca = np.load("sim_data/x_val_nepochs100.npy")
	cosmo = np.load("sim_data/y_val_nepochs100.npy")
	pred = np.load("sim_data/y_pred_nepochs100.npy")
	rearr = np.array(np.load("rearr.npy"),dtype=int)

	for i in range(N_NU):
		# Get Cls for PCA spectrum
		pca0 = (pca.T[i].T).flatten()
		pca0 = pca0[rearr]
		alm_pca = hp.map2alm(pca0)
		Cl_pca = hp.alm2cl(alm_pca)

		# Get Cls for COSMO spectrum
		cosmo0 = (cosmo.T[i].T).flatten()
		cosmo0 = cosmo0[rearr]
		alm_cosmo = hp.map2alm(cosmo0)
		Cl_cosmo = hp.alm2cl(alm_cosmo)

		# Get Cls for PCA spectrum
		pred0 = (pred.T[i].T).flatten()
		pred0 = pred0[rearr]
		alm_pred = hp.map2alm(pred0)
		Cl_pred = hp.alm2cl(alm_pred)
		fig = plt.figure()
		plt.plot(((Cl_pca-Cl_cosmo)/Cl_cosmo)[1:],c=plt.cm.bwr(.9),label="PCA")
		plt.plot(((Cl_pred-Cl_cosmo)/Cl_cosmo)[1:],c=plt.cm.bwr(.1),label="UNet")
		plt.axhline(c="w")
		#plt.xscale("log")
		plt.xlim(1,1000)
		plt.ylim(-.3,.3)
		plt.xlabel(r"$\ell$")
		plt.ylabel(r"$(C_{\ell,{\rm true}} - C_{\ell,{\rm pred}})/C_{\ell,{\rm true}}$")
		plt.title(r"$\nu$ = %d MHz"%(nu_arr[i]))
		plt.legend()
		plt.gcf().set_size_inches((1.25*3.37, 3.37))
		plt.tight_layout()
		#plt.savefig("Figures/Cell_comp%02d.png"%i)