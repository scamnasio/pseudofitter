''' 
Written by Sara Camnasio
CUNY Hunter College class of 2016
sara.camnasio@gmail.com
'''

import numpy as np
import astrotools2 as at
from matplotlib import pyplot as plt
from numpy import *
import utilities as u
import pyspeckit
import StringIO
import corner
import shutil
import os
import decimal
from scipy.stats import norm
from astropy.io import fits

# db = astrodb.get_db('/Users/saracamnasio/Dropbox/BDNYCdb/BDNYC.db')

# data = np.genfromtxt("/Users/saracamnasio/Research/Projects/UnusuallyRB/Source_Data/2M2151+34.txt", delimiter='', dtype = float)
def MC():
	'''
	*n*
		int - number of MC iterations
	'''
	path = "/Users/saracamnasio/Dropbox/DATA/Master_data.csv"
	source = np.genfromtxt(path, delimiter=',', dtype = str)
	data_paths = source[:,1]
	names = source[:,0]
	
	
	for n in range(len(data_paths)):
		if source[n,2] == "err":
			name = names[n]
			data = data_paths[n]
			spectrum = np.genfromtxt(data, delimiter=',', dtype = float)
			W1 = np.array(spectrum[:,0])
			F1 = np.array(spectrum[:,1])
			U1 = np.array(spectrum[:,2])
			print W1, F1 ,U1
			
		elif source[n,2] == "no":
			name = "{0}_estim_unc".format(names[n])
			data = (data_paths[n])
			spectrum = np.genfromtxt(data, delimiter=',', dtype = float)
			W1 = np.array(spectrum[:,0])
			F1 = np.array(spectrum[:,1])
			U1 = F1 * 0.05

		elif source[n,2] == "snr":
			name = names[n]
			data = (data_paths[n])
			spectrum = np.genfromtxt(data, delimiter=',', dtype = float)
			W1 = np.array(spectrum[:,0])
			F1 = np.array(spectrum[:,1])
			U_RAW = np.array(spectrum[:,2])
			U1 = F/U_RAW

		else:
			pass
	
		# Trimming the data
		W2,F2,U2 = [i[np.where(np.logical_and(W1>1.15, W1<1.325))] for i in [W1,F1,U1]]
		# 1.325
		
		# Removing NaN's:
		W2[np.isnan(W2)] = 0
		F2[np.isnan(F2)] = 0
		U2[np.isnan(U2)] = 0
	
		# Recombining the W,F into one array in order to normalize it
		recombined = np.vstack([W2,F2,U2])
		band = [1.15, 1.325]
	
		# Normalizing over *** ENTIRE BAND *** :
		data_clean = recombined
		data = np.array(at.norm_spec(data_clean, band))
	
		# Assigning arrays after normalization:
		W3 = data[:,0]
		F3 = data[:,1]
		U3 = data[:,2]
		
		# Finalizing arrays:
		W = np.squeeze(W3) 
		F = np.squeeze(F3)
		F = np.array(F)
		U = np.squeeze(U3)
		U = np.array(U)
	
		# Check what the spectrum looks like:
		plt.figure()
		plt.errorbar(W, F, yerr=U, color='black') 
		plt.ylabel('Normalized Flux F$_{\lambda}$')
		plt.xlabel('Wavelength ($\mu$m) - $W_0$')
		plt.annotate('{0}'.format(name), xy=(1.26, 0.7), xytext=(1.26, 0.7), color='black', weight='semibold', fontsize=15)
		plt.ylim(1.26,1.4)
		# if not os.path.exists('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}'.format(name)):
			# os.makedirs('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}'.format(name))
		# plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}/{0}_plot.png'.format(name), format='png')
		# plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}/{0}_plot.pdf'.format(name), format='pdf')
		# plt.clf()
		
		# Loading the W,F and U into a spectrum 
		medres = pyspeckit.Spectrum(xarr=W, data=F, error=U)
		medres.xarr.xtype ='wavelength'
		medres2 = medres.copy()
	
		# Creating empty arrays for all the values in output:
		IP = [] 
		lmax = [] 
		lmin = []
		
		coeff0 = []
		coeff1 = []
		coeff2 = []
		coeff3 = []
		coeff4 = []

		IP_names = [] 
		IP_types = [] 
		IP_spt = [] 
		IP_JK = []
		
		lmax_names = []
		lmax_spt = []
		lmax_types = []
		lmax_JK = []
		
		lmin_names = []
		lmin_spt = []
		lmin_types = []
		lmin_JK = []
	
		for i in range(n):
			medres2.data = medres.data + np.random.randn(medres.data.size)*medres.error
			medres2.baseline(xmin=1.15, xmax=1.325, ymin=0.15, ymax=1.4, subtract=False, highlight_fitregion=False, selectregion=True, exclude=[1.167129, 1.1817135, 1.238683, 1.257725], order=4)
			coeffs = medres2.baseline.baselinepars
			 
			poly = coeffs[n]
			p = np.poly1d(poly)
	
			p2 = np.polyder(p, m=1)
			first_der = p2(W)
			first_der = np.array(first_der)
	
			p3 = np.polyder(p, m=2)
			second_der = p3(W)
	
			minmax_raw = p2.r
			inf_raw = p3.r
	
			minmax_corrected = np.array(minmax_corr)
			
			inf_corrected = np.array(inf_corr)
			IP.append(inf_corrected)
			IP_names.append(names[n])
			IP_types.append(types[n])
			IP_spt.append(spt[n])
			IP_JK.append(JK[n])
	
			C0 = coeffs[4]
			C1 = coeffs[3]
			C2 = coeffs[2]
			C3 = coeffs[1]
			C4 = coeffs[0]
			
			coeff0.append(C0)
			coeff1.append(C1)
			coeff2.append(C2)
			coeff3.append(C3)
			coeff4.append(C4)

	
			for i in range(len(minmax_raw)):
				x = minmax_raw[i]
				# print x
				# Plug in critical values into the second deriv.
				y = p3(x)
				# Differentiating between loc max and loc min:
				if y > 0:
					# print "Local minimum is {0}".format(x)
					lmin.append(x)
					lmin_names.append(names[n])
					lmin_types.append(types[n])
					lmin_spt.append(spt[n])
					lmin_JK.append(JK[n])
	
				elif y < 0:
					lmax.append(x)
					lmax_types.append(types[n])
					lmax_names.append(names[n])
					lmax_spt.append(spt[n])
					lmax_JK.append(JK[n])
				else: 
					pass
			
			new_flux = medres2.baseline.basespec
			plt.plot(W, new_flux, color='red', alpha=0.4)
			
		
		# This provides the black base spectrum to be overplotted with the red fits in the line above, leave here!
		plt.plot(W, F, color='black')
		# plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}/{0}_specfit.png'.format(name), format='png')
		
		# This calculates mean and standard deviation for all of the coefficients & critical points
		mu0,sigma0 = norm.fit(coeff0)
		mu1,sigma1 = norm.fit(coeff1)
		mu2,sigma2 = norm.fit(coeff2)
		mu3,sigma3 = norm.fit(coeff3)
		mu4,sigma4 = norm.fit(coeff4)
		mu5,sigma5 = norm.fit(IP)
		mu6,sigma6 = norm.fit(lmax)
		mu7,sigma7 = norm.fit(lmin)

		# This plots the histogram distribution of the data (much like the corner plot). It's a sanity check to see if std and mu make sense.
		plt.figure()
		plt.hist(coeff0, 10, normed=True, facecolor='orange', histtype='stepfilled')
		plt.title("Coefficient 0")
		plt.ylabel('Probability')
		plt.xlabel('Coefficient value')
		# plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}/{0}_hist_0th.png'.format(name), format='png')
	
	
		plt.figure()
		plt.hist(coeff1, 10, normed=True, facecolor='orange', histtype='stepfilled') 
		plt.title("Coefficient 1")
		plt.ylabel('Probability')
		plt.xlabel('Coefficient value')
		# plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}/{0}_hist_1st.png'.format(name), format='png')
	
		plt.figure()
		plt.hist(coeff2, 10, normed=True, facecolor='orange', histtype='stepfilled')
		plt.title("Coefficient 2")
		plt.ylabel('Probability')
		plt.xlabel('Coefficient value')
		# plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}/{0}_hist_2nd.png'.format(name), format='png')
		
		plt.figure()
		plt.hist(coeff3, 10, normed=True, facecolor='orange', histtype='stepfilled') 
		plt.title("Coefficient 3")
		plt.ylabel('Probability')
		plt.xlabel('Coefficient value')
		# plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}/{0}_hist_3rd.png'.format(name), format='png')
	
	
		plt.figure()
		plt.hist(coeff4, 10, normed=True, facecolor='orange', histtype='stepfilled')
		plt.title("Coefficient 4")
		plt.ylabel('Probability')
		plt.xlabel('Coefficient value')
		# plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}/{0}_hist_4th.png'.format(name), format='png')
 		
 		plt.figure()
		plt.hist(IP, 10, normed=True, facecolor='orange', histtype='stepfilled')
		plt.title("Inflection Point")
		plt.ylabel('Probability')
		plt.xlabel('IP value')

		plt.figure()
		plt.hist(lmax, 10, normed=True, facecolor='orange', histtype='stepfilled')
		plt.title("Local Maximum")
		plt.ylabel('Probability')
		plt.xlabel('Lmax value')

		plt.figure()
		plt.hist(lmin, 10, normed=True, facecolor='orange', histtype='stepfilled')
		plt.title("Local Minimum")
		plt.ylabel('Probability')
		plt.xlabel('Lmin value')

		coeff0 = np.array(coeff0)
		coeff1 = np.array(coeff1)
		coeff2 = np.array(coeff2)
		coeff3 = np.array(coeff3)
		coeff4 = np.array(coeff4)
		IP = np.array(IP)
		lmin = np.array(lmin)
		lmax = np.array(lmax)
		
		coeff_MC = np.vstack([coeff0, coeff1, coeff2, coeff3, coeff4])
		coeff_MC2 =  np.transpose(coeff_MC)

		figure = corner.corner(coeff_MC2, labels=[r"$0th Coefficient$", r"$1st Coefficient$", r"$2nd Coefficient$", r"$3rd Coefficient$", r"$4th Coefficient$"], quantiles=[0.16, 0.5, 0.84], plot_contours=True, label_args={'fontsize':15}, color='black')
		figure.gca().annotate("MC Uncertainty Analysis of {0} Pyspeckit Fitting".format(name), xy=(1.2, 1.0), xycoords="figure fraction", xytext=(0, -5), textcoords="offset points", ha="center", va="top")

		# plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}/{0}_MCplot_coeffs.png'.format(name), format='png')

		CP_MC = np.vstack([IP, lmax, lmin])
		CP_MC2 =  np.transpose(CP_MC)

		figure = corner.corner(CP_MC2, labels=[r"$Inflection Point$", r"$Local Maximum$", r"$Local Minimum$"], quantiles=[0.16, 0.5, 0.84], plot_contours=True, label_args={'fontsize':15}, color='black')
		figure.gca().annotate("MC Uncertainty Analysis of {0} Critical Points".format(name), xy=(1.2, 1.0), xycoords="figure fraction", xytext=(0, -5), textcoords="offset points", ha="center", va="top")

		# plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}/{0}_MCplot_CP.png'.format(name), format='png')
		
		# print '{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10} '.format(name, mu4, sigma4, mu3, sigma3, mu2, sigma2, mu1, sigma1, mu0, sigma0)
		new_array = np.vstack([coeff0, coeff1, coeff2, coeff3, coeff4, IP, lmax, lmin])
		new_array = np.transpose(new_array)
		# np.savetxt('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}/{0}_unc_arrays'.format(name), new_array)

def fits(n, input):
	from astropy.io import fits
	'''
	*n*
		int - number of MC iterations
	*input*
		str - full path of obj spectra path (fits file)

	'''

	hdu_list=fits.open(input)
	image_data=hdu_list[0].data
	W1 = image_data[0]
	F1 = image_data[1]
	U1 = image_data[2]

	
	path_name = input.split('/')
	name_1 = path_name[6]
	name_2 = name_1.split('.')
	name =  name_2[0]
	
	hdu_list.close()

	'''
	If Angstroms:
	'''
	# W1 = W1/10000

	'''
	- If U1 does not exist (data without uncertainty) please comment out line 42 ("U = np.array(raw[:,2])") and uncomment the first line below
	- If U1 is SNR uncomment the second line
	'''

	# U1 = 0.05*F1
	# U1 = F1/U1

	# Trimming the data
	W2,F2,U2 = [i[np.where(np.logical_and(W1>1.15, W1<1.325))] for i in [W1,F1,U1]]
	print W2,F2,U2
	W2[np.isnan(W2)] = 0
	F2[np.isnan(F2)] = 0
	U2[np.isnan(U2)] = 0

	# Recombining the W,F into one array in order to normalize it
	recombined = np.vstack([W2,F2, U2])
	band = [1.15, 1.325]

	data_clean = recombined
	data = np.array(at.norm_spec(data_clean, band))

	W3 = data[:,0]
	F3 = data[:,1]
	U3 = data[:,2]
	
	# Squeezing into one dimension and shifting the wavelength to the 0 axis
	W = np.squeeze(W3) 
	W[:] = [x - 1.15 for x in W] #original was 1.15, 1.24 is to test Kraus idea  
	F = np.squeeze(F3)
	F = np.array(F)
	U = np.squeeze(U3)
	U = np.array(U)

	# Check what the spectrum looks like:
	plt.figure()
	plt.errorbar(W, F, yerr=U, color='black') 
	plt.ylabel('Normalized Flux F$_{\lambda}$')
	plt.xlabel('Wavelength ($\mu$m) - $W_0$')
	plt.annotate('{0}'.format(name), xy=(0.11, 0.7), xytext=(0.11, 0.7), color='black', weight='semibold', fontsize=15)
	plt.ylim(0.15,1.4)
	
	# Loading the W,F and U into a spectrum 
	medres = pyspeckit.Spectrum(xarr=W, data=F, error=U)
	medres.xarr.xtype ='wavelength'
	medres2 = medres.copy()

	coeff0 = []
	coeff1 = []
	coeff2 = []
	coeff3 = []
	coeff4 = []

	for i in range(n):
		medres2.data = medres.data + np.random.randn(medres.data.size)*medres.error
		medres2.baseline(xmin=0, xmax=0.175, ymin=0.15, ymax=1.4, subtract=False, highlight_fitregion=False, selectregion=True, exclude=[0.017129, 0.0317135, 0.088683, 0.107725], order=4)
		coeffs = medres2.baseline.baselinepars
		C0 = coeffs[4]
		C1 = coeffs[3]
		C2 = coeffs[2]
		C3 = coeffs[1]
		C4 = coeffs[0]
		coeff0.append(C0)
		coeff1.append(C1)
		coeff2.append(C2)
		coeff3.append(C3)
		coeff4.append(C4)
		# print C0
		new_flux = medres2.baseline.basespec
		plt.plot(W, new_flux, color='red', alpha=0.4)
		
	if not os.path.exists('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/{0}'.format(name)):
		os.makedirs('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/{0}'.format(name))

	# This provides the black base spectrum to be overplotted with the red fits in the line above, leave here!
	plt.plot(W, F, color='black')
	plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/{0}/{0}_specfit.png'.format(name), format='png')
	
	# This calculates mean and standard deviation for all of the coefficients 
	mu0,sigma0 = norm.fit(coeff0)
	mu1,sigma1 = norm.fit(coeff1)
	mu2,sigma2 = norm.fit(coeff2)
	mu3,sigma3 = norm.fit(coeff3)
	mu4,sigma4 = norm.fit(coeff4)
	
	# This plots the histogram distribution of the data (much like the corner plot). It's a sanity check to see if std and mu make sense.
	plt.figure()
	plt.hist(coeff0, 10, normed=True, facecolor='orange', histtype='stepfilled')
	plt.title("Coefficient 0")
	plt.ylabel('Probability')
	plt.xlabel('Coefficient value')
	plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/{0}/{0}_hist_0th.png'.format(name), format='png')


	plt.figure()
	plt.hist(coeff1, 10, normed=True, facecolor='orange', histtype='stepfilled') 
	plt.title("Coefficient 1")
	plt.ylabel('Probability')
	plt.xlabel('Coefficient value')
	plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/{0}/{0}_hist_1st.png'.format(name), format='png')

	plt.figure()
	plt.hist(coeff2, 10, normed=True, facecolor='orange', histtype='stepfilled')
	plt.title("Coefficient 2")
	plt.ylabel('Probability')
	plt.xlabel('Coefficient value')
	plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/{0}/{0}_hist_2nd.png'.format(name), format='png')
	
	plt.figure()
	plt.hist(coeff3, 10, normed=True, facecolor='orange', histtype='stepfilled') 
	plt.title("Coefficient 3")
	plt.ylabel('Probability')
	plt.xlabel('Coefficient value')
	plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/{0}/{0}_hist_3rd.png'.format(name), format='png')


	plt.figure()
	plt.hist(coeff4, 10, normed=True, facecolor='orange', histtype='stepfilled')
	plt.title("Coefficient 4")
	plt.ylabel('Probability')
	plt.xlabel('Coefficient value')
	plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/{0}/{0}_hist_4th.png'.format(name), format='png')
	
	# print mu1, sigma1
	# print mu2, sigma2
 
	coeff0 = np.array(coeff0)
	coeff1 = np.array(coeff1)
	coeff2 = np.array(coeff2)
	coeff3 = np.array(coeff3)
	coeff4 = np.array(coeff4)
	
	coeff_MC = np.vstack([coeff0, coeff1, coeff2, coeff3, coeff4])
	coeff_MC2 =  np.transpose(coeff_MC)
	figure = corner.corner(coeff_MC2, labels=[r"$0th Coefficient$", r"$1st Coefficient$", r"$2nd Coefficient$", r"$3rd Coefficient$", r"$4th Coefficient$"], quantiles=[0.16, 0.5, 0.84], plot_contours=True, label_args={'fontsize':15}, color='black')
	figure.gca().annotate("MC Uncertainty Analysis of {0} Pyspeckit Fitting".format(name), xy=(0.5, 1.0), xycoords="figure fraction", xytext=(0, -5), textcoords="offset points", ha="center", va="top")
	plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/{0}/{0}_MCplot.png'.format(name), format='png')
	print '{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}'.format(name, mu4, sigma4, mu3, sigma3, mu2, sigma2, mu1, sigma1, mu0, sigma0)

