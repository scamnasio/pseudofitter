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
def MC(n):
	'''
	*n*
		int - number of MC iterations
	'''

	path_input = input("Enter last bracket of obj path:")
	path = "/Users/saracamnasio/Desktop/New_data/{0}".format(path_input)	
	# path = "/Users/saracamnasio/Research/Projects/UnusuallyRB/Source_Data/{0}".format(path_input)
	path_name2 = path_input.split('.')
	name = path_name2[0]
	
	raw = np.genfromtxt(path, delimiter='', dtype = float)
	W1 = np.array(raw[:,0])
	F1 = np.array(raw[:,1])
	U1 = np.array(raw[:,2])

	'''
	If Angstroms:
	'''
	# W1 = W1/10000

	'''
	- If U1 does not exist (data without uncertainty) please comment out line 34 ("U = np.array(raw[:,2])") and uncomment the first line below
	- If U1 is SNR uncomment the second line
	'''

	# U1 = 0.05*F1
	U1 = F1/U1

	# Trimming the data
	W2,F2,U2 = [i[np.where(np.logical_and(W1>1.15, W1<1.325))] for i in [W1,F1,U1]]
	# 1.325
	
	W2[np.isnan(W2)] = 0
	F2[np.isnan(F2)] = 0
	U2[np.isnan(U2)] = 0

	# Recombining the W,F into one array in order to normalize it
	recombined = np.vstack([W2,F2,U2])
	band = [1.15, 1.325]

	# data_clean = u.scrub(recombined) 
	# had to remove SCRUB because it fucked up baseline

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
	print len(W)

	

	# SNR = input("SNR:")
	# if SNR == "y":
	# 	U = F/U
	# elif SNR == "n/a":
	# 	U = 0.05*F
	# elif SNR == "n":
	# 	U = U

	# Check what the spectrum looks like:
	plt.figure()
	plt.errorbar(W, F, yerr=U, color='black') 
	plt.ylabel('Normalized Flux F$_{\lambda}$')
	plt.xlabel('Wavelength ($\mu$m) - $W_0$')
	plt.annotate('{0}'.format(name), xy=(0.11, 0.7), xytext=(0.11, 0.7), color='black', weight='semibold', fontsize=15)
	plt.ylim(0.15,1.4)
	
	# Loading the W,F and U into a spectrum 
	medres = pyspeckit.Spectrum(xarr=W, data=F, error=U)
	# medres.xarr.unit = 'micron'
	medres.xarr.xtype ='wavelength'
	medres2 = medres.copy()

	coeff0 = []
	coeff1 = []
	coeff2 = []
	coeff3 = []
	coeff4 = []

	for i in range(n):
		medres2.data = medres.data + np.random.randn(medres.data.size)*medres.error
		# medres2.plotter(xmin=0, xmax=0.175, ymin=0.15, ymax=1.4, errstyle='bars', color='grey')
		# plt.ylabel('Flux F$_{\lambda}$')
		# plt.xlabel('Wavelength ($\mu$m) - $W_0$')
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
	print '{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10} '.format(name, mu4, sigma4, mu3, sigma3, mu2, sigma2, mu1, sigma1, mu0, sigma0)

def fits(n, input):
	from astropy.io import fits
	'''
	*n*
		int - number of MC iterations
	*input*
		str - full path of obj spectra path (fits file)

	'''

	hdu_list=fits.open(input)
	
	# print hdu_list.info()
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

	# data_clean = u.scrub(recombined) 
	# had to remove SCRUB because it fucked up baseline

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
	# medres.xarr.unit = 'micron'
	medres.xarr.xtype ='wavelength'
	medres2 = medres.copy()

	coeff0 = []
	coeff1 = []
	coeff2 = []
	coeff3 = []
	coeff4 = []

	for i in range(n):
		medres2.data = medres.data + np.random.randn(medres.data.size)*medres.error
		# medres2.plotter(xmin=0, xmax=0.175, ymin=0.15, ymax=1.4, errstyle='bars', color='grey')
		# plt.ylabel('Flux F$_{\lambda}$')
		# plt.xlabel('Wavelength ($\mu$m) - $W_0$')
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

def compare():

	'''
	WRITE THIS CODE TO COMPARE TWO OBJECTS, SPECIFICALLY NEED TO SEE 2351 '''





	
	'''
	*n*
		int - number of MC iterations
	'''
	
	path_input1 = input("Enter last bracket of obj1 path:")
	path_input2 = input("Enter last bracket of obj2 path:")
	path1 = "/Users/saracamnasio/Desktop/New_data/{0}".format(path_input1)
	path2 = "/Users/saracamnasio/Desktop/New_data/{0}".format(path_input2)	
	# path = "/Users/saracamnasio/Research/Projects/UnusuallyRB/Source_Data/{0}".format(path_input)
	path_name1 = path_input1.split('.')
	path_name2 = path_input2.split('.')
	name = path_name2[0]
	name = path_name2[0]
	
	raw1 = np.genfromtxt(path, delimiter='', dtype = float)
	raw2 = np.genfromtxt(path, delimiter='', dtype = float)
	
	W1 = np.array(raw[:,0])
	F1 = np.array(raw[:,1])
	U1 = np.array(raw[:,2])

	W1 = np.array(raw[:,0])
	F1 = np.array(raw[:,1])
	U1 = np.array(raw[:,2])

	'''
	If Angstroms:
	'''
	# W1 = W1/10000

	'''
	- If U1 does not exist (data without uncertainty) please comment out line 34 ("U = np.array(raw[:,2])") and uncomment the first line below
	- If U1 is SNR uncomment the second line
	'''

	# U1 = 0.05*F1
	U1 = F1/U1

	# Trimming the data
	W2,F2,U2 = [i[np.where(np.logical_and(W1>1.15, W1<1.325))] for i in [W1,F1,U1]]
	# 1.325
	
	W2[np.isnan(W2)] = 0
	F2[np.isnan(F2)] = 0
	U2[np.isnan(U2)] = 0

	# Recombining the W,F into one array in order to normalize it
	recombined = np.vstack([W2,F2,U2])
	band = [1.15, 1.325]

	# data_clean = u.scrub(recombined) 
	# had to remove SCRUB because it fucked up baseline

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

	

	# SNR = input("SNR:")
	# if SNR == "y":
	# 	U = F/U
	# elif SNR == "n/a":
	# 	U = 0.05*F
	# elif SNR == "n":
	# 	U = U

	# Check what the spectrum looks like:
	plt.figure()
	plt.errorbar(W, F, yerr=U, color='black') 
	plt.ylabel('Normalized Flux F$_{\lambda}$')
	plt.xlabel('Wavelength ($\mu$m) - $W_0$')
	plt.annotate('{0}'.format(name), xy=(0.11, 0.7), xytext=(0.11, 0.7), color='black', weight='semibold', fontsize=15)
	plt.ylim(0.15,1.4)
	
	# Loading the W,F and U into a spectrum 
	medres = pyspeckit.Spectrum(xarr=W, data=F, error=U)
	# medres.xarr.unit = 'micron'
	medres.xarr.xtype ='wavelength'
	medres2 = medres.copy()

	coeff0 = []
	coeff1 = []
	coeff2 = []
	coeff3 = []
	coeff4 = []

	for i in range(n):
		medres2.data = medres.data + np.random.randn(medres.data.size)*medres.error
		# medres2.plotter(xmin=0, xmax=0.175, ymin=0.15, ymax=1.4, errstyle='bars', color='grey')
		# plt.ylabel('Flux F$_{\lambda}$')
		# plt.xlabel('Wavelength ($\mu$m) - $W_0$')
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
	print '{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10} '.format(name, mu4, sigma4, mu3, sigma3, mu2, sigma2, mu1, sigma1, mu0, sigma0)


