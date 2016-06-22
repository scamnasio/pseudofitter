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
import csv

# db = astrodb.get_db('/Users/saracamnasio/Dropbox/BDNYCdb/BDNYC.db')

# data = np.genfromtxt("/Users/saracamnasio/Research/Projects/UnusuallyRB/Source_Data/2M2151+34.txt", delimiter='', dtype = float)
def MC(z):
	'''
	*z*
		int - number of MC iterations
	'''

	# plt.ioff() # --- toggle to speed up code and not show plots
	path = "/Users/saracamnasio/Dropbox/DATA/Master_data.csv"
	source = np.genfromtxt(path, delimiter=',', dtype = str)
	data_paths = source[:,1]
	names = source[:,0]
	
	# Iterating through different uncertainty types:
	for n in range(len(data_paths)):
		if source[n,2] == "err":
			name = names[n]
			print "Object: {0}".format(name)
			data = data_paths[n]
			spectrum = np.genfromtxt(data, delimiter=' ', dtype = float)
			W1 = np.array(spectrum[:,0])
			F1 = np.array(spectrum[:,1])
			U1 = np.array(spectrum[:,2])
			
		elif source[n,2] == "no":
			name = "{0}_estim_unc".format(names[n])
			data = (data_paths[n])
			spectrum = np.genfromtxt(data, delimiter=' ', dtype = float)
			W1 = np.array(spectrum[:,0])
			F1 = np.array(spectrum[:,1])
			U1 = F1 * 0.05

		elif source[n,2] == "snr":
			name = names[n]
			data = (data_paths[n])
			spectrum = np.genfromtxt(data, delimiter=' ', dtype = float)
			W1 = np.array(spectrum[:,0])
			F1 = np.array(spectrum[:,1])
			U_RAW = np.array(spectrum[:,2])
			U1 = F/U_RAW

		else:
			pass
	
	
		# Trimming the data noisy ends:
		W2 = W1[W1>1.15]
		W3 = W2[W2<1.325]

		F2 = F1[W1>1.15]
		F3 = F2[W2<1.325]

		U2 = U1[W1>1.15]
		U3 = U2[W2<1.325]
		
		# Removing NaN's:
		W3[np.isnan(W3)] = 0
		F3[np.isnan(F3)] = 0
		U3[np.isnan(U3)] = 0
	
		# Recombining the W,F into one array in order to normalize it:
		recombined = np.vstack([W3,F3,U3])
		band = [1.15, 1.325]
	
		# Normalizing over *** ENTIRE BAND *** :
		data_clean = recombined
		data = np.array(at.norm_spec(data_clean, band))
	
		# Assigning arrays after normalization:
		W4 = data[:,0]
		F4 = data[:,1]
		U4 = data[:,2]
		
		# Finalizing arrays:
		W = np.squeeze(W4) 
		F = np.squeeze(F4)
		F = np.array(F)
		U = np.squeeze(U4)
		U = np.array(U)
	
		# Check what the spectrum looks like:
		plt.figure()
		plt.errorbar(W, F, yerr=U, color='black') 
		plt.ylabel('Normalized Flux F$_{\lambda}$')
		plt.xlabel('Wavelength ($\mu$m) - $W_0$')
		plt.annotate('{0}'.format(name), xy=(1.26, 0.7), xytext=(1.26, 0.7), color='black', weight='semibold', fontsize=15)
		plt.ylim(0.3,1.4)
		plt.xlim(1.15,1.325)

		# Save the plain spectrum: 
		if not os.path.exists('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}'.format(n, name)):
			os.makedirs('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}'.format(n, name))
		plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_plot.png'.format(n, name), format='png')
		plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_plot.pdf'.format(n, name), format='pdf')
		# plt.close()
		
		# Loading the W,F and U into a Pyspeckit spectrum:
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

		# MC loop:

		for i in range(z):
			medres2.data = medres.data + np.random.randn(medres.data.size)*medres.error
			medres2.baseline(xmin=1.15, xmax=1.325, ymin=0.15, ymax=1.4, subtract=False, highlight_fitregion=False, selectregion=True, exclude=[1.167129, 1.1817135, 1.238683, 1.257725], order=4)
			
			# Grabbing the coefficients of the polynomial:
			coeffs = medres2.baseline.baselinepars

			# Storing the coefficient arrays
			coeff0.append(coeffs[4])
			coeff1.append(coeffs[3])
			coeff2.append(coeffs[2])
			coeff3.append(coeffs[1])
			coeff4.append(coeffs[0])

			# Calculating critical points:
			p = np.poly1d(coeffs) 
			p2 = np.polyder(p, m=1)
			first_der = p2(W)
			first_der = np.array(first_der)
			p3 = np.polyder(p, m=2)
			second_der = p3(W)
			second_der = np.array(second_der)
			minmax_raw = p2.r
			inf_raw = p3.r
			
			# Making sure the inflection points are within range of interest:
			for c in range(len(inf_raw)):
				x = inf_raw[c] 
				if 1.15 <= x <= 1.25:
					IP.append(x)
				else:
					pass

			# Differentiating between loc max and loc min through 2nd deriv test:
			for l in range(len(minmax_raw)):
				x = minmax_raw[l]
				y = p3(x)
				if y > 0:
					if 1.15 <= x <= 1.22: # making sure point is within range of interest 1.15-1.325
						lmin.append(x)
					else:
						pass
				elif y < 0:
					if 1.15 <= x <= 1.325:
						lmax.append(x)
					else:
						pass
		
			
			# Keeping track of the varying spectrum due to MC loop:
			new_flux = medres2.baseline.basespec
			plt.plot(W, new_flux, color='red', alpha=0.1)
			plt.scatter(lmin, p(lmin), color = 'green', alpha = 1)
			plt.scatter(lmax, p(lmax), color = 'orange', alpha = 1)
			plt.scatter(IP, p(IP), color = 'blue', alpha = 1)
			


		# This provides the black base spectrum to be overplotted with the red fits in the line above, leave here!
		plt.plot(W, F, color='black')
		plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_specfit.png'.format(n, name), format='png')
		plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_specfit.pdf'.format(n, name), format='pdf')
		
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
		plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_hist_0th.png'.format(n, name), format='png')
		plt.close()
	
		plt.figure()
		plt.hist(coeff1, 10, normed=True, facecolor='orange', histtype='stepfilled') 
		plt.title("Coefficient 1")
		plt.ylabel('Probability')
		plt.xlabel('Coefficient value')
		plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_hist_1st.png'.format(n, name), format='png')
		plt.close()

		plt.figure()
		plt.hist(coeff2, 10, normed=True, facecolor='orange', histtype='stepfilled')
		plt.title("Coefficient 2")
		plt.ylabel('Probability')
		plt.xlabel('Coefficient value')
		plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_hist_2nd.png'.format(n, name), format='png')
		plt.close()
		
		plt.figure()
		plt.hist(coeff3, 10, normed=True, facecolor='orange', histtype='stepfilled') 
		plt.title("Coefficient 3")
		plt.ylabel('Probability')
		plt.xlabel('Coefficient value')
		plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_hist_3rd.png'.format(n, name), format='png')
		plt.close()
	
	
		plt.figure()
		plt.hist(coeff4, 10, normed=True, facecolor='orange', histtype='stepfilled')
		plt.title("Coefficient 4")
		plt.ylabel('Probability')
		plt.xlabel('Coefficient value')
		plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_hist_4th.png'.format(n, name), format='png')
		plt.close()
 		
 		plt.figure()
		plt.hist(IP, 10, normed=True, facecolor='orange', histtype='stepfilled')
		plt.title("Inflection Point")
		plt.ylabel('Probability')
		plt.xlabel('IP value')
		plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_hist_IP.png'.format(n, name), format='png')
		plt.close()

		plt.figure()
		plt.hist(lmax, 10, normed=True, facecolor='orange', histtype='stepfilled')
		plt.title("Local Maximum")
		plt.ylabel('Probability')
		plt.xlabel('Lmax value')
		plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_hist_lmax.png'.format(n, name), format='png')
		plt.close()

		plt.figure()
		plt.hist(lmin, 10, normed=True, facecolor='orange', histtype='stepfilled')
		plt.title("Local Minimum")
		plt.ylabel('Probability')
		plt.xlabel('Lmin value')
		plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_hist_lmin.png'.format(n, name), format='png')
		plt.close()

		coeff0 = np.array(coeff0)
		coeff1 = np.array(coeff1)
		coeff2 = np.array(coeff2)
		coeff3 = np.array(coeff3)
		coeff4 = np.array(coeff4)
		IP = np.array(IP)
		lmin = np.array(lmin)
		lmax = np.array(lmax)

		# Making corner plot:
		coeff_MC = np.vstack([coeff0, coeff1, coeff2, coeff3, coeff4])
		coeff_MC2 =  np.transpose(coeff_MC)

		figure = corner.corner(coeff_MC2, labels=[r"$0th Coefficient$", r"$1st Coefficient$", r"$2nd Coefficient$", r"$3rd Coefficient$", r"$4th Coefficient$"], quantiles=[0.16, 0.5, 0.84], plot_contours=True, label_args={'fontsize':15}, color='black')
		figure.gca().annotate("MC Uncertainty Analysis of {0} Pyspeckit Fitting".format(name), xy=(1.2, 1.0), xycoords="figure fraction", xytext=(0, -5), textcoords="offset points", ha="center", va="top")
		plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_MCplot_coeffs.png'.format(n, name), format='png')

		output_arrays = np.vstack([coeff0, coeff1, coeff2, coeff3, coeff4, lmin, lmax, IP])
		output_arrays = np.transpose(output_arrays)
		np.savetxt('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_unc_arrays.txt'.format(n, name), output_arrays)

		results_row = [name, mu0, sigma0, mu1, sigma1, mu2, sigma2, mu3, sigma3, mu4, sigma4, mu5, sigma5, mu6, sigma6, mu7, sigma7]

		with open("/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/Results.csv", "a") as fp:
    		wr = csv.writer(fp, dialect='excel')
    		wr.writerow(results_row)
