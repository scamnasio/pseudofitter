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
def MC(N, z):
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
	for n in range(N-1, len(data_paths)):
		if source[n,2] == "err":
			name = names[n]
			print "Object: {0}, index {1}".format(name, n)
			data = data_paths[n]
			spectrum = np.genfromtxt(data, delimiter=' ', dtype = float)
			W1 = np.array(spectrum[:,0])
			F1 = np.array(spectrum[:,1])
			U1 = np.array(spectrum[:,2])
			
		elif source[n,2] == "no":
			name = names[n]
			print "Object: {0}, index {1}".format(name, n)
			data = (data_paths[n])
			spectrum = np.genfromtxt(data, delimiter=' ', dtype = float)
			W1 = np.array(spectrum[:,0])
			F1 = np.array(spectrum[:,1])
			U1 = F1 * 0.05

		elif source[n,2] == "snr":
			name = names[n]
			data = (data_paths[n])
			print "Object: {0}, index {1}".format(name, n)
			spectrum = np.genfromtxt(data, delimiter=' ', dtype = float)
			W1 = np.array(spectrum[:,0])
			F1 = np.array(spectrum[:,1])
			U_RAW = np.array(spectrum[:,2])
			U1 = F1/U_RAW

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
		plt.close()
		
		# Loading the W,F and U into a Pyspeckit spectrum:
		medres = pyspeckit.Spectrum(xarr=W, data=F, error=U)
		medres.xarr.xtype ='wavelength'
		medres2 = medres.copy()
	
		# Creating empty arrays for all the values in output:
		 
		
		IP =[]
		lmin = []
		lmax =[]
		coeff0 = []
		coeff1 = []
		coeff2 = []
		coeff3 = []
		coeff4 = []
		outliers = []

		
		# MC loop:
		for i in range(z):
			lmax_raw = [] 
			inf_raw = []
			lmin_raw = []
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
			minmax_raw1 = p2.r
			inf_raw1 = p3.r
			# print "inf_raw is {0} and the min max are: {1}".format(inf_raw, minmax_raw)
			inf_raw2 = inf_raw1[inf_raw1>1.10]
			inf_raw = inf_raw2[inf_raw2<1.33]
			if max(minmax_raw1)>1.33:
				print "Value is greater than boundaries: {0}".format(max(minmax_raw1))
			elif min(minmax_raw1)<1.10:
				print "Value is lower than boundaries: {0}".format(min(minmax_raw1))
			else: 
				pass
			minmax_raw2 = minmax_raw1[minmax_raw1>1.10]
			minmax_raw = minmax_raw2[minmax_raw2<1.33]

			if len(inf_raw)>=2:
				# print "More than one inflection points: {0}. They are {1}".format(len(inf_raw), inf_raw)
				IP_interest = inf_raw[0]
				IP.append(IP_interest)
			elif len(inf_raw)<=1:
				# print "Just one inflection point: {0}".format(inf_raw)
				IP.append(inf_raw[0])
			else:
				pass


			# Differentiating between loc max and loc min through 2nd deriv test:
			
			for l in range(len(minmax_raw)):
				x = minmax_raw[l]
				y = p3(x)
				if y > 0:
					lmin_raw.append(x)
				elif y < 0:
					lmax_raw.append(x)
				else:
					pass

			if len(lmin_raw)>=2:
				lmin_raw = np.flipud(lmin_raw)
				# print "More than one local minima: {0}. They are {1}".format(len(lmin_raw), lmin_raw)
				lmin_interest = lmin_raw[0]
				lmin.append(lmin_interest)
				# print "lowest minima is {0}".format(lmin_raw[0])
			elif len(lmin_raw)==1:
				# print "Just one local minimum point: {0}".format(lmin_raw)
				lmin.append(lmin_raw[0])
			else: 
				print "Warning: 0 values for lmin"

			if len(lmax_raw)>=2:
				lmax_raw = np.flipud(lmax_raw)
				# print "More than one local minima: {0}. They are {1}".format(len(lmax_raw), lmax_raw)
				lmax_interest = lmax_raw[0]
				lmax.append(lmax_interest)
			elif len(lmax_raw)==1:
				# print "Just one local minimum point: {0}".format(lmax_raw)
				lmax.append(lmax_raw[0])
			else:
				print "Warning: 0 values for lmax"
				
			# Keeping track of the varying spectrum due to MC loop:
			# new_flux = medres2.baseline.basespec
			# plt.plot(W, new_flux, color='red', alpha=0.1)
			# plt.scatter(lmin, p(lmin), color = 'green', alpha = 1)
			# plt.scatter(lmax, p(lmax), color = 'orange', alpha = 1)
			# plt.scatter(IP, p(IP), color = 'blue', alpha = 1)
			

		print "the length of the arrays: {0},{1},{2}".format(len(lmax), len(lmin), len(IP))

		# This provides the black base spectrum to be overplotted with the red fits in the line above, leave here!
		# plt.plot(W, F, color='black')
		# plt.scatter(lmin, p(lmin), color = 'green', alpha = 1)
		# plt.scatter(lmax, p(lmax), color = 'orange', alpha = 1)
		# plt.scatter(IP, p(IP), color = 'blue', alpha = 1)
		
		# Keeping it real:

		lmin = (np.real(lmin))
		lmax = (np.real(lmax))
		IP = (np.real(IP))

		print "the length of the arrays: {0},{1},{2}".format(len(lmax), len(lmin), len(IP))
	

		# This calculates mean and standard deviation for all of the coefficients & critical points
		mu0,sigma0 = norm.fit(coeff0)
		mu1,sigma1 = norm.fit(coeff1)
		mu2,sigma2 = norm.fit(coeff2)
		mu3,sigma3 = norm.fit(coeff3)
		mu4,sigma4 = norm.fit(coeff4)
		mu5,sigma5 = norm.fit(IP)
		mu6,sigma6 = norm.fit(lmax)
		mu7,sigma7 = norm.fit(lmin)

		# Trimming out any statistical  outliers:

		# value_1 = [lmin_raw, mu7, sigma7, lmin]
		# value_2 = [lmax_raw, mu6, sigma6, lmax]
		# value_3 = [IP_raw, mu5, sigma5, IP]

		''' 
		legend for this part of the code:
		v = array of lmin, lmax or IP
		j = mean value of array of lmin, lmax or IP
		k = sigma of array of lmin, lmax or IP
		d = [0,1000] index
		h = individual value of lmin, lmax, or IP


		'''
		at.random_subset(lmin,950)
# 


		# This plots the histogram distribution of the data (much like the corner plot). It's a sanity check to see if std and mu make sense.
		# plt.figure()
		# plt.hist(coeff0, 10, normed=True, facecolor='orange', histtype='stepfilled')
		# plt.title("Coefficient 0")
		# plt.ylabel('Probability')
		# plt.xlabel('Coefficient value')
		# plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_hist_0th.png'.format(n, name), format='png')
		# plt.close()
	
		# plt.figure()
		# plt.hist(coeff1, 10, normed=True, facecolor='orange', histtype='stepfilled') 
		# plt.title("Coefficient 1")
		# plt.ylabel('Probability')
		# plt.xlabel('Coefficient value')
		# plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_hist_1st.png'.format(n, name), format='png')
		# plt.close()

		# plt.figure()
		# plt.hist(coeff2, 10, normed=True, facecolor='orange', histtype='stepfilled')
		# plt.title("Coefficient 2")
		# plt.ylabel('Probability')
		# plt.xlabel('Coefficient value')
		# plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_hist_2nd.png'.format(n, name), format='png')
		# plt.close()
		
		# plt.figure()
		# plt.hist(coeff3, 10, normed=True, facecolor='orange', histtype='stepfilled') 
		# plt.title("Coefficient 3")
		# plt.ylabel('Probability')
		# plt.xlabel('Coefficient value')
		# plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_hist_3rd.png'.format(n, name), format='png')
		# plt.close()
	
	
		# plt.figure()
		# plt.hist(coeff4, 10, normed=True, facecolor='orange', histtype='stepfilled')
		# plt.title("Coefficient 4")
		# plt.ylabel('Probability')
		# plt.xlabel('Coefficient value')
		# plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_hist_4th.png'.format(n, name), format='png')
		# plt.close()
 		
 		plt.figure()
		plt.hist(IP, 10, normed=True, facecolor='orange', histtype='stepfilled')
		plt.title("Inflection Point")
		plt.ylabel('Probability')
		plt.xlabel('IP value')
		plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_hist_IP.png'.format(n, name), format='png')
		# plt.close()

		plt.figure()
		plt.hist(lmax, 10, normed=True, facecolor='orange', histtype='stepfilled')
		plt.title("Local Maximum")
		plt.ylabel('Probability')
		plt.xlabel('Lmax value')
		plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_hist_lmax.png'.format(n, name), format='png')
		# plt.close()

		plt.figure()
		plt.hist(lmin, 10, normed=True, facecolor='orange', histtype='stepfilled')
		plt.title("Local Minimum")
		plt.ylabel('Probability')
		plt.xlabel('Lmin value')
		plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_hist_lmin.png'.format(n, name), format='png')
		# plt.close()

		coeff0 = np.array(coeff0)
		coeff1 = np.array(coeff1)
		coeff2 = np.array(coeff2)
		coeff3 = np.array(coeff3)
		coeff4 = np.array(coeff4)
		IP = np.array(IP)
		lmin = np.array(lmin)
		lmax = np.array(lmax)

		# Making corner plot:
		# coeff_MC = np.vstack([coeff0, coeff1, coeff2, coeff3, coeff4])
		# coeff_MC2 =  np.transpose(coeff_MC)
# 
		# figure = corner.corner(coeff_MC2, labels=[r"$0th Coefficient$", r"$1st Coefficient$", r"$2nd Coefficient$", r"$3rd Coefficient$", r"$4th Coefficient$"], quantiles=[0.16, 0.5, 0.84], plot_contours=True, label_args={'fontsize':15}, color='black')
		# figure.gca().annotate("MC Uncertainty Analysis of {0} Pyspeckit Fitting".format(name), xy=(1.2, 1.0), xycoords="figure fraction", xytext=(0, -5), textcoords="offset points", ha="center", va="top")
		# plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_MCplot_coeffs.png'.format(n, name), format='png')
		# plt.close()
# 
		# CP_MC = np.vstack([IP, lmin, lmax])
		# CP_MC2 =  np.transpose(CP_MC)

		# figure2 = corner.corner(CP_MC2, labels=[r"$Inflection Point$", r"$Local Min$", r"$Local max$"], quantiles=[0.16, 0.5, 0.84], plot_contours=True, label_args={'fontsize':15}, color='black')
		# figure2.gca().annotate("MC Uncertainty Analysis of {0} Pyspeckit Fitting".format(name), xy=(1.2, 1.0), xycoords="figure fraction", xytext=(0, -5), textcoords="offset points", ha="center", va="top")
		# plt.savefig('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_MC_CPpoints.png'.format(n, name), format='png')
		# plt.close()
# 
		output_arrays = np.vstack([coeff0, coeff1, coeff2, coeff3, coeff4, lmin, lmax, IP])
		output_arrays = np.transpose(output_arrays)
		np.savetxt('/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/{0}_{1}/{1}_unc_arrays.txt'.format(n, name), output_arrays)

		results_row = [n, name, mu0, sigma0, mu1, sigma1, mu2, sigma2, mu3, sigma3, mu4, sigma4, mu5, sigma5, mu6, sigma6, mu7, sigma7]

		with open("/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/Results.csv", "a") as fp:
 			wr = csv.writer(fp, dialect='excel')
 			wr.writerow(results_row)
