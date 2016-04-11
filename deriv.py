''' 
Written by Sara Camnasio
CUNY Hunter College class of 2016
sara.camnasio@gmail.com

Current to do list 04/04/2016:
- FIXED (It doesn't): I am pretty sure poly1d rounds up coefficients. Need to fix that
- Add second derivative test for min max
- Local maximum is defined when the first derivative (p2) is equal to 0 and the second derivative at p2 (f''(p2)) is less than 0
- Local minimum is defined when the first derivative (p2) is equal to 0 and the second derivative at p2 (f''(p2)) is more than 0
- An inflection point is the x value when the second derivative is equal to 0
- DONE --- Add 1.15 to results to get original wavelength 
'''

import numpy as np
import time
from matplotlib import pyplot as plt

def derivative():

	''' This function outputs local max, min, and inflection points in terms of wavelength '''

	# Grabbing the coefficients from the analysis results table
	source = np.genfromtxt('/Users/saracamnasio/Desktop/Master_Sheet.csv',delimiter=',', skip_header=1, dtype = str)

	# Importing the names from the same table, same order, just as strings
	# source_name = np.genfromtxt('/Users/saracamnasio/Desktop/Master_Sheet.csv',delimiter=',', skip_header=1, dtype = str)

	# Grabbing only the coefficients from the table
	coeffs = np.column_stack((source[:,8], source[:,10], source[:,12], source[:,14], source[:,16]))
	coeffs = coeffs.astype(np.float)
	names = source[:,0]
	

	spt = np.array(source[:,2])
	spt = spt.astype(np.float)
	types = np.array(source[:,4])
	JK = np.array(source[:,7])
	JK = JK.astype(np.float)


	# Artificial wavelength array ---- DO I NEED TO USE THE REAL DATA HERE?
	fake_data = np.arange(0, 0.176, 0.00001) # this last value is based on getting as close as possible to the resolution of med res data
	IP = []
	lmax = []
	lmin = []

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

	# Loop through coefficients:
	for n in range(len(coeffs)):
		# Converting coefficient values into a polynomial form
		poly = coeffs[n]
		p = np.poly1d(poly)
		poly_fit = p(fake_data)
		# plt.plot(new_fake_data, poly_fit)

		# Calculating first derivative
		p2 = np.polyder(p, m=1)
		first_der = p2(fake_data)
		first_der = np.array(first_der)
		# plt.plot(new_fake_data, first_der)

		# Calculating second derivative
		p3 = np.polyder(p, m=2)
		second_der = p3(fake_data)
		

		# Calculating the roots (or critical values of wavelength where min/max are) & inflection points:
		minmax_raw = p2.r
		inf_raw = p3.r
		# print "inflection points are: {0}".format(inf_raw)
		
		minmax_corr = [s+1.15 for s in minmax_raw]
		minmax_corrected = np.array(minmax_corr)
		
		inf_corr = [g+1.15 for g in inf_raw]
		inf_corrected = np.array(inf_corr)
		IP.append(inf_corrected)
		IP_names.append(names[n])
		IP_types.append(types[n])
		IP_spt.append(spt[n])
		IP_JK.append(JK[n])


		# This is the base plot to show where the loop finds min and max
		new_fake_data = [c+1.15 for c in fake_data]
		
		''' Figures for testing '''

		### Plot of poly fit + min/max:
		# plt.xlim(1.15, 1.325)
		# plt.ylim(0, 1.4)
		# plt.plot(new_fake_data, poly_fit, color ='grey', alpha = 0.5) #poly fit
		# plt.scatter(minmax_corrected, p(minmax_raw), color = 'red', alpha = 0.8) 
		# plt.title("Min/max points on polynomial fit")
		# plt.grid()

		### Plot of poly fit + inf points:
		# plt.xlim(1.15, 1.325)
		# plt.ylim(0, 1.4)
		# plt.plot(new_fake_data, poly_fit, color ='grey', alpha = 0.5) #poly fit
		# plt.scatter(inf_corrected, p(inf_raw), color = 'red', alpha = 0.8) # inf points on poly fit
		# plt.title("Inflection points on polynomial fit")
		# plt.grid()

		### Plot of second deriv
		# plt.xlim(1.15, 1.325)
		# plt.plot(new_fake_data, second_der, color ='grey', alpha = 0.5) # second derivative of poly fit
		# plt.title("Second derivative of polynomial fit")
		# plt.grid()
		
		### Inf points on second derivative:
		# plt.xlim(1.15, 1.325)
		# plt.plot(new_fake_data, second_der, color ='grey', alpha = 0.5)
		# plt.scatter(inf_corrected, p3(inf_raw), color = 'red', alpha = 0.8) # inf points on second derivative
		# plt.title("Inflection points on second derivative of polynomial fit")
		# plt.grid()

		

		# For each root, do second derivative test to figure out loc max or loc min:
		for i in range(len(minmax_raw)):
			x = minmax_raw[i]
			print x
			# Plug in critical values into the second deriv.
			y = p3(x)
			# Differentiating between loc max and loc min:
			if y > 0:
				# print "Local minimum is {0}".format(x)
				lmin_names.append(names[n])
				lmin_types.append(types[n])
				lmin.append(x)
				lmin_spt.append(spt[n])
				lmin_JK.append(JK[n])
				# print "value is: {0}".format(lmin)
				# print "name is: {0}".format(lmin_names)

			elif y < 0:
				# print "Local maximum is {0}".format(x)
				lmax.append(x)
				lmax_types.append(types[n])
				lmax_names.append(names[n])
				lmax_spt.append(spt[n])
				lmax_JK.append(JK[n])
	
	
	categories = []
	shapes = []
	for n in zip(lmin_types, lmin_types, IP_types):

		if n == 'young':
			categories.append("r")
			shapes.append()
		elif n == "blue":
			categories.append("b")
			shapes.append()
		elif n == "red":
			categories.append("r")
			shapes.append()
		elif n == "standard":
			categories.append("k")
			shapes.append()
		elif n == "subdwarf":
			categories.append("b")
			shapes.append()


	lmin = np.real(np.array(lmin))
	lmin = [o+1.15 for o in lmin]
	# print np.size(lmin)
	# print lmin
	lmin_all = np.column_stack((lmin, lmin_names, lmin_types, lmin_spt, lmin_JK))
	# print lmin_all

	lmax = np.real(np.array(lmax))
	lmax = [o+1.15 for o in lmax]
	lmax_all = np.column_stack((lmax, lmax_names, lmax_types, lmax_spt, lmax_JK))

	IP = np.real(np.array(IP))
	IP = IP[:,0]
	IP = [o+1.15 for o in IP]
	IP_all = np.column_stack((IP, IP_names, IP_types, IP_spt, IP_JK))


	# NEED TO WORK ON PLOTTING BELOW
	

	plt.figure()
	plt.subplot(221)
	plt.scatter(lmin_all[:,3], lmin_all[:, 0])
	plt.xlabel("Spectral Type")
	plt.ylabel("Local mininimum")

	plt.subplot(222)
	plt.scatter(lmax_all[:,3], lmax_all[:, 0])
	plt.xlabel("Spectral Type")
	plt.ylabel("Local maximum")

	plt.subplot(223)
	plt.scatter(IP_all[:,3], IP_all[:, 0])
	plt.xlabel("Spectral Type")
	plt.ylabel("Inflection point")

	plt.figure()
	plt.subplot(221)
	plt.scatter(lmin_all[:,4], lmin_all[:,0])
	plt.xlabel("J-K Color")
	plt.ylabel("Local mininimum")

	plt.subplot(222)
	plt.scatter(lmax_all[:,4], lmax_all[:,0])
	plt.xlabel("J-K Color")
	plt.ylabel("Local maximum")

	plt.subplot(223)
	plt.scatter(IP_all[:,4], IP_all[:,0])
	plt.xlabel("J-K Color")
	plt.ylabel("Inflection point")


	# np.savetxt()

	# to use when saving plotting result or plotting table:
	#date_format = time.strftime("%d/%m/%Y")

