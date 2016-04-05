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
	source = np.genfromtxt('/Users/saracamnasio/Desktop/Master_Sheet.csv',delimiter=',', skip_header=1, dtype = float)

	# Importing the names from the same table, same order, just as strings
	source_name = np.genfromtxt('/Users/saracamnasio/Desktop/Master_Sheet.csv',delimiter=',', skip_header=1, dtype = str)

	# Grabbing only the coefficients from the table
	coeffs = np.column_stack((source[:,8], source[:,10], source[:,12], source[:,14], source[:,16]))
	names = source_name[:,0]

	# Artificial wavelength array ---- DO I NEED TO USE THE REAL DATA HERE?
	fake_data = np.arange(0, 0.176, 0.0001) # this last value is based on getting as close as possible to the resolution of med res data

	# Loop through coefficients:
	for n in range(len(coeffs)):
		# Converting coefficient values into a polynomial form
		poly = coeffs[n]
		p = np.poly1d(poly)
		poly_fit = p(fake_data)
		# plt.plot(fake_data, poly_fit)

		# Calculating first derivative
		p2 = np.polyder(p, m=1)
		first_der = p2(fake_data)
		first_der = np.array(first_der)
		# plt.plot(fake_data, first_der)

		# Calculating second derivative
		p3 = np.polyder(p, m=2)
		second_der = p3(fake_data)
		second_der = np.array(second_der)
		# plt.plot(fake_data, second_der)

		# Calculating the roots (or critical values of wavelength where min/max are)
		minmax_raw = p2.r
		# print "Minmaxraw is:{0}".format(minmax_raw)
		minmax_corrected = [s+1.15 for s in minmax_raw]
		minmax_corrected = np.array(minmax_corrected)
		# print "Minmax_Corrected is:{0}".format(minmax_corrected)

		# This is the base plot to show where the loop finds min and max
		new_fake_data = [c+1.15 for c in fake_data]
		# plt.figure()
		plt.xlim(1.15, 1.325)
		plt.ylim(0, 1.4)
		plt.plot(new_fake_data, poly_fit, color ='grey', alpha = 0.5)
		plt.scatter(minmax_corrected, p(minmax_raw), color = 'red', alpha = 0.8)
		plt.title("{0} Local max/min".format(names[n]))

		# For each root, do second derivative test to figure out loc max or loc min:
		for i in range(len(minmax_corrected)):
			# Add 1.15 to the whole array of wavelength roots to correct for the subtraction performed in pseudo_fitter when finding a fit
			x = minmax_corrected[i]
			print "x is: {0}".format(x)
			# print "P3 is: {0}".format(p3)

			# Plug in critical values into the second deriv.
			y = p3(x)
			print "p3 is: {0}".format(p3)
			print "y is: {0}".format(y)
			# Differentiating between loc max and loc min:
			if y > 0:
				print "Local minimum is {0}".format(y)
			elif y < 0:
				print "Local maximum is {0}".format(y)
			
		
		# Artificial wavelength array ---- DO I NEED TO USE THE REAL DATA HERE?
		wavelength = np.arange(1.15, 1.326, 0.0001) # this last value is based on getting as close as possible to the resolution of med res data
		
		final = []
		# For each element in the fictional array, find inflection points by finding where the second deriv is equal to zero:
		for element in wavelength:
			# plugging in a fictional arrays of x that matches resolution and range of analysis data 
			m = p3(element)
			flux = final.append(m)
			if m == 0:
				print "There is an inflection point at: {0}".format(element)
			else: 
				continue
		# plt.figure()
		# plt.plot(wavelength, final, color='k')
	 	# np.savetxt()

	# to use when saving plotting result or plotting table:
	#date_format = time.strftime("%d/%m/%Y")