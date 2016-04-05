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

def derivative():

	''' This function outputs local max, min, and inflection points in terms of wavelength '''

	# Grabbing the coefficients from the analysis results table
	source = np.genfromtxt('/Users/saracamnasio/Desktop/Master_Sheet.csv',delimiter=',', skip_header=1, dtype = float)

	# Importing the names from the same table, same order, just as strings
	source_name = np.genfromtxt('/Users/saracamnasio/Desktop/Master_Sheet.csv',delimiter=',', skip_header=1, dtype = str)

	# Grabbing only the coefficients from the table
	coeffs = np.column_stack((source[:,8], source[:,10], source[:,12], source[:,14], source[:,16]))
	names = source_name[:,0]

	# Loop through coefficients:
	for n in range(len(coeffs)):
		# Converting coefficient values into a polynomial form
		p = np.poly1d(coeffs[n])

		# Calculating first derivative
		p2 = np.polyder(p, m=1)

		# Calculating second derivative
		p3 = np.polyder(p, m=2)

		# Calculating the roots
		minmax_raw = p2.r

		# For each root, do second derivative test to figure out loc max or loc min:
		for i in range(len(minmax_raw)):
			# Add 1.15 to the whole array of wavelength roots to correct for the subtraction performed in pseudo_fitter when finding a fit
			x = minmax_raw[i] + 1.15

			# Plug in critical values into the second deriv.
			y = p3(x)

			# Differentiating between loc max and loc min:
			if y > 0:
				print "Local minimum is {0}".format(y)
			elif y < 0:
				print "Local maximum is {0}".format(y)
		
		# Artificial wavelength array ---- DO I NEED TO USE THE REAL DATA HERE?
		wavelength = np.arange(1.15, 1.326, 0.0001) # this last value is based on getting as close as possible to the resolution of med res data
		
		# For each element in the fictional array, find inflection points by finding where the second deriv is equal to zero:
		for element in wavelength:

			# plugging in a fictional arrays of x that matches resolution and range of analysis data 
			m = p3(wavelength[element])
			if m == 0:
				print "There is an inflection point at: {0}".format(wavelength)
			else: 
				continue

	 	# np.savetxt()

	# to use when saving plotting result or plotting table:
	#date_format = time.strftime("%d/%m/%Y")