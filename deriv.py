''' 
Written by Sara Camnasio
CUNY Hunter College class of 2016
sara.camnasio@gmail.com

Current to do list 04/04/2016:
- FIXED (It doesn't): I am pretty sure poly1d rounds up coefficients. Need to fix that
- Add second derivative test for min max


'''

import numpy as np
import time

def derivative():
	source = np.genfromtxt('/Users/saracamnasio/Desktop/Master_Sheet.csv',delimiter=',', skip_header=1, dtype = float)
	source_name = np.genfromtxt('/Users/saracamnasio/Desktop/Master_Sheet.csv',delimiter=',', skip_header=1, dtype = str)
	coeffs = np.column_stack((source[:,8], source[:,10], source[:,12], source[:,14], source[:,16]))
	names = source_name[:,0]
	for n in range(len(coeffs)):
		p = np.poly1d(coeffs[n])
		p2 = np.polyder(p, m=1)
		p3 = np.polyder(p, m=2)
		# print p3
		minmax_raw = p2.r
		minmax_trim = minmax_raw[minmax_raw<0.175]
		minmax = minmax_trim[minmax_trim>0]
		# print minmax
		for i in range(len(minmax)):
			# print minmax[i]
			x = minmax[i]
			y = p3(x)
			if y > 0:
				print "Local minimum is {0}".format(y)
			elif y < 0:
				print "Local maximum is {0}".format(y)
			# print i
		x1 = np.arange(0, 0.176, 0.001)
		# print x1
		for element in x1:
			m = p3(element)
			if m == 0:
				print "There is an inflection point at: {0}".format(x1)
			else: 
				continue

			# print minmax[i]
		
		# for i in range(len(p2)):

		
	
		# print names[n], minmax
		# inf_raw = p3.r

	 	# np.savetxt()

	# to use when saving plotting result or plotting table:
	#date_format = time.strftime("%d/%m/%Y")