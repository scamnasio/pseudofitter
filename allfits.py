'''
Code to plot up MC fits + all the results of one object
'''

import numpy as np
from matplotlib import pyplot as plt
from numpy import *

def show(path, data_value):
	'''
	*path* - last two branches of path to output array file of object
	'''

	source = np.genfromtxt("/Users/saracamnasio/Research/Projects/UnusuallyRB/2016_Analysis/New_fits_June16/Gwiazda_fits/{0}".format(path), delimiter=' ', dtype = float)
	DATA_LENGTH = source[:,0]
	W = np.arange(.8, 1.5, 0.00001)
	lmin = source[:,5]
	lmax = source[:,6]
	IP = source[:,7]
	if data_value == "lmin":
	
		plt.figure()
		plt.ylim(0.8,1.2)
		plt.xlim(1.15,1.325)
		for n in range(len(DATA_LENGTH)):
			coeff0 = source[n,0]
			coeff1 = source[n,1]
			coeff2 = source[n,2]
			coeff3 = source[n,3]
			coeff4 = source[n,4]
			lmin = source[n,5]
			lmax = source[n,6]
			IP = source[n,7]
			coeffs = [coeff4, coeff3, coeff2, coeff1, coeff0]
			p = np.poly1d(coeffs) 
			plt.plot(W, p(W), color='red', alpha=0.3)
			plt.scatter(lmin, p(lmin), color='blue', alpha = 0.3)
	
		
	elif data_value == 'lmax':
		plt.figure()
		plt.ylim(0.8,1.2)
		plt.xlim(1.15,1.325)
		for n in range(len(DATA_LENGTH)):
			coeff0 = source[n,0]
			coeff1 = source[n,1]
			coeff2 = source[n,2]
			coeff3 = source[n,3]
			coeff4 = source[n,4]
			lmin = source[n,5]
			lmax = source[n,6]
			IP = source[n,7]
			coeffs = [coeff4, coeff3, coeff2, coeff1, coeff0]
			p = np.poly1d(coeffs) 
			plt.plot(W, p(W), color='red', alpha=0.3)
			plt.scatter(lmax, p(lmax), color='green', alpha = 0.7)
			
		
	elif data_value == 'IP':
		plt.figure()
		plt.ylim(0.8,1.2)
		plt.xlim(1.15,1.325)
		for n in range(len(DATA_LENGTH)):
			coeff0 = source[n,0]
			coeff1 = source[n,1]
			coeff2 = source[n,2]
			coeff3 = source[n,3]
			coeff4 = source[n,4]
			lmin = source[n,5]
			lmax = source[n,6]
			IP = source[n,7]
			coeffs = [coeff4, coeff3, coeff2, coeff1, coeff0]
			p = np.poly1d(coeffs) 
			plt.plot(W, p(W), color='red', alpha=0.3)
			plt.scatter(IP, p(IP), color='purple', alpha = 0.3)	