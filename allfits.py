'''
Code to plot up MC fits + all the results of one object
'''

import numpy as np
from matplotlib import pyplot as plt
from numpy import *

def show(path):
	'''
	*path* - path to output array file of object
	'''

	source = np.genfromtxt(path, delimiter=' ', dtype = float)
	DATA_LENGTH = source[:,0]
	for n in range(len(DATA_LENGTH)):
		coeff0 = source[:,0]
		coeff1 = source[:,1]
		coeff2 = source[:,2]
		coeff3 = source[:,3]
		coeff4 = source[:,4]
		lmin = source[:,5]
		lmax = source[:,6]
		IP = source[:,7]
		coeffs = [coeff4, coeff3, coeff2, coeff1, coeff0]
		p = np.poly1d(coeffs) 
		W = np.arange(.8, 1.5, 0.00001)
		plt.plot(W, p(W), color='red', alpha=0.3)